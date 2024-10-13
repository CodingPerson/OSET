import os

import pytorch_transformers
import transformers
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertOnlyMLMHead,BertForMaskedLM
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import utils
from .graph import GraphEncoder
from .hie_losses import HMLC, SupConLoss, mb_sup_loss, consistency_loss, ce_loss
from transformers.modeling_outputs import SequenceClassifierOutput

class BertPoolingLayer(nn.Module):
    def __init__(self, config, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x


class BertOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.constant_(self.dense.weight, 1)
        nn.init.constant_(self.dense.bias, 0)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        nn.init.constant_(self.out_proj.weight, 1)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x, **kwargs):
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NTXent(nn.Module):

    def __init__(self, config, tau=1.):
        super(NTXent, self).__init__()
        self.tau = tau
        self.norm = 1.
        self.transform = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        for name,param in self.transform.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 1)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, labels=None):
        ##x的维度为(2*Batch,268)
        x = self.transform(x)
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        ##将相似度矩阵的对角线上的值设置为很小的值，避免自己好自己进行比较
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss


class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    past_key_values = None
    attentions = None
    cross_attentions = None
    input_embeds = None


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            embedding_weight=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if embedding_weight is not None:
            if len(embedding_weight.size()) == 2:
                embedding_weight = embedding_weight.unsqueeze(-1)
            inputs_embeds = inputs_embeds * embedding_weight
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ##这个地方是新加的
            embedding_weight=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            ##这个地方也是新加的
            embedding_weight=embedding_weight,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, inputs_embeds) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            ##这个也是新加的
            inputs_embeds=inputs_embeds,
        )
class ContrastiveHead(nn.Module):
    def __init__(self, config):
        super(ContrastiveHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(features)
        return x

class AugModel(BertPreTrainedModel):
    def __init__(self, config,cls_loss=True, contrast_loss=True, graph=True, layer=1, data_path=None,
                 lamb=1, lambda_mb=1,lambda_op=1,threshold=0.6, tau=1,memory_bank=False,positive_num=16,end_k=25,knn_num=25,
                 queue_size=32000,random_positive=False,coarse_label_num=0,fine_grained_label_num=0,ood_num=100,p_cutoff=0,
                 open_flag=True,open_lamda=0,augment=True,ova_flag='max'):
        super(AugModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.coarse_label_num=coarse_label_num
        self.fine_grained_label_num=fine_grained_label_num
        self.norm_coef = 0.1
        self.tokenizer = AutoTokenizer.from_pretrained('/mnt/0281fbf2-266c-42ec-80f3-ca8d282c3a4a/contrastive-htc-main (copy)/bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, config.num_labels)
        nn.init.constant_(self.classifier.weight, 1)
        nn.init.constant_(self.classifier.bias, 0)
        self.classifier_liner = ClassificationHead(config, self.num_labels)
        #self.classifier_liner_k = ClassificationHead(config, self.num_labels)
        self.encoder_q = BertModel.from_pretrained('/mnt/0281fbf2-266c-42ec-80f3-ca8d282c3a4a/contrastive-htc-main (copy)/bert-base-uncased')
        #self.encoder_k = BertModel.from_pretrained('/mnt/0281fbf2-266c-42ec-80f3-ca8d282c3a4a/contrastive-htc-main (copy)/bert-base-uncased')
        # self.hi_sup_loss = HMLC(temperature=0.07)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau)
        self.lamb = lamb
        self.lambda_mb=lambda_mb
        self.lambda_op = lambda_op
        self.augment = augment
        params_to_train = ["layer." + str(i) for i in range(0, 12)]
        for name, param in self.encoder_q.named_parameters():
            param.requires_grad_(False)
            for term in params_to_train:
                if term in name:
                    param.requires_grad_(True)

        # self.mlm = BertOnlyMLMHead(config)
        # for param in self.mlm.parameters():
        #     param.requires_grad = False

        self.K = queue_size
        # self.register_buffer("label_queue", torch.randint(0, 2, [self.K,self.fine_grained_label_num]))
        # self.register_buffer("coarse_label_queue", torch.randint(0, 2, [self.K, self.coarse_label_num]))
        # self.register_buffer("feature_queue", torch.randn(self.K, config.hidden_size))
        # self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        # self.register_buffer("ood_feature_queue", torch.randn(self.K, config.hidden_size))
        # self.ood_feature_queue = torch.nn.functional.normalize(self.ood_feature_queue, dim=0)
        # self.register_buffer("ood_queue_ptr", torch.zeros(1, dtype=torch.long))


        self.top_k = knn_num
        self.end_k = end_k
        self.update_num = positive_num
        self.memory_bank = memory_bank
        self.ood_num = ood_num
        self.random_positive = random_positive
        # self.contrastive_liner_q = ContrastiveHead(config)
        # self.contrastive_liner_k = ContrastiveHead(config)
        self.m = 0.999
        self.T = 0.07


        self.mb_classifiers = nn.Linear(config.hidden_size, config.num_labels * 2, bias=False)
        # self.mb_classifiers_k = nn.Linear(config.hidden_size, config.num_labels * 2, bias=False)
        #
        nn.init.xavier_normal_(self.mb_classifiers.weight.data)
        # nn.init.xavier_normal_(self.mb_classifiers_k.weight.data)
        self.p_cutoff=p_cutoff
        self.q_cutoff=0.5
        self.loss_fct = nn.CrossEntropyLoss()
        self.open_flag=open_flag
        self.open_lamda=open_lamda
        self.ova_flag=ova_flag
        # self.init_weights()

    # def init_weights(self):
    #     for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
    #         param_k.data = param_q.data
    #     for param_q, param_k in zip(self.mb_classifiers.parameters(), self.mb_classifiers_k.parameters()):
    #         param_k.data = param_q.data
    #     for param_q, param_k in zip(self.classifier_liner.parameters(), self.classifier_liner_k.parameters()):
    #         param_k.data = param_q.data

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
            coarse_labels=None,
            fine_grained_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            is_train=True,
            return_dict=True,
            positive_samples=None,
            ood_ids=None,
            ood_attention_mask=None,
            ood_labels=None,
            epoch=-1,
    ):


        contrast_mask = None
        contrast_mask_values=None
        pooled_output_ood=None
        contrast_logits=None
        mlm_logits=None
        loss_con_coarse=0
        loss_con_ood=0



        bert_output_q = self.encoder_q(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        pooled_output = bert_output_q[0]

        pooled_output_cls = pooled_output[:, 0, :]


        # liner_q = self.contrastive_liner_q(pooled_output_cls)
        # liner_q = l2norm(liner_q)







        logits_cls = self.classifier_liner(pooled_output_cls)

        ##batch, num_label*2
        logits_mb_cls = self.mb_classifiers(pooled_output_cls)


        fine_grained_targets = fine_grained_labels.to(torch.float32)
        sup_closed_loss = ce_loss(logits_cls.view(-1, self.num_labels), fine_grained_targets,reduction='mean')
        sup_mb_loss = mb_sup_loss(logits_mb_cls, fine_grained_targets,ova_flag=self.ova_flag)

        total_loss = sup_closed_loss + sup_mb_loss



        if is_train==True:
            contrast_mask, contrast_mask_values = self.graph_encoder(pooled_output,
                                                                         attention_mask, labels,
                                                                         lambda x: self.bert.embeddings(x)[0])

            contrast_output = self.encoder_q(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    embedding_weight=contrast_mask,
                )
            contrast_sequence_output = contrast_output[0]
            contrast_sequence_output_cls = contrast_sequence_output[:, 0, :]
            contrast_logits_cls = self.classifier_liner(contrast_sequence_output_cls)
            contrast_logits_mb_cls = self.mb_classifiers(contrast_sequence_output_cls)
            sup_closed_aug_loss = ce_loss(contrast_logits_cls.view(-1, self.num_labels), fine_grained_targets, reduction='mean')
            sup_mb_aug_loss = mb_sup_loss(contrast_logits_mb_cls, fine_grained_targets, ova_flag=self.ova_flag)

            total_loss = total_loss+sup_closed_aug_loss+sup_mb_aug_loss



        return {
            'loss': total_loss,
            'logits': logits_cls,
            'contrast_logits': contrast_logits,
            'contrast_mask':contrast_mask,
            'contrast_mask_values':contrast_mask_values,
            'mlm_logits':mlm_logits,
            'pooled_out':pooled_output_cls,
            'pooled_out_ood': pooled_output_ood,
            # "hie_sup_loss":loss_con_coarse * self.lamb + loss_con_ood*self.lamb
        }
    def evaluate(self,
                 input_ids=None,
                 attention_mask=None,
                 labels=None,
                 return_dict=True,
                 is_train=False
                 ):
        bert_output_q = self.encoder_q(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        pooled_output = bert_output_q[0]
        # pooled_output = self.dropout(pooled_output)
        pooled_output_cls = pooled_output[:, 0, :]
        logits_cls = self.classifier_liner(pooled_output_cls)
        logits_mb_cls = self.mb_classifiers(pooled_output_cls)
        return {
            'logits': logits_cls,
            'logits_mb_cls': logits_mb_cls,
            'pooled_out': pooled_output_cls,
        }