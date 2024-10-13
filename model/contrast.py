import copy
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
# from transformers.utils import ModelOutput

import utils
from .contrast_moco import ClassificationHead
from .graph import GraphEncoder
from .hie_losses import HMLC, SupConLoss
from typing import Optional, Tuple

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
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    inputs_embeds=None


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
def pair_cosine_similarity(x, x_adv, eps=1e-8):
    n = x.norm(p=2, dim=1, keepdim=True)
    n_adv = x_adv.norm(p=2, dim=1, keepdim=True)
    return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_adv @ x_adv.t()) / (n_adv * n_adv.t()).clamp(min=eps), (x @ x_adv.t()) / (n * n_adv.t()).clamp(min=eps)


def nt_xent(x, x_adv, mask, cuda=True, t=0.1):
    x, x_adv, x_c = pair_cosine_similarity(x, x_adv)
    x = torch.exp(x / t)
    x_adv = torch.exp(x_adv / t)
    x_c = torch.exp(x_c / t)
    mask_count = mask.sum(1)
    mask_reverse = (~(mask.bool())).long()
    if cuda:
        dis = (x * (mask - torch.eye(x.size(0)).long().cuda()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long().cuda()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    else:
        dis = (x * (mask - torch.eye(x.size(0)).long()) + x_c * mask) / (x.sum(1) + x_c.sum(1) - torch.exp(torch.tensor(1 / t))) + mask_reverse
        dis_adv = (x_adv * (mask - torch.eye(x.size(0)).long()) + x_c.T * mask) / (x_adv.sum(1) + x_c.sum(0) - torch.exp(torch.tensor(1 / t))) + mask_reverse
    loss = (torch.log(dis).sum(1) + torch.log(dis_adv).sum(1)) / mask_count
    return -loss.mean()
def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def contrastive_loss_fc(temp, batch_emb, labels):
    #labels = labels.view(-1, 1)
    batch_size = batch_emb.shape[0]
    mask = torch.mm(labels, labels.T).bool().long()
    #mask = torch.eq(labels, labels.T).float()
    #mask=torch.any(torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)),dim=-1).float()
    norm_emb = F.normalize(batch_emb, dim=1, p=2)
    # compute logits
    dot_contrast = torch.div(torch.matmul(norm_emb, norm_emb.T), temp)
    # for numerical stability
    logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)  # _返回索引
    logits = dot_contrast - logits_max.detach()
    # 索引应该保证设备相同
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(batch_emb.device), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mask_sum = mask.sum(1)
    # 防止出现NAN
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
    mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_sum
    return mean_log_prob_pos.mean()
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    # cosine_sim = sim_matrix(embedding, embedding)
    # n = cosine_sim.shape[0]
    # dis = cosine_sim.masked_select(~torch.eye(n, dtype=bool).cuda()).view(n, n - 1)
    #
    # # apply temperature to elements
    # dis = dis / temp
    # cosine_sim = cosine_sim / temp
    # # apply exp to elements
    # dis = torch.exp(dis)
    # cosine_sim = torch.exp(cosine_sim)
    #
    # # calculate row sum
    # row_sum = torch.sum(dis, -1)
    #
    # unique_labels, inverse_indices, unique_label_counts = torch.unique(label, dim=0,sorted=False, return_inverse=True, return_counts=True)
    # # calculate outer sum
    # contrastive_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    # for i in range(n):
    #     n_i = unique_label_counts[inverse_indices[i]] - 1
    #     inner_sum = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    #     # calculate inner sum
    #     for j in range(n):
    #         if torch.equal(label[i] ,label[j]) and i != j:
    #             inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
    #     if n_i != 0:
    #         contrastive_loss += (inner_sum / (-n_i))
    #return contrastive_loss


#以bert为基础的模型
class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, cls_loss=True, contrast_loss=True, graph=False, layer=1, data_path=None,
                 multi_label=True, lamb=1, threshold=0.01, tau=1):
    # def __init__(self, config):
        super(ContrastModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.hidden_dim = 128

        self.norm_coef = 0.1
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, config.num_labels)
        # self.logistic = nn.Linear(1,2)
        nn.init.constant_(self.classifier.weight, 1)
        nn.init.constant_(self.classifier.bias, 0)
        # nn.init.constant_(self.logistic.weight, 1)
        # nn.init.constant_(self.logistic.bias, 0)
        # self.bert = BertModel(config)
        self.bert = BertModel(config)

        # for name, param in self.bert.named_parameters():
        #     if name.startswith('pooler'):
        #         continue
        #     else:
        #         param.requires_grad_(False)

        #self.bert.embeddings.word_embeddings.weight.requires_grad = True
        self.pooler = BertPoolingLayer(config, 'cls')
        self.contrastive_lossfct = NTXent(config)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.token_classifier = BertOutputLayer(config)



        self.num_layers=1

        # self.rnn = nn.GRU(input_size=768, hidden_size=self.hidden_dim,
        #               num_layers=self.num_layers,
        #               batch_first=True, bidirectional=True).to(0)

        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau)
        for name,param in self.graph_encoder.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 1)
            if 'bias' in name:
                nn.init.constant_(param, 0)
        self.lamb = lamb
        #
        # #self.init_weights()
        self.multi_label = multi_label
        self.mlm = BertOnlyMLMHead(config)
        for param in self.mlm.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            contrast_flag=False,
            mlm_flag=False,
            ood_ids=None,
            ood_attention_mask=None,
            ids_ood=None,
            ood_lamda=None,
            id_labels=None,
            bi_flag=None,
            ood_flag=None,
            ood_labels=None,
            train_flag=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None
        contrast_mask_values=None
        pooled_output_ood=None
        loss=0
        contrast_logits=None
        ood_loss=0
        mlm_logits=None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            #用于将某些层的计算无效化
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # #是否返回中间层的attention
            # output_attentions=output_attentions,
            # #是否返回中间层的embedding
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # embedding_weight=None,

        )
        pooled_output = outputs[0]
        seq_embed = self.dropout(pooled_output)
        #seq_embed = torch.tensor(seq_embed,requires_grad=True)
        #seq_embed = seq_embed.clone().detach().requires_grad_(True).float()

        if mlm_flag == True:
            mlm_logits = self.mlm(pooled_output)

        if train_flag==True:

        #pooled_output = self.dropout(self.pooler(pooled_output))

            #seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            # _, ht = self.pooler(pooled_output)
            # ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            ht = self.pooler(seq_embed)
            logits = self.classifier(ht)
            loss_fct = nn.BCEWithLogitsLoss()
            target = labels.to(torch.float32)
            bi_loss = loss_fct(logits.view(-1, self.num_labels), target)
            if ood_ids != None and ood_flag == True:
                ood_outputs = self.bert(
                    ood_ids,
                    attention_mask=ood_attention_mask,
                )
                ood_pooled_output = ood_outputs[0]
                ood_hv = self.pooler(self.dropout(ood_pooled_output))
                ood_logits = self.classifier(ood_hv)
                # loss_fct = nn.BCEWithLogitsLoss()
                # target = ood_labels.to(torch.float32)
                # ood_loss = loss_fct(ood_logits.view(-1, self.num_labels), target)

                #ht = torch.cat([ht,ood_hv],dim=0)
                #pooled_output = torch.cat([pooled_output,ood_pooled_output],dim=0)

                #labels = torch.cat([labels,ood_labels],dim=0)
                pairwise_conf_diffs = torch.max(torch.sigmoid(ood_logits), dim=-1)[0].unsqueeze(
                    1
                ) - torch.max(torch.sigmoid(logits), dim=-1)[0].unsqueeze(0)
                pos_pairwise_conf_diffs = torch.nn.functional.relu(pairwise_conf_diffs)
                #pos_pairwise_conf_diffs_max,_ = pos_pairwise_conf_diffs.max(dim=-1)
                ccl_loss = pos_pairwise_conf_diffs.mean()
                ood_loss=ccl_loss





            #
            contrastive_loss = None
            contrast_logits = None
            loss=0




            # return sup_cont_loss

            if bi_flag == True:
                loss=bi_loss+ood_loss
            if mlm_flag == True:
                # seq_embed.retain_grad()  # we need to get gradient w.r.t embeddings
                # bi_loss.backward(retain_graph=True)
                # unnormalized_noise = seq_embed.grad.detach_()
                # for p in self.parameters():
                #     if p.grad is not None:
                #         p.grad.detach_()
                #         p.grad.zero_()
                # norm = unnormalized_noise.norm(p=2, dim=-1)
                # normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid NaN
                # noise_embedding = seq_embed + self.norm_coef * normalized_noise
                # # _, h_adv = self.rnn(noise_embedding, None)
                # # h_adv = torch.cat((h_adv[0].squeeze(0), h_adv[1].squeeze(0)), dim=1)
                # h_adv=self.pooler(noise_embedding)
                #label_mask = torch.mm(labels, labels.T).bool().long()
                #label_mask = torch.any(torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)), dim=-1).bool().long()
                #sup_cont_loss = nt_xent(ht, ht, label_mask, cuda=True)

                sup_cont_loss=contrastive_loss_fc(0.1,ht,labels)

                loss = sup_cont_loss
        else:

            seq_embed = seq_embed.clone().detach().requires_grad_(True).float()
            ht=self.pooler(seq_embed)
            # _, ht = self.rnn(seq_embed)
            # ht = torch.cat((ht[0].squeeze(0), ht[1].squeeze(0)), dim=1)
            logits = self.classifier(ht)
        # if labels is not None:
        #
        #
        #
        #     #这个地方表示有OOD数据输入
        #
        #     # if ood_ids != None and ood_flag == True:
        #     #
        #     #
        #     #
        #     #
        #     #     # all_labels = torch.cat([labels,ood_labels],dim=0)
        #     #     # pooled_output = torch.cat([pooled_output, ood_pooled_output], dim=0)
        #     #     # id_dist = ((pooled_output.unsqueeze(1) - pooled_output.unsqueeze(0)) ** 2).mean(-1)
        #     #     #
        #     #     # #pooled_output_all = torch.cat([pooled_output,pooled_output_ood],dim=0)
        #     #     # dist = ((pooled_output.unsqueeze(1) - pooled_output.unsqueeze(0)) ** 2).mean(-1)
        #     #     # mask = torch.any(torch.eq(all_labels.unsqueeze(1),all_labels.unsqueeze(0)),dim=-1).float()
        #     #     # mask = mask - torch.diag(torch.diag(mask))
        #     #     # neg_mask = torch.any(torch.ne(all_labels.unsqueeze(1),all_labels.unsqueeze(0)),dim=-1).float()
        #     #     # max_dist = (id_dist * mask).max()
        #     #     # cos_loss = ((id_dist * mask).sum(-1) / (mask.sum(-1) + 1e-3)).mean() + ((F.relu(max_dist - dist) * neg_mask).sum(
        #     #     #     -1) / (neg_mask.sum(-1) + 1e-3)).mean()
        #     #     # # cos_loss = ((F.relu(max_dist - dist) * neg_mask).sum(
        #     #     # #     -1) / (neg_mask.sum(-1) + 1e-3)).mean()
        #     #     #
        #     #     # ood_loss += ood_lamda * cos_loss
        #     #     # loss += ood_lamda * cos_loss
        #     #     pooled_output_all = torch.cat([pooled_output, ood_pooled_output], dim=0)
        #     #     labels = torch.cat([labels, ood_labels], dim=0)
        #     #     contrastive_l = contrastive_loss_fc(0.3, pooled_output_all, labels)
        #     #     loss = (ood_lamda * contrastive_l) + (1 - ood_lamda) * (bi_loss)
        #     # else:
        #     #     loss = bi_loss
        #
        #
        #
        #
            if contrast_flag!= False:
                ##outputs['inputs_embeds']表示文本的初始嵌入
                ##contrast_mask是一堆0/1矩阵
                contrast_mask,contrast_mask_values = self.graph_encoder(outputs['inputs_embeds'],
                                                   attention_mask, labels, lambda x: self.bert.embeddings(x)[0])

                contrast_output = self.bert(
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
                contrast_sequence_output = self.dropout(self.pooler(contrast_output[0]))
                contrast_logits = self.classifier(contrast_sequence_output)
                # contrastive_loss = self.contrastive_lossfct(
                #     torch.cat([pooled_output, contrast_sequence_output], dim=0), )

            #     loss += loss_fct(contrast_logits.view(-1, self.num_labels), target) \
            #
            # if contrastive_loss is not None and self.contrast_loss:
            #     loss += contrastive_loss * self.lamb


        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
            'contrast_mask':contrast_mask,
            'contrast_mask_values':contrast_mask_values,
            'mlm_logits':mlm_logits,
            'pooled_out':ht,
            'pooled_out_ood': pooled_output_ood,
            "energy_loss":ood_loss
        }
class BertForMasked(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMasked, self).__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.token_classifier = BertOutputLayer(config)
        self.init_weights()
        self.mlm = BertOnlyMLMHead(config)
        for param in self.mlm.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            return_dict=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            #用于将某些层的计算无效化
            head_mask=None,
            inputs_embeds=inputs_embeds,
            #是否返回中间层的attention
            output_attentions=None,
            #是否返回中间层的embedding
            output_hidden_states=None,
            return_dict=return_dict,
            embedding_weight=None,

        )

        pooled_output = outputs[0]

        mlm_logits = self.mlm(pooled_output)

        pooled_output = (self.pooler(pooled_output))

        logits = self.classifier(pooled_output)



        return {
            'logits': logits,
            'mlm_logits':mlm_logits,
            'pooled_out':pooled_output
        }

class BaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        nn.init.constant_(self.classifier.weight, 1)
        nn.init.constant_(self.classifier.bias, 0)
        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.bert_mlm = BertForMaskedLM(config)
        self.mlm = BertOnlyMLMHead(config)
        for param in self.mlm.parameters():
            param.requires_grad = False
        for param in self.bert_mlm.parameters():
            param.requires_grad = False



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            contrast_flag=False,
            mlm_flag=False,
            ood_ids=None,
            ood_attention_mask=None,
            ood_lamda=None

    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        mlm_outputs = self.bert_mlm(
            input_ids,
            attention_mask=attention_mask,
        )
        raw_mlm_out = mlm_outputs['logits']
        pooled_output = outputs[0]

        mlm_logits = self.mlm(pooled_output)
        pooled_output = (self.pooler(pooled_output))


        loss = 0


        logits = self.classifier(pooled_output)

        if labels is not None:

            loss_fct = nn.BCEWithLogitsLoss()
            target = labels.to(torch.float32)

            loss += loss_fct(logits.view(-1, self.num_labels), target)

        return {
            'loss': loss,
            'logits': logits,
            'mlm_logits': mlm_logits,
            'pooled_out':pooled_output,
            'raw_mlm_logits':raw_mlm_out
        }

class GenerateModel(BertPreTrainedModel):
    def __init__(self, config, cls_loss=True, contrast_loss=True, graph=True, layer=1, data_path=None,
                 lamb=1, threshold=0.6, tau=1):
        super(GenerateModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.norm_coef = 0.1
        self.tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/DM_Group/CHEN/contrastive-htc-main (copy)/bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, config.num_labels)
        #self.classifier_liner = ClassificationHead(config, self.num_labels)
        nn.init.constant_(self.classifier.weight, 1)
        nn.init.constant_(self.classifier.bias, 0)
        self.bert = BertModel.from_pretrained('/home/ubuntu/DM_Group/CHEN/contrastive-htc-main (copy)/bert-base-uncased')
        self.contrastive_lossfct = NTXent(config)
        self.hi_sup_loss = HMLC(temperature=0.07)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau)
        for name,param in self.graph_encoder.named_parameters():
            if 'weight' in name:
                nn.init.constant_(param, 1)
            if 'bias' in name:
                nn.init.constant_(param, 0)
        self.lamb = lamb

        self.mlm = BertOnlyMLMHead(config)
        for param in self.mlm.parameters():
            param.requires_grad = False

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
            return_dict=None,
            positive_samples=None,
            ood_ids=None,
            ood_attention_mask=None,
            ood_labels=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None
        contrast_mask_values=None
        pooled_output_ood=None
        loss=0
        contrast_logits=None
        ood_loss=0
        mlm_logits=None
        contrastive_loss=None
        hie_sup_loss=0
        input_ids_copy = copy.deepcopy(input_ids)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        pooled_output_cls = pooled_output[:, 0, :]

        #mlm_logits = self.mlm(pooled_output)


        logits = self.classifier(pooled_output_cls)
        loss_fct = nn.BCEWithLogitsLoss()
        target = labels.to(torch.float32)
        bi_loss = loss_fct(logits.view(-1, self.num_labels), target)
        if self.cls_loss == True:
            loss += bi_loss


        if labels is not None:


            if is_train== True and self.contrast_loss == True:
                contrast_mask,contrast_mask_values = self.graph_encoder(pooled_output,
                                                   attention_mask, labels, lambda x: self.bert.embeddings(x)[0])

                contrast_output = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    embedding_weight=contrast_mask
                )
                contrast_output = self.dropout(contrast_output[0])
                contrast_output_cls = contrast_output[:, 0, :]
                contrast_sequence_output = contrast_output_cls
                contrast_logits = self.classifier(contrast_sequence_output)
                contrastive_loss = self.contrastive_lossfct(
                    torch.cat([pooled_output_cls, contrast_sequence_output], dim=0), )
                pooled_output_cls = F.normalize(pooled_output_cls, dim=1)
                contrast_sequence_output = F.normalize(contrast_sequence_output, dim=1)
                contrast_input = torch.cat([pooled_output_cls.unsqueeze(1), contrast_sequence_output.unsqueeze(1)], dim=1)
                hie_sup_loss = self.hi_sup_loss(contrast_input,labels,coarse_labels,fine_grained_labels)
                #loss += loss_fct(contrast_logits.view(-1, self.num_labels), target)
            if contrastive_loss is not None and self.contrast_loss:
                loss +=hie_sup_loss * self.lamb


        return {
            'loss': loss,
            'logits': logits,
            'contrast_logits': contrast_logits,
            'contrast_mask':contrast_mask,
            'contrast_mask_values':contrast_mask_values,
            'mlm_logits':mlm_logits,
            'pooled_out':pooled_output_cls,
            'pooled_out_ood': pooled_output_ood,
            "hie_sup_loss":hie_sup_loss
        }