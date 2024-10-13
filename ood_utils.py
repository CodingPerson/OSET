import copy
import os
import random
from collections import defaultdict

import numpy as np
import torch

import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from transformers import BertTokenizer,BertForMaskedLM

import utils

utils.seed_torch(0)
os.environ['PYTHONHASHSEED']=str(0)
special_words=['[CLS]','[SEP]','[PAD]','.','',"'s",',',"``",'--','$','%','?','&','',"''"]
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def has_english_char(token):
    # 遍历字符串中的每个字符
    for char in token:
        # 判断字符是否为英文字母
        if char.isalpha() and char.isascii():
            return True  # 如果找到英文字母，返回True
    return False  # 如果没有找到英文字母，返回False
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
# def fpr_at_recall95(labels, scores):
#     recall_point = 0.95
#     labels = np.asarray(labels)
#     scores = np.asarray(scores)
#     # Sort label-score tuples by the score in descending order.
#     indices = np.argsort(scores)[::-1]    #降序排列
#     sorted_labels = labels[indices]
#     sorted_scores = scores[indices]
#     n_match = sum(sorted_labels)
#     n_thresh = recall_point * n_match
#     thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
#     FP = np.sum(sorted_labels[:thresh_index] == 0)
#     TN = np.sum(sorted_labels[thresh_index:] == 0)
#     return float(FP) / float(FP + TN), (sorted_scores[thresh_index-1] + sorted_scores[thresh_index])/2
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))





def get_auroc(key, prediction,ood_label):
    new_key = np.array([0 if ood_label in l else 1 for l in key ])
    fpr, tpr, threshold_1 = roc_curve(new_key, np.array(prediction))
    # acc_list=[]
    # for thres in threshold_1:
    #     y_pred = np.where(np.array(prediction)>thres,1,0)
    #     acc = accuracy_score(new_key,y_pred)
    #     acc_list.append(acc)

    return auc(fpr, tpr)
def get_aupr(key, prediction,ood_label):
    new_key = np.array([0 if ood_label in l else 1 for l in key ])
    precision_1, recall_1, threshold_1 = precision_recall_curve(new_key, np.array(prediction))
    return auc(recall_1, precision_1)
def get_fpr_95(key, prediction,ood_label):
    new_key = np.array([0 if ood_label in l else 1 for l in key ])
    score= fpr_and_fdr_at_recall(new_key, np.array(prediction))
    return score
def get_f1_score(key, prediction,ood_label):
    new_key = np.array([0 if ood_label in l else 1 for l in key ])
    return f1_score(new_key,np.array(prediction),average='macro')
def prepare_ood(dataloader=None,model=None):
    bank = None
    label_bank = None
    for inputs, label,idxs in dataloader:

        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=label,
            return_dict=True, contrast_flag=True
        )
        pooled = outputs['pooled_out']
        if bank is None:
            bank = pooled.clone().detach()
            label_bank = label.clone().detach()
        else:
            bank_temp = pooled.clone().detach()
            label_bank_temp = label.clone().detach()
            bank = torch.cat([bank_temp, bank], dim=0)
            label_bank = torch.cat([label_bank_temp, label_bank], dim=0)

    norm_bank = F.normalize(bank, dim=-1)
    N, d = bank.size()
    ## can't directly apply to multi-label classes
    # all_classes = list(set(label_bank.tolist()))
    # class_mean = torch.zeros(max(all_classes) + 1, d).cuda()
    # for c in all_classes:
    #     class_mean[c] = (bank[label_bank == c].mean(0))
    # centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
    # class_var = torch.from_numpy(precision).float().cuda()
    #return {'all_classes':all_classes,'class_mean':class_mean,'class_var':class_var}
    return {'norm_bank':norm_bank}

def compute_ood(
        replaced_logits=None,origin_logits_mb=None,origin_pooled_out=None,lof=None
):
    # outputs = self.bert(
    #     input_ids,
    #     attention_mask=attention_mask,
    # )
    # pooled_output = pooled = outputs[1]
    # pooled_output = self.dropout(pooled_output)
    # logits = self.classifier(pooled_output)

    ood_keys = None
    lof_score=None
    lof_negative_score=None
    p = F.softmax(replaced_logits, 1)
    pred_p = p.data.max(1)[1]
    pred_prob = p.data.max(1)[0]
    r = F.softmax(origin_logits_mb.view(origin_logits_mb.size(0), 2, -1), 1)
    tmp_range = torch.arange(0, origin_logits_mb.size(0)).long().cuda()
    unk_score = r[tmp_range, 0, pred_p]



    # softmax_score,_ = torch.sigmoid(replaced_logits).max(-1)

    # maha_score = []
    # for c in ood_prepare_dict['all_classes']:
    #     centered_pooled = replaced_pooled_out - ood_prepare_dict["class_mean"][c].unsqueeze(0)
    #     ms = torch.diag(centered_pooled @ ood_prepare_dict["class_var"] @ centered_pooled.t())
    #     maha_score.append(ms)
    # maha_score = torch.stack(maha_score, dim=-1)
    # maha_score = maha_score.min(-1)[0]
    # maha_score = -maha_score

    #norm_pooled = F.normalize(replaced_pooled_out, dim=-1)
    # cosine_score = norm_pooled @ ood_prepare_dict["norm_bank"].t()
    # cosine_score = cosine_score.max(-1)[0]
    # energy_score,_ = torch.max(replaced_logits, dim=-1)
    # if lof != None:
    #     origin_pooled_out = origin_pooled_out.cpu().detach().numpy()
    #     lof_score = lof.predict(origin_pooled_out)
    #     lof_negative_score = lof.score_samples(origin_pooled_out)
    ood_keys = {
        # 'softmax': softmax_score.tolist(),
        #  'lof': lof_score,
        # 'lof_score':lof_negative_score,
        # # 'cosine': cosine_score.tolist(),
        # 'energy': energy_score.tolist(),
        "softmax":pred_prob.tolist(),
        "ova":unk_score.tolist()
    }
    return ood_keys
def compute_ood_batch(
        replaced_logits=None,origin_logits_mb=None,device="cuda:0"
):
    # outputs = self.bert(
    #     input_ids,
    #     attention_mask=attention_mask,
    # )
    # pooled_output = pooled = outputs[1]
    # pooled_output = self.dropout(pooled_output)
    # logits = self.classifier(pooled_output)

    # ood_keys = None
    # lof_score=None
    # lof_negative_score=None
    # replaced_logits = torch.cat(replaced_logits,dim=0)
    #
    # softmax_score,_ = torch.sigmoid(replaced_logits).max(-1)
    # origin_pooled_out = torch.cat(origin_pooled_out,dim=0)
    p = F.softmax(replaced_logits, 1)
    pred_p = p.data.max(1)[1]
    pred_prob = p.data.max(1)[0]
    r = F.softmax(origin_logits_mb.view(origin_logits_mb.size(0), 2, -1), 1)
    tmp_range = torch.arange(0, origin_logits_mb.size(0)).long().to(device)
    unk_score = r[tmp_range, 0, pred_p]


    # maha_score = []
    # for c in ood_prepare_dict['all_classes']:
    #     centered_pooled = replaced_pooled_out - ood_prepare_dict["class_mean"][c].unsqueeze(0)
    #     ms = torch.diag(centered_pooled @ ood_prepare_dict["class_var"] @ centered_pooled.t())
    #     maha_score.append(ms)
    # maha_score = torch.stack(maha_score, dim=-1)
    # maha_score = maha_score.min(-1)[0]
    # maha_score = -maha_score

    #norm_pooled = F.normalize(replaced_pooled_out, dim=-1)
    # cosine_score = norm_pooled @ ood_prepare_dict["norm_bank"].t()
    # cosine_score = cosine_score.max(-1)[0]
    # energy_score,_ = torch.max(replaced_logits, dim=-1)
    # if lof != None:
    #     origin_pooled_out = origin_pooled_out.cpu().detach().numpy()
    #     lof_score = lof.predict(origin_pooled_out)
    #     lof_negative_score = lof.score_samples(origin_pooled_out)
    # ood_keys = {
    #     'softmax': softmax_score.tolist(),
    #      'lof': lof_score.tolist(),
    #     'lof_score':lof_negative_score.tolist(),
    #     # 'cosine': cosine_score.tolist(),
    #     'energy': energy_score.tolist(),
    # }
    ood_keys = {
        # 'softmax': softmax_score.tolist(),
        #  'lof': lof_score,
        # 'lof_score':lof_negative_score,
        # # 'cosine': cosine_score.tolist(),
        # 'energy': energy_score.tolist(),
        "softmax": pred_prob.tolist(),
        "ova": unk_score.tolist()
    }
    return ood_keys
def compute_ood_threshold(dataloader,model,ood_prepare_dict):
    all_ood_score_dict={'softmax':[],'energy':[]}
    all_label_list=[]
    for inputs, label, idxs in dataloader:
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=label,
            return_dict=True, contrast_flag=True
        )

        logits = outputs['logits']
        pooled_out = outputs['pooled_out']
        for l in label:
            t = []
            for i in range(l.size(0)):
                if l[i].item() == 1:
                    t.append(i)
            if t == []:
                all_label_list.append([logits.shape[-1]])
            else:
                all_label_list.append(t)
        ood_score_dict=compute_ood(logits,pooled_out,ood_prepare_dict)
        for k,v in ood_score_dict.items():
            all_ood_score_dict[k].extend(v)
    ood_label=logits.shape[-1]
    ood_score_outputs = {}
    for key in all_ood_score_dict.keys():

        auroc = get_auroc(all_label_list, all_ood_score_dict[key],ood_label)
        #fpr_95=get_fpr_95(all_label_list, all_ood_score_dict[key],ood_label)

        ood_score_outputs[key + "_auroc"] = auroc
        # ood_score_outputs[key + "_fpr95"] = fpr_95
        # ood_score_outputs[key+"_thresh"] = threshold
    return ood_score_outputs


def find_sublist_positions(main_list, sublist):
    positions = []
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i:i + len(sublist)] == sublist:
            positions.append(i)
    return positions
def _tokenize(seq_list, decoded_texts,tokenizer,length):
    words_list=[]
    sub_words_list=[]
    keys_list=[]
    j = 0
    for seq in seq_list:
        #seq = seq.replace('\n', '')
        words = seq

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub =tokenizer.wordpiece_tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        PAD_count = decoded_texts[j].count('[PAD]')
        if PAD_count != (length-len(sub_words)-1):
            print(PAD_count)
            print((length-len(sub_words)-1))
        assert PAD_count == (length-len(sub_words)-1)
        for i in range(length-len(sub_words)-1):
            word = '[PAD]'
            words.append(word)
            sub = tokenizer.wordpiece_tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        word='[SEP]'
        words.append(word)
        sub = tokenizer.wordpiece_tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
        words_list.append(words)
        sub_words_list.append(sub_words)
        keys_list.append(keys)
        j = j+1

    return words_list, sub_words_list, keys_list
def create_masks(words_list,sub_words_list,keys_list,contrast_mask,contrast_mask_values):
    all_masked_sents_list = []
    all_masked_words_list=[]
    sent_id=0
    for words in words_list:
        currrent_keys = keys_list[sent_id]
        currrent_sub_words = sub_words_list[sent_id]
        current_mask = contrast_mask[sent_id]
        current_mask_value = contrast_mask_values[sent_id]

        mention_idxes = []
        mention_str=words[words.index('[CLS]')+1:words.index('[SEP]')]
        #mention_position=find_sublist_positions(words,mention_str)
        # for mention_pos in mention_position:
        #     mention_idxes.extend(range(mention_pos,mention_pos+len(mention_str)))
        mention_idxes.extend(range(words.index('[CLS]')+1,words.index('[SEP]')))
        word_mask_idx = []
        word_value=[]
        for i in range(len(words)):
            sub_word_mask = current_mask[currrent_keys[i][0]:currrent_keys[i][1]]
            sub_word_value = current_mask_value[currrent_keys[i][0]:currrent_keys[i][1]]
            #and i not in mention_idxes
            if (1 in sub_word_mask)  and words[i] not in special_words and words[i] not in stop_words :
                word_mask_idx.append(i)
                word_value.append(sum(sub_word_value))
        if word_value == []:
            random_number = random.randrange(1,len(words))
            sub_word_value = current_mask_value[currrent_keys[random_number][0]:currrent_keys[random_number][1]]
            word_mask_idx.append(random_number)
            word_value.append(sum(sub_word_value))

        sent_id = sent_id+1

        word_merge_list = list(zip(word_mask_idx,word_value))
        sorted_merged_list = sorted(word_merge_list, key=lambda x: x[1], reverse=True)
        masked_sent_list=[]
        masked_sent = words
        masked_words=[]
        for w_v in sorted_merged_list:
            temp_mask_sent = copy.deepcopy(masked_sent)
            masked_words.append(temp_mask_sent[w_v[0]])
            temp_mask_sent[w_v[0]] = '[MASK]'
            masked_sent_list.append(temp_mask_sent)

        all_masked_sents_list.append(masked_sent_list)
        all_masked_words_list.append(masked_words)

    return all_masked_sents_list,all_masked_words_list
#
def get_sub_word(words,tokenizer):
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
    return sub_words,keys

import re

def is_english_word(word):
    pattern = r'^[a-zA-Z-]+$'
    return re.match(pattern, word) is not None
def get_substitues(mask_prediction_list,predicted_indices,predicted_words,filter_words,max_num,orgin_word):
    substitues=[]
    substitues_idxs=[]
    # word_score_dict={key:0 for key in sub_word_dict.keys()}
    # for word,sub_dict in sub_word_dict.items():
    #     word_score_dict[word] = sum(mask_prediction_list[sub_dict[1]])/len(sub_dict[1])
    num=0
    # sorted_word_score=sorted(word_score_dict.items(), key=lambda d: d[1], reverse=True)
    for i in range(len(predicted_words)):
        word = predicted_words[i]
        if orgin_word == word:
            continue
        # word_id = predicted_indices[i]
        if not is_english_word(word):
            continue
        if not all(ord(c)<128 for c in word):
            continue
        if '##' in word:
            continue
        if 'unused' in word:
            continue
        if word in stop_words:
            continue
        if word in filter_words:
            continue
        if num > max_num:
            break
        substitues.append(word)
        # substitues_idxs.append(word_id)
        num = num+1
    return substitues
def load_out_vocab(vocab_file, tokenizer):
    lines = open(vocab_file, 'r', encoding='utf-8').readlines()
    sub_word_dict = defaultdict(dict)
    for idx, line in enumerate(lines):
        word = line.split('\t')[0].lower()
        sub_word = tokenizer.tokenize(word)
        sub_word_dict[word] = (sub_word, tokenizer.convert_tokens_to_ids(sub_word))

    return sub_word_dict
def tokenize_by_word(words, tokenizer):
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append((index, index + len(sub)))
        index += len(sub)

    return words, sub_words, keys

def fill_masks(params):
    orgin_sent, masked_sents_list, max_num,  idx,model,filter_list,lof,label,mlm_model = params[0], params[1], params[2], params[3], \
    params[4],params[5],params[6],params[7],params[8]

    return_feature = defaultdict()
    return_feature['idx']=idx
    return_feature['change_num'] = 0
    return_feature['mask_words'] = []
    return_feature['change_words'] = []
    return_feature['ood_flag'] = True
    most_gap = 0.0
    temp = copy.deepcopy(masked_sents_list)
    masked_sent = masked_sents_list[0]
    mask_sent_id = 0
    # orgin_sent_str = ' '.join(orgin_sent)
    # orgin_inputs = tokenizer.encode_plus(
    #     orgin_sent_str,
    #     add_special_tokens=False,
    #     return_tensors="pt",
    # )
    origin_ids = tokenizer.convert_tokens_to_ids(orgin_sent)
    origin_ids = torch.from_numpy(np.array(origin_ids)).to(0)
    origin_ids = origin_ids.unsqueeze(0)
    origin_masks = (origin_ids != 0)
    origin_masks = origin_masks.to(0)

    origin_output = model.evaluate(input_ids=origin_ids, attention_mask=origin_masks,
                                    labels=label,
                                    return_dict=True, is_train=False)
    origin_logits = origin_output["logits"]
    origin_pooled_out = origin_output["pooled_out"]
    origin_logits_mb = origin_output["logits_mb_cls"]
    origin_ood_score_dict = compute_ood(origin_logits,origin_logits_mb,origin_pooled_out=origin_pooled_out,lof=lof)
    origin_softmax_score = origin_ood_score_dict['softmax'][0]
    origin_ova_score = origin_ood_score_dict['ova'][0]
    # raw_predictions = origin_output['raw_mlm_logits']
    last_mask_sent = orgin_sent
    for _ in range(len(masked_sents_list)):
        masked_sent_copy = list(copy.deepcopy(masked_sent))
        for i in range(len(masked_sent_copy) - 1, -1,-1):  # 同样不能用正序循环，for i in range(0,len(alist)), 用了remove()之后，len(alist)是动态的，会产生列表下标越界错误
            if masked_sent_copy[i] == '[PAD]':
                masked_sent_copy.remove('[PAD]')
        SEP_id = masked_sent_copy.index('[SEP]')
        mention_name_str = masked_sent_copy[0:SEP_id]
        # masked_sent_copy = ['[CLS]']+masked_sent_copy[SEP_id+1:]
        replaced_sent_ids = tokenizer.convert_tokens_to_ids(masked_sent_copy)
        replaced_sent_tensor = torch.tensor([replaced_sent_ids]).cuda()
        mask_idxs = [i for i, t in enumerate(masked_sent_copy) if t == '[MASK]']
        mask_id_idxs = [i for i, t in enumerate(replaced_sent_ids) if t == tokenizer.mask_token_id]
        mask_idx = mask_idxs[0]
        orgin_word = last_mask_sent[mask_idx]
        indices=[]
        if mask_idxs[0] < SEP_id or orgin_word in mention_name_str:
            indices = [index for index,word in enumerate(masked_sent_copy) if word == orgin_word]
        if len(mask_id_idxs) != 1:
            print(masked_sent)
        #assert len(mask_idxs) == len(mask_id_idxs) == 1

        mask_id_idx = mask_id_idxs[0]

        with torch.no_grad():
            predictions = mlm_model(replaced_sent_tensor)
        mask_prediction_list = predictions[0][0][mask_id_idx]
        #raw_mask_prediction_list = raw_predictions[0][mask_id_idx]
        predicted_indices = torch.argsort(-mask_prediction_list, dim=-1)
       # raw_predicted_indices = torch.argsort(-raw_mask_prediction_list, dim=-1)
        predicted_words = tokenizer.convert_ids_to_tokens(predicted_indices.tolist())
       # raw_predicted_words = tokenizer.convert_ids_to_tokens(raw_predicted_indices.tolist())
        substitues= get_substitues(mask_prediction_list,predicted_indices, predicted_words,filter_list,
                                                     max_num=max_num,orgin_word=orgin_word)
        # raw_substitues, raw_substitues_idxs = get_substitues(raw_predicted_indices, raw_predicted_words, filter_list,
        #                                              max_num=max_num)
        candidate = None
        replaced_logits_list=[]
        replaced_logits_mb_list=[]
        replaced_pooled_out_list=[]
        replaced_ids_list=[]
        msk_list=[]
        for i in range(len(substitues)):
            substitue_ = substitues[i]
            masked_sent_temp = masked_sent.copy()
            masked_sent_temp = np.array(masked_sent_temp)
            if indices != []:
                masked_sent_temp[indices]=substitue_
            masked_sent_temp[mask_idx] = substitue_
            msk_list.append(masked_sent_temp)
            fi_masked_sent_ids = tokenizer.convert_tokens_to_ids(masked_sent_temp)
            replaced_ids_list.append(copy.deepcopy(fi_masked_sent_ids))
            fi_masked_sent_ids = torch.from_numpy(np.array(fi_masked_sent_ids)).to(0)
            fi_masked_sent_ids = fi_masked_sent_ids.unsqueeze(0)
            fi_masked_sent_masks = (fi_masked_sent_ids != 0)
            fi_masked_sent_masks = fi_masked_sent_masks.to(0)
            # replaced_sent = ' '.join(masked_sent_temp)
            # replaced_inputs = tokenizer(
            #     replaced_sent,
            #     return_tensors="pt",
            # ).to(0)
            replaced_output = model.evaluate(input_ids=fi_masked_sent_ids,
                                    attention_mask=fi_masked_sent_masks,
                                    labels=label,
                                    return_dict=True,is_train=False)
            replaced_logits = replaced_output['logits']
            replaced_logits_mb = replaced_output['logits_mb_cls']
            replaced_logits_copy = copy.deepcopy(replaced_logits)
            replaced_logits_mb_copy = copy.deepcopy(replaced_logits_mb)
            replaced_pooled_out = replaced_output['pooled_out']
            replaced_pooled_out_copy = copy.deepcopy(replaced_pooled_out)
            replaced_logits_list.append(replaced_logits_copy)
            replaced_logits_mb_list.append(replaced_logits_mb_copy)
            replaced_pooled_out_list.append(replaced_pooled_out_copy)
        ###这个地方来计算OOD分数
        ood_score_dict = compute_ood_batch(torch.cat(replaced_logits_list,dim=0),origin_logits_mb=torch.cat(replaced_logits_mb_list,dim=0))
         ##我们获取其中最适合的单词进行替换
        min_ova_score = max(ood_score_dict['ova'])
        min_ova_score_index = ood_score_dict['ova'].index(min_ova_score)
        most_gap = min_ova_score-origin_ova_score
        candidate =  substitues[min_ova_score_index]

        if ood_score_dict['ova'][min_ova_score_index] > 0.5:
            return_feature['change_num'] += 1
            return_feature['mask_words'].append(orgin_word)
            return_feature['change_words'].append(substitues[min_ova_score_index])
            return_feature['orgin_sent'] = orgin_sent
            return_feature['replaced_sent'] = msk_list[min_ova_score_index]
            return return_feature

        ##这地方如果candidate==None的话，说明前面没有拿到候选，即使most_gap大于0也没用，因为可能是上一轮替换留下的gap
        if most_gap > 0 and candidate != None:
            return_feature['change_num'] += 1
            return_feature['mask_words'].append(orgin_word)
            return_feature['change_words'].append(candidate)
            return_feature['most_gap'] = most_gap

            ##对下一轮需要替换的句子进行更新,我们这轮替换完的单词要更新进去
            mask_sent_id = mask_sent_id + 1
            if mask_sent_id == len(masked_sents_list):
                masked_sent = np.array(masked_sent)
                if indices != []:
                    masked_sent[indices] = candidate
                masked_sent[mask_idx] = candidate
                return_feature['ood_flag'] = False
                return_feature['orgin_sent'] = orgin_sent
                return_feature['replaced_sent'] = masked_sent
                return return_feature
            else:
                masked_sent = np.array(masked_sent)
                if indices != []:
                    masked_sent[indices] = candidate
                masked_sent[mask_idx] = candidate
                next_mask_ids = [i for i, t in enumerate(masked_sents_list[mask_sent_id]) if
                                 t == '[MASK]']
                assert len(next_mask_ids) == 1
                next_mask_id = next_mask_ids[0]
                last_mask_sent = copy.deepcopy(masked_sent)
                masked_sent[next_mask_id] = '[MASK]'
        else:
            mask_sent_id = mask_sent_id + 1
            masked_sent = np.array(masked_sent)
            if mask_sent_id == len(masked_sents_list):
                if indices != []:
                    masked_sent[indices] = last_mask_sent[mask_idx]
                masked_sent[mask_idx] = last_mask_sent[mask_idx]
                return_feature['ood_flag'] = False
                return_feature['orgin_sent'] = orgin_sent
                return_feature['replaced_sent'] = masked_sent
                return return_feature
            else:
                if indices != []:
                    masked_sent[indices] = orgin_sent[mask_idx]
                masked_sent[mask_idx] = orgin_sent[mask_idx]
                next_mask_ids = [i for i, t in enumerate(masked_sents_list[mask_sent_id]) if
                                 t == '[MASK]']
                assert len(next_mask_ids) == 1
                next_mask_id = next_mask_ids[0]
                last_mask_sent = copy.deepcopy(masked_sent)
                masked_sent[next_mask_id] = '[MASK]'
    return return_feature
# def fill_masks_batch(batch_task):
#     origin_logits_list=[]
#     origin_pooled_out_list=[]
#     raw_predictions_list=[]
#     all_masked_sents_list=[]
#     all_return_feature=[]
#     orgin_sent_list=[]
#     all_most_gap=[]
#     lof=None
#     filter_list=None
#     device=None
#     max_num=None
#     model=None
#     return_flag=[0]*len(batch_task)
#     for params in batch_task:
#         orgin_sent, masked_sents_list, max_num,  idx,model,filter_list,lof,device = params[0], params[1], params[2], params[3], \
#         params[4],params[5],params[6],params[7]
#         all_masked_sents_list.append(masked_sents_list)
#         orgin_sent_list.append(orgin_sent)
#         return_feature = defaultdict()
#         return_feature['idx']=idx
#         return_feature['change_num'] = 0
#         return_feature['mask_words'] = []
#         return_feature['change_words'] = []
#         return_feature['ood_flag'] = True
#         all_return_feature.append(return_feature)
#         most_gap = 0.0
#         all_most_gap.append(most_gap)
#         temp = copy.deepcopy(masked_sents_list)
#
#         mask_sent_id = 0
#         orgin_sent_str = ' '.join(orgin_sent)
#         orgin_inputs = tokenizer.encode_plus(
#             orgin_sent_str,
#             add_special_tokens=False,
#             return_tensors="pt",
#         )
#         orgin_inputs = orgin_inputs.to(device)
#         origin_output = model(input_ids=orgin_inputs["input_ids"],
#                               attention_mask=orgin_inputs["attention_mask"],
#                               return_dict=True)
#         raw_predictions = origin_output['raw_mlm_logits']
#         raw_predictions_list.append(raw_predictions)
#         origin_logits = origin_output["logits"]
#         origin_pooled_out = origin_output["pooled_out"]
#         origin_logits_list.append(origin_logits)
#         origin_pooled_out_list.append(origin_pooled_out)
#     origin_ood_score_dict = compute_ood_batch(origin_logits_list,origin_logits_mb=origin_pooled_out_list)
#     origin_ood_score_list = origin_ood_score_dict['lof_score']
#     # predictions = origin_output['mlm_logits']
#     origin_lof_score_list = origin_ood_score_dict['lof']
#     for m in range(len(all_masked_sents_list)):
#         masked_sent = all_masked_sents_list[m][0]
#         orgin_sent = orgin_sent_list[m]
#         raw_predictions = raw_predictions_list[m]
#         return_feature=all_return_feature[m]
#         if return_flag[m] == 1:
#             continue
#         replaced_pooled_out_list=[]
#         replaced_logits_list=[]
#         for _ in range(len(all_masked_sents_list[m])):
#
#             replaced_sent_str = ' '.join(masked_sent)
#
#             inputs = tokenizer.encode_plus(
#                 replaced_sent_str,
#                 add_special_tokens=False,
#                 return_tensors="pt",
#             )
#             mask_idxs = [i for i, t in enumerate(masked_sent) if t == '[MASK]']
#             mask_id_idxs = [i for i, t in enumerate(inputs['input_ids'][0]) if t == tokenizer.mask_token_id]
#             if len(mask_id_idxs) != 1:
#                 print(masked_sent)
#
#         #assert len(mask_idxs) == len(mask_id_idxs) == 1
#             mask_idx = mask_idxs[0]
#             mask_id_idx = mask_id_idxs[0]
#             orgin_word = orgin_sent[mask_idx]
#             # mask_prediction_list = predictions[0][mask_id_idx]
#             raw_mask_prediction_list = raw_predictions[0][mask_id_idx]
#             # predicted_indices = torch.argsort(mask_prediction_list, dim=-1)
#             raw_predicted_indices = torch.argsort(raw_mask_prediction_list, dim=-1)
#             # predicted_words = tokenizer.convert_ids_to_tokens(predicted_indices.tolist())
#             raw_predicted_words = tokenizer.convert_ids_to_tokens(raw_predicted_indices.tolist())
#             # substitues, substitues_idxs = get_substitues(predicted_indices, predicted_words,filter_list,
#             #                                              max_num=max_num)
#             raw_substitues, raw_substitues_idxs = get_substitues(raw_predicted_indices, raw_predicted_words, filter_list,
#                                                          max_num=max_num)
#             candidate = None
#
#             for i in range(len(raw_substitues)):
#                 substitue_ = raw_substitues[i]
#                 masked_sent[mask_idx] = substitue_
#                 replaced_sent = ' '.join(masked_sent)
#                 replaced_inputs = tokenizer(
#                     replaced_sent,
#                     return_tensors="pt",
#                 ).to(device)
#                 replaced_output = model(input_ids=replaced_inputs["input_ids"],
#                                         attention_mask=replaced_inputs["attention_mask"],
#                                         return_dict=True)
#                 replaced_logits = replaced_output['logits']
#                 replaced_pooled_out = replaced_output['pooled_out']
#
#              ###这个地方来计算OOD分数
#             ood_score_dict = compute_ood(replaced_logits,replaced_pooled_out,lof=lof)
#             choose_ood_score = ood_score_dict['lof_score'][0]
#             ##ood生成成功了，可以结束替换了
#             if ood_score_dict['lof'][0] == -1 and origin_ood_score_dict['lof'][0] != -1:
#                 return_feature['change_num'] += 1
#                 return_feature['mask_words'].append(orgin_word)
#                 return_feature['change_words'].append(substitue_)
#                 return_feature['orgin_sent'] = orgin_sent
#                 return_feature['replaced_sent'] = masked_sent
#                 return return_feature
#             else:
#                 gap = origin_ood_score-choose_ood_score
#                 if gap > most_gap:
#                     most_gap = gap
#                     candidate = substitue_
#         ##这地方如果candidate==None的话，说明前面没有拿到候选，即使most_gap大于0也没用，因为可能是上一轮替换留下的gap
#         if most_gap > 0 and candidate != None:
#             return_feature['change_num'] += 1
#             return_feature['mask_words'].append(orgin_word)
#             return_feature['change_words'].append(candidate)
#             return_feature['most_gap'] = most_gap
#
#             ##对下一轮需要替换的句子进行更新,我们这轮替换完的单词要更新进去
#             mask_sent_id = mask_sent_id + 1
#             if mask_sent_id == len(masked_sents_list):
#                 masked_sent[mask_idx] = candidate
#                 return_feature['ood_flag'] = False
#                 return_feature['orgin_sent'] = orgin_sent
#                 return_feature['replaced_sent'] = masked_sent
#                 return return_feature
#             else:
#                 masked_sent[mask_idx] = candidate
#                 next_mask_ids = [i for i, t in enumerate(masked_sents_list[mask_sent_id]) if
#                                  t == '[MASK]']
#                 assert len(next_mask_ids) == 1
#                 next_mask_id = next_mask_ids[0]
#                 masked_sent[next_mask_id] = '[MASK]'
#         else:
#             mask_sent_id = mask_sent_id + 1
#             if mask_sent_id == len(masked_sents_list):
#                 masked_sent[mask_idx] = orgin_sent[mask_idx]
#                 return_feature['ood_flag'] = False
#                 return_feature['orgin_sent'] = orgin_sent
#                 return_feature['replaced_sent'] = masked_sent
#                 return return_feature
#             else:
#                 masked_sent[mask_idx] = orgin_sent[mask_idx]
#                 next_mask_ids = [i for i, t in enumerate(masked_sents_list[mask_sent_id]) if
#                                  t == '[MASK]']
#                 assert len(next_mask_ids) == 1
#                 next_mask_id = next_mask_ids[0]
#                 masked_sent[next_mask_id] = '[MASK]'
#     return return_feature



#     return return_feature




