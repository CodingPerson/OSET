#!/usr/bin/env python
# coding:utf-8
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance, GraphicalLasso
from sklearn.metrics import f1_score, multilabel_confusion_matrix, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import utils
from ood_utils import get_auroc, get_fpr_95, compute_ood, get_f1_score, get_aupr
from sklearn.metrics import confusion_matrix

def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]

    for idx in range(n_class):
        bi_cm = cm[idx]
        tp = bi_cm[1, 1]
        fp = bi_cm[0, 1]
        fn = bi_cm[1, 0]

        # 计算Precision和Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        rs.append(recall * 100)
        ps.append(precision * 100)
        fs.append(f1 * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Known'] = f_seen
    result['Open'] = f_unseen
    result['F1-score'] = f

    return result
def get_score(cm):
    # metric is same as SCL' metric
    fs = []
    ps = []
    rs = []
    n_class = cm.shape[0]
    correct = []
    total = []
    for idx in range(n_class):
        TP = cm[idx][idx]
        correct.append(TP)
        total.append(cm[idx].sum())
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f * 100)
        ps.append(p * 100)
        rs.append(r * 100)

    f = np.mean(fs).round(2)
    p_seen = np.mean(ps[:-1]).round(2)
    r_seen = np.mean(rs[:-1]).round(2)
    f_seen = np.mean(fs[:-1]).round(2)
    p_unseen = round(ps[-1], 2)
    r_unseen = round(rs[-1], 2)
    f_unseen = round(fs[-1], 2)
    acc = (sum(correct) / sum(total) * 100).round(2)
    acc_in = (sum(correct[:-1]) / sum(total[:-1]) * 100).round(2)
    acc_ood = (correct[-1] / total[-1] * 100).round(2)

    return f, acc, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen
def get_score_macro(true_and_prediction):
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.
    precision = 0
    recall = 1
    seen_precision=0
    seen_recall = 1
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p

        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_label_count > 0:
        recall = r / gold_label_count
    f=f1(precision, recall)

    return f
def f1(p, r):
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)
def macro(true_and_prediction):
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  precision=0
  recall=1
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
  if pred_example_count > 0:
    precision = p / pred_example_count
  if gold_label_count > 0:
    recall = r / gold_label_count
  return f1(precision, recall)
def micro(true_and_prediction):

  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return 0
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return f1(precision, recall)
def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f
def evaluate(logits, labels, id2label, map_ids=None,train_dataloader=None,threshold=0.5,
             model=None,flag=None,feature_test=None,logits_mbs=None,logits_opens=None):

    # assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # epoch_gold = epoch_labels
    all_pred_id_list = []

    all_test_pred_label = []
    all_test_truth = []
    results = {}

    if flag == 'test':
        label_num = len(id2label)
        num_sample = logits.shape[0]
        p = F.softmax(logits, 1)
        pred_p = p.data.max(1)[1]
        pred_prob = p.data.max(1)[0]
        # pred_p = pred_p.cpu().tolist()
        r = F.softmax(logits_mbs.view(logits_mbs.size(0), 2, -1), 1)
        tmp_range = torch.arange(0, logits_mbs.size(0)).long().cuda()
        unk_score = r[tmp_range, 0, pred_p]

        ind_unk = unk_score > 0.5
        # pred_prob_unk = pred_prob < 0.5
        pred_p[ind_unk] = label_num
        # pred_p[pred_prob_unk] = label_num
        pred_p = pred_p.cpu().tolist()

        #### 这个地方要改成，细粒度计算正确，对应的粗粒度也要出来
        epoch_predicts=pred_p
        epoch_labels=labels
        all_ood_truth=[]
        all_ood_pred=[]
        right_count_list = [0 for _ in range(len(id2label)+1)]
        gold_count_list = [0 for _ in range(len(id2label)+1)]
        predicted_count_list = [0 for _ in range(len(id2label)+1)]
        for p_i in range(len(epoch_predicts)):
            p = epoch_predicts[p_i]
            if p !=  label_num:
                p_map = map_ids[p]
            # np_sample_predict = np.array(p_map, dtype=np.float32)
            #test_predict_idx = np.argsort(-np_sample_predict)
            # sample_predict_id_list = []
            # and is_inlier[p_i] != -1
            # for j in range(len(test_predict_idx)):
            #     ##只有当预测的概率大于阈值的时候才可以
            #     if np_sample_predict[test_predict_idx[j]] > threshold :
            #         sample_predict_id_list.append(test_predict_idx[j])
            #
            # if sample_predict_id_list != [] :

                all_test_pred_label.append(p_map)
            else:
                p_map=[label_num]
                # sample_predict_id_list.append(label_num)
                all_test_pred_label.append(p_map)
            all_test_truth.append(epoch_labels[p_i])

            for gold in epoch_labels[p_i]:
                gold_count_list[gold] += 1
                for label in p_map:
                    if gold == label:
                        right_count_list[gold] += 1
            # count for the predicted items
            for label in p_map:
                predicted_count_list[label] += 1

            if label_num in epoch_labels[p_i]:
                all_ood_truth.append(0)
            else:
                all_ood_truth.append(1)
            if label_num in p_map:
                all_ood_pred.append(0)
            else:
                all_ood_pred.append(1)
        precision_dict = dict()
        recall_dict = dict()
        fscore_dict = dict()
        right_total, predict_total, gold_total = 0, 0, 0
        id_gold_total=0
        for i in range(len(id2label)+1):
            label = str(i)
            precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                                 predicted_count_list[
                                                                                                     i],
                                                                                                 gold_count_list[i])
            right_total += right_count_list[i]
            gold_total += gold_count_list[i]
            predict_total += predicted_count_list[i]
            if i != len(id2label):
                id_gold_total += gold_count_list[i]

        # Macro-F1
        precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
        recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
        macro_f1 = sum([v*gold_count_list[int(i)] for i, v in fscore_dict.items()]) / gold_total
        id_macro_f1 = sum([v * gold_count_list[int(i)] for i, v in fscore_dict.items() if int(i) != len(id2label)]) / id_gold_total
        macro_f1_avg = sum([v for i, v in fscore_dict.items()]) / len(fscore_dict.keys())
        id_macro_f1_avg = sum(
            [v for i, v in fscore_dict.items() if int(i) != len(id2label)]) / (len(fscore_dict.keys()) - 1)
        # Micro-F1
        precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
        recall_micro = float(right_total) / gold_total
        micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
        f_all= get_score_macro(zip(all_test_truth,all_test_pred_label))
        f_ood = f1_score(all_ood_truth,all_ood_pred)
        results = {}
        results['F1_ALL'] = f_all
        results['Macro-F1_ALL'] = macro_f1
        results['Macro-F1_ID'] = id_macro_f1
        results['Macro-F1_ALL_AVG'] = macro_f1_avg
        results['Macro-F1_ID_AVG'] = id_macro_f1_avg
        results['Micro-F1_ALL'] = micro_f1
        results['F1_OOD'] = f_ood
        acc = sum(
            [set(all_test_pred_label[i]) == set(epoch_labels[i]) for i in range(len(all_test_pred_label))]) * 1.0 / len(
            all_test_pred_label)
        results['ACC_ALL'] = acc


    elif flag=='eval':
        label_num = len(id2label)
        num_sample = logits.shape[0]
        p = F.softmax(logits, 1)
        pred_p = p.data.max(1)[1]
        pred_prob = p.data.max(1)[0]
        #pred_p = pred_p.cpu().tolist()
        r = F.softmax(logits_mbs.view(logits_mbs.size(0), 2, -1), 1)
        tmp_range = torch.arange(0, logits_mbs.size(0)).long().cuda()
        unk_score = r[tmp_range, 0, pred_p]

        ind_unk = unk_score > 0.5
        #pred_prob_unk = pred_prob < 0.5
        pred_p[ind_unk] = label_num
        #pred_p[pred_prob_unk] = label_num
        pred_p = pred_p.cpu().tolist()


        epoch_predicts=pred_p
        epoch_labels=labels
        all_ood_truth=[]
        all_ood_pred=[]
        right_count_list = [0 for _ in range(len(id2label)+1)]
        gold_count_list = [0 for _ in range(len(id2label)+1)]
        predicted_count_list = [0 for _ in range(len(id2label)+1)]
        for p_i in range(len(epoch_predicts)):
            p = epoch_predicts[p_i]
            if p !=  label_num:
                p_map = map_ids[p]
            # np_sample_predict = np.array(p_map, dtype=np.float32)
            #test_predict_idx = np.argsort(-np_sample_predict)
            # sample_predict_id_list = []
            # and is_inlier[p_i] != -1
            # for j in range(len(test_predict_idx)):
            #     ##只有当预测的概率大于阈值的时候才可以
            #     if np_sample_predict[test_predict_idx[j]] > threshold :
            #         sample_predict_id_list.append(test_predict_idx[j])
            #
            # if sample_predict_id_list != [] :
                all_test_pred_label.append(p_map)
            else:
                p_map=[label_num]
                # sample_predict_id_list.append(label_num)
                all_test_pred_label.append(p_map)
            all_test_truth.append(epoch_labels[p_i])

            for gold in epoch_labels[p_i]:
                gold_count_list[gold] += 1
                for label in p_map:
                    if gold == label:
                        right_count_list[gold] += 1
            # count for the predicted items
            for label in p_map:
                predicted_count_list[label] += 1

            if label_num in epoch_labels[p_i]:
                all_ood_truth.append(0)
            else:
                all_ood_truth.append(1)
            if label_num in p_map:
                all_ood_pred.append(0)
            else:
                all_ood_pred.append(1)
        precision_dict = dict()
        recall_dict = dict()
        fscore_dict = dict()
        right_total, predict_total, gold_total = 0, 0, 0
        id_gold_total=0
        for i in range(len(id2label)+1):
            label = str(i)
            precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                                 predicted_count_list[
                                                                                                     i],
                                                                                                 gold_count_list[i])
            right_total += right_count_list[i]
            gold_total += gold_count_list[i]
            predict_total += predicted_count_list[i]
            if i != len(id2label):
                id_gold_total += gold_count_list[i]

        # Macro-F1
        precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
        recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
        macro_f1 = sum([v*gold_count_list[int(i)] for i, v in fscore_dict.items()]) / gold_total
        id_macro_f1 = sum([v * gold_count_list[int(i)] for i, v in fscore_dict.items() if int(i) != len(id2label)]) / id_gold_total
        macro_f1_avg = sum([v for i, v in fscore_dict.items()]) / len(fscore_dict.keys())
        id_macro_f1_avg = sum(
            [v for i, v in fscore_dict.items() if int(i) != len(id2label)]) / (len(fscore_dict.keys()) - 1)
        # Micro-F1
        precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
        recall_micro = float(right_total) / gold_total
        micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
        f_all= get_score_macro(zip(all_test_truth,all_test_pred_label))
        f_ood = f1_score(all_ood_truth,all_ood_pred)
        results = {}
        results['F1_ALL'] = f_all
        results['Macro-F1_ALL'] = macro_f1
        results['Macro-F1_ID'] = id_macro_f1
        results['Macro-F1_ALL_AVG'] = macro_f1_avg
        results['Macro-F1_ID_AVG'] = id_macro_f1_avg
        results['Micro-F1_ALL'] = micro_f1
        results['F1_OOD'] = f_ood
        acc = sum(
            [set(all_test_pred_label[i]) == set(epoch_labels[i]) for i in range(len(all_test_pred_label))]) * 1.0 / len(
            all_test_pred_label)
        results['ACC_ALL'] = acc




    return results
