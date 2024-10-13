#!/usr/bin/env python
# coding:utf-8
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance, GraphicalLasso
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

import utils
from ood_utils import get_auroc, get_fpr_95, compute_ood, get_f1_score, get_aupr


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
        bi_cm = cm[idx]
        total.append(np.sum(bi_cm))
        correct.append(bi_cm[1][1])
        precision = bi_cm[1][1] / (bi_cm[1][1] + bi_cm[0][1]) if (bi_cm[1][1] + bi_cm[0][1]) != 0 else 0
        recall = bi_cm[1][1] / (bi_cm[1][1] + bi_cm[1][0]) if (bi_cm[1][1] + bi_cm[1][0]) != 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        fs.append(f1 * 100)
        ps.append(precision * 100)
        rs.append(recall * 100)

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
def evaluate(epoch_predicts, epoch_labels, id2label, train_dataloader=None,threshold=0.5,
             model=None,flag=None,feature_test=None,feature_valid=None,prob_valid=None,valid_truth=None):

    all_ood_score_dict = {'softmax': [], 'energy': [],'lof':[],'lof_score':[]}

    ood_score_outputs=defaultdict()
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    epoch_gold = epoch_labels
    all_pred_id_list = []
    out_domian_sample_predict = []
    in_domian_sample_predict = []
    in_domian_gold_list=[]
    all_test_pred_label = []
    all_test_truth = []
    out_domian_gold_list = []
    results = {}

    if flag == 'test':
        feature_train=None
        label_num = len(id2label)
        class_mean = torch.zeros(label_num, 768).cuda()
        class_num_list = np.zeros(label_num,)
        label_list=[]
        with torch.no_grad():
            for inputs, label, idxs,id_labels,coarse_labels,fine_grained_labels in train_dataloader:
                output=model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=label,coarse_labels=coarse_labels,fine_grained_labels=fine_grained_labels,
                      return_dict=True,is_train=False)
                if feature_train != None:
                    feature_train = torch.cat((feature_train, output['pooled_out']), dim=0)
                else:
                    feature_train = output['pooled_out']

                # for l in range(len(label)):
                #     t = []
                #     for i in range(label[l].size(0)):
                #         if label[l][i].item() == 1:
                #             class_mean[i] += output['pooled_out'][l]
                #             class_num_list[i] += 1
                #             t.append(i)
                #     label_list.append(t)

        # for i in range(len(class_num_list)):
        #     if class_num_list[i] != 0:
        #         class_mean[i] = class_mean[i] / class_num_list[i]
        # feature_class = torch.zeros(feature_train.shape).cuda()
        # for i in range(len(label_list)):
        #     feature_class[i] = class_mean[label_list[i]].mean(0)
        #
        # centered_bank = (feature_train - feature_class).detach().cpu().numpy()
        # precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        # class_var = torch.from_numpy(precision).float().cuda()
        # maha_score = []
        # for c in range(len(class_num_list)):
        #     centered_pooled = feature_test - class_mean[c].unsqueeze(0)
        #     ms = torch.diag(centered_pooled @ class_var@ centered_pooled.t())
        #     maha_score.append(ms)
        # max_maha = []
        # for c in range(len(class_num_list)):
        #     centered_pooled = feature_train - class_mean[c].unsqueeze(0)
        #     ms = torch.diag(centered_pooled @ class_var@centered_pooled.t())
        #     max_maha.append(ms)
        # max_maha_score = torch.stack(max_maha, dim=-1)
        # max_maha_score = max(max_maha_score.max(-1)[0])
        #
        #
        # maha_score = torch.stack(maha_score, dim=-1)
        # maha_score = maha_score.min(-1)[0]


        feature_train = feature_train.cpu().detach().numpy()

        feature_test = feature_test.cpu().detach().numpy()
        #feature_valid = feature_valid.cpu().detach().numpy()






        lof = LocalOutlierFactor(n_neighbors=20, metric='l2',novelty=True,
                                 n_jobs=-1)
        lof.fit(feature_train)
        lof_score = lof.score_samples(feature_test)
        #lof_score_val = lof.score_samples(feature_valid)
        best_item = -1.5
        best_ind_f1 = 0
        best_oos_f1 = 0
        best_acc_ood = 0
        best_acc_in = 0
        best_f1=0
        min_thres = min(lof_score)
        max_thres = max(lof_score)
        theshold_list = np.arange(-1, -5, -0.01)

        # ood = np.zeros(len(id2label)+1)
        # ood[[len(id2label)]]=1
        # prob_valid = [torch.sigmoid(item).tolist() for item in prob_valid]
        epoch_predicts = [torch.sigmoid(item).tolist() for item in epoch_predicts]
        # for item in theshold_list:
        #     is_inlier = np.zeros((len(feature_test)))
        #
        #     index_out = lof_score - item
        #     is_inlier[index_out <= 0] = -1
        #
        #     all_valid_pred_label=[]
        #     all_valid_truth=[]
        #     for p_i in range(len(epoch_predicts)):
        #         p = epoch_predicts[p_i]
        #         np_sample_predict = np.array(p, dtype=np.float32)
        #         valid_predict_idx = np.argsort(-np_sample_predict)
        #         sample_predict_id_list = []
        #
        #         for j in range(len(valid_predict_idx)):
        #             ##只有当预测的概率大于阈值的时候才可以
        #             if np_sample_predict[valid_predict_idx[j]] > threshold:
        #                 sample_predict_id_list.append(valid_predict_idx[j])
        #
        #         if sample_predict_id_list != [] and is_inlier[p_i] != -1:
        #             temp = np.zeros(len(id2label) + 1)
        #             temp[sample_predict_id_list] = 1
        #             # all_valid_pred_label.append(temp)
        #             all_valid_pred_label.append(sample_predict_id_list)
        #         else:
        #             temp = np.zeros(len(id2label) + 1)
        #             temp[[len(id2label)]] = 1
        #             # all_valid_pred_label.append(temp)
        #             all_valid_pred_label.append([len(id2label)])
        #         temp = np.zeros(len(id2label) + 1)
        #         temp[epoch_labels[p_i]] = 1
        #         # all_valid_truth.append(temp)
        #         all_valid_truth.append(epoch_labels[p_i])
        #     classes = [i for i in range(len(id2label))]
        #     #cm=multilabel_confusion_matrix(all_valid_truth, all_valid_pred_label)
        #     f_all= get_score_macro(zip(all_valid_truth,all_valid_pred_label))
        #     # just only use in-domain data
        #     if f_all >= best_f1:
        #         best_item = item
        #         best_f1 = f_all

        print('best_item: '+str(best_item))
        print('best_f1: '+str(best_f1))
        is_inlier = np.zeros(len(feature_test))
        # y_pred_lof = lof.predict(feature_test)
        best_item = -1.5
        index_out = lof_score - best_item

        is_inlier[index_out <= 0] = -1


        all_ood_truth=[]
        all_ood_pred=[]
        right_count_list = [0 for _ in range(len(id2label)+1)]
        gold_count_list = [0 for _ in range(len(id2label)+1)]
        predicted_count_list = [0 for _ in range(len(id2label)+1)]
        for p_i in range(len(epoch_predicts)):
            p = epoch_predicts[p_i]
            np_sample_predict = np.array(p, dtype=np.float32)
            test_predict_idx = np.argsort(-np_sample_predict)
            sample_predict_id_list = []
            # and is_inlier[p_i] != -1
            for j in range(len(test_predict_idx)):
                ##只有当预测的概率大于阈值的时候才可以
                if np_sample_predict[test_predict_idx[j]] > threshold :
                    sample_predict_id_list.append(test_predict_idx[j])

            if sample_predict_id_list != [] :
                all_test_pred_label.append(sample_predict_id_list)
            else:
                sample_predict_id_list.append(label_num)
                all_test_pred_label.append([label_num])
            all_test_truth.append(epoch_labels[p_i])

            for gold in epoch_labels[p_i]:
                gold_count_list[gold] += 1
                for label in sample_predict_id_list:
                    if gold == label:
                        right_count_list[gold] += 1
            # count for the predicted items
            for label in sample_predict_id_list:
                predicted_count_list[label] += 1



            if label_num in epoch_labels[p_i]:
                all_ood_truth.append(0)
            else:
                all_ood_truth.append(1)
            if label_num in sample_predict_id_list:
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




        # for i in range(len(epoch_predicts)):
        #     ood_score_dict = compute_ood(epoch_predicts[i])
        #     for k,v in ood_score_dict.items():
        #         all_ood_score_dict[k].append(v)



        # epoch_predicts = [torch.sigmoid(item).tolist() for item in epoch_predicts]
        # i = 0
        # for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        #     np_sample_predict = np.array(sample_predict, dtype=np.float32)
        #     sample_predict_descent_idx = np.argsort(-np_sample_predict)
        #     sample_predict_id_list = []
        #
        #     for j in range(len(sample_predict)):
        #         ##只有当预测的概率大于阈值的时候才可以
        #         if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
        #             sample_predict_id_list.append(sample_predict_descent_idx[j])
        #     # if sample_predict_id_list != [] and maha_score[i] <= max_maha_score:
        #     if sample_predict_id_list != [] and lof_score[i] == 1:
        #         all_pred_id_list.append(sample_predict_id_list)
        #     else:
        #         sample_predict_id_list = [len(id2label)]
        #         all_pred_id_list.append(sample_predict_id_list)
        #     if len(id2label) not in sample_gold:
        #         sample_predict_id_list = []
        #         for j in range(len(sample_predict)):
        #             ##只有当预测的概率大于阈值的时候才可以
        #             if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
        #                 sample_predict_id_list.append(sample_predict_descent_idx[j])
        #         in_domian_sample_predict.append(sample_predict_id_list)
        #         in_domian_gold_list.append(sample_gold)
        #     i = i+1
        #
        # true_and_predicts = zip(epoch_gold, all_pred_id_list)
        # f1 = macro(true_and_predicts)
        # true_and_predicts = zip(epoch_gold, all_pred_id_list)
        # f1_micro = micro(true_and_predicts)
        # f1_ind = macro(zip(in_domian_gold_list,in_domian_sample_predict))
        # results['F1_ALL'] = f1
        # results['F1_ALL_micro'] = f1_micro
        # # results['F1_OOD'] = f_unseen
        # results['F1_IND'] = f1_ind
    elif flag=='eval':
        label_num = len(id2label)
        right_count_list = [0 for _ in range(len(id2label) + 1)]
        gold_count_list = [0 for _ in range(len(id2label) + 1)]
        predicted_count_list = [0 for _ in range(len(id2label) + 1)]
        epoch_predicts=[torch.sigmoid(item).tolist() for item in epoch_predicts]
        all_ood_truth = []
        all_ood_pred = []
        for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
                np_sample_predict = np.array(sample_predict, dtype=np.float32)
                sample_predict_descent_idx = np.argsort(-np_sample_predict)
                sample_predict_id_list = []

                for j in range(len(sample_predict)):
                    ##只有当预测的概率大于阈值的时候才可以
                    if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                        sample_predict_id_list.append(sample_predict_descent_idx[j])
                if sample_predict_id_list != []:
                    all_test_pred_label.append(sample_predict_id_list)
                else:
                    sample_predict_id_list.append(label_num)
                    all_test_pred_label.append([label_num])
                all_pred_id_list.append(sample_predict_id_list)
                all_test_truth.append(sample_gold)
                for gold in sample_gold:
                    gold_count_list[gold] += 1
                    for label in sample_predict_id_list:
                        if gold == label:
                            right_count_list[gold] += 1
                    # count for the predicted items
                for label in sample_predict_id_list:
                    predicted_count_list[label] += 1
                if label_num in sample_predict_id_list:
                    all_ood_truth.append(0)
                else:
                    all_ood_truth.append(1)
                if label_num in sample_predict_id_list:
                    all_ood_pred.append(0)
                else:
                    all_ood_pred.append(1)
        precision_dict = dict()
        recall_dict = dict()
        fscore_dict = dict()
        right_total, predict_total, gold_total = 0, 0, 0
        id_gold_total = 0
        for i in range(len(id2label) + 1):
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
        macro_f1 = sum([v * gold_count_list[int(i)] for i, v in fscore_dict.items()]) / gold_total
        id_macro_f1 = sum(
            [v * gold_count_list[int(i)] for i, v in fscore_dict.items() if int(i) != len(id2label)]) / id_gold_total

        macro_f1_avg = sum([v  for i, v in fscore_dict.items()]) / len(fscore_dict.keys())
        id_macro_f1_avg = sum(
            [v  for i, v in fscore_dict.items() if int(i) != len(id2label)]) /  (len(fscore_dict.keys())-1)
        # Micro-F1
        precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
        recall_micro = float(right_total) / gold_total
        micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

        f_all = get_score_macro(zip(all_test_truth, all_test_pred_label))
        f_ood = f1_score(all_ood_truth, all_ood_pred)
        results = {}
        results['F1_ALL'] = f_all
        results['Macro-F1_ALL'] = macro_f1
        results['Macro-F1_ID'] = id_macro_f1
        results['Macro-F1_ALL_AVG'] = macro_f1_avg
        results['Macro-F1_ID_AVG'] = id_macro_f1_avg
        results['Micro-F1_ALL'] = micro_f1
        results['F1_OOD'] = f_ood

        #f1 = macro(true_and_predicts)

    if flag == 'test':
        acc = sum([set(all_test_pred_label[i]) == set(epoch_gold[i]) for i in range(len(all_test_pred_label))]) * 1.0 / len(
            all_test_pred_label)
        results['ACC_ALL']=acc
        return results
    else:
        acc = sum([set(all_pred_id_list[i]) == set(epoch_gold[i]) for i in range(len(all_pred_id_list))]) * 1.0 / len(
            all_pred_id_list)
        results['ACC_ALL'] = acc
        return results
