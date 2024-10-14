import copy
import pickle
import queue
from collections import defaultdict

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from tqdm.contrib.concurrent import process_map
from transformers import BertTokenizer,BertModel,BertForMaskedLM
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
import torch.multiprocessing as mp
from data_utils import BertDataset, BertDataset_OOD
from model.contrast_moco import ContrastiveModelMoco
from ood_utils import _tokenize, create_masks, prepare_ood, compute_ood_threshold, get_substitues, \
    compute_ood, fill_masks, load_out_vocab
# from train_aug import  logger_config
from open_eval import evaluate
from model.contrast_aug import AugModel
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='fnerd', choices=['BBN', 'fnerd'], help='Dataset.')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=10, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast',  default=True,type=bool, help='Whether use contrastive model.')
parser.add_argument('--open_flag',  default=True,type=bool, help='Whether use open loss.')
parser.add_argument('--ova_flag',  default='max',type=str, choices=['mean', 'max'])
parser.add_argument('--cls_loss',  default=True,type=bool, help='Whether use cls loss.')
parser.add_argument('--layer', default=2, type=int, help='Layer of Graphormer.')
parser.add_argument('--lamb', default=0.0, type=float, help='lambda')
parser.add_argument('--open_lamda', default=0.5, type=float, help='lambda')
parser.add_argument('--p_cutoff', default=0.5, type=float, help='lambda')
parser.add_argument('--thre', default=0.0001, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=6, type=int, help='Random seed.')
parser.add_argument('--max_num', type=int, default=200, help='mask word num')
parser.add_argument('--max_epoch', type=int, default=50, help='mask word num')
parser.add_argument('--positive_num', type=int, default=3, help='positive num sample')
parser.add_argument('--extra', default='replace', choices=['acc,f1,replace'], help='An extra string in the name of checkpoint.')
args = parser.parse_args()



checkpoint = torch.load(os.path.join('checkpoints', args.data, 'checkpoint_best_acc_aug.pt'))
checkpoint2 = torch.load(os.path.join('checkpoints', args.data, 'checkpoint_best_acc_PLM_lamb_0.0_ood_False_contrast_False.pt'))
batch_size = args.batch
device = args.device
extra = args.extra
args = checkpoint['args'] if checkpoint['args'] is not None else args
data_path = os.path.join('data', args.data)

if not hasattr(args, 'graph'):
    args.graph = False
print(args)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
num_class = len(label_dict)
train_name = args.data + '/' + 'ood_type_train.json'
dev_name = args.data + '/' + 'ood_type_dev.json'
test_name = args.data + '/' + 'ood_type_test.json'
train_data_path = os.path.join('data', train_name)
dev_data_path = os.path.join('data', dev_name)
data_path = os.path.join('data', args.data)
test_data_path = os.path.join('data', test_name)
train_ood_data_path = os.path.join('checkpoints', args.data)
train_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=train_data_path, data_name=args.data,
                                args=args)
# train_data = BertDataset(device=device, tokenizer=tokenizer, data_path=train_data_path, data_name=args.data,id_flag=True)


dev_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=dev_data_path,data_name=args.data)
test_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=test_data_path, data_name=args.data,flag='test')
train = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, collate_fn=train_dataset.collate_fn)

dev = DataLoader(dev_dataset, batch_size=args.batch, shuffle=False, collate_fn=dev_dataset.collate_fn)

test = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=test_dataset.collate_fn)
model = AugModel.from_pretrained('bert-base-uncased',num_labels=train_dataset.fine_grained_num,cls_loss=args.cls_loss,
                                          contrast_loss=args.contrast, layer=args.layer, data_path=data_path,
                                          lamb=args.lamb, threshold=args.thre, tau=args.tau,
                                                 memory_bank=False,
                                                 queue_size=len(train_dataset),knn_num=25,end_k=25,ood_num=25,positive_num=args.positive_num,
                                                 coarse_label_num=train_dataset.coarse_num,fine_grained_label_num = train_dataset.fine_grained_num,
                                                 open_flag=args.open_flag,p_cutoff=args.p_cutoff,open_lamda=args.open_lamda,ova_flag=args.ova_flag)

filter_model = ContrastiveModelMoco.from_pretrained('bert-base-uncased',num_labels=train_dataset.fine_grained_num,cls_loss=args.cls_loss,
                                          contrast_loss=args.contrast,
                                          lamb=args.lamb,
                                            queue_size=len(train_dataset),knn_num=25,end_k=25,ood_num=25,positive_num=args.positive_num,
                                            coarse_label_num=train_dataset.coarse_num,fine_grained_label_num = train_dataset.fine_grained_num,)
model.load_state_dict(checkpoint['param'])
filter_model.load_state_dict(checkpoint2['param'])
model.to(device)
model.eval()
filter_model.to(device)
filter_model.eval()

mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.to(device)
mlm_model.eval()




if __name__ == '__main__':

    truth = []
    pred = []
    index = []
    slot_truth = []
    slot_pred = []




    all_mask_sents = []
    pbar = tqdm(train)

    # sub_word_dict=load_out_vocab('vocab_100k.txt',tokenizer)
    with torch.no_grad():


        all_ood_generations=[]
        task=[]
        masked_words=[]
        feature_train=None
        lof=None
        # for inputs, label, idxs, id_labels, coarse_labels, fine_grained_labels in train:
        #     output = model.evaluate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        #                             labels=label,
        #                             return_dict=True, is_train=False)
        #     if feature_train != None:
        #         feature_train = torch.cat((feature_train, output['pooled_out']), dim=0)
        #     else:
        #         feature_train = output['pooled_out']
        #
        # pbar.close()
        # feature_train = feature_train.cpu().detach().numpy()
        #
        # lof = LocalOutlierFactor(n_neighbors=20, metric='l2', novelty=True,
        #                          n_jobs=-1)
        # lof.fit(feature_train)
        #
        # pbar = tqdm(train)
        for inputs, label, idxs, id_labels, coarse_labels, fine_grained_labels in train:
            coarse_truth = []
            fine_grained_truth = []
            for l in coarse_labels:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                coarse_truth.append(t)
            for l in fine_grained_labels:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                fine_grained_truth.append(t)
            coarse_positive_sample, fine_grained_positive_sample = train_dataset.generate_positive_samples(coarse_truth,
                                                                                                           fine_grained_truth)
            positive_samples = train_dataset.collate_fn(fine_grained_positive_sample)
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=label,
                           coarse_labels=coarse_labels, fine_grained_labels=fine_grained_labels, return_dict=True, positive_samples = positive_samples,
                           is_train=True, epoch=0)
            context_list = []
            for idx in idxs:
                data_item = train_dataset.__getitem__(idx)
                context = ['[CLS]'] + data_item['mention_word'] + ['[SEP]'] + data_item['left_context'] + \
                                  data_item[
                                      'mention_word'] + data_item['right_context']
                context_list.append(context)

            decoded_texts = tokenizer.batch_decode(inputs["input_ids"])

            words_list, sub_words_list, keys_list = _tokenize(context_list, decoded_texts, tokenizer,
                                                                      inputs['input_ids'].size(1))
            contrast_mask = output['contrast_mask'].cpu().detach().numpy()
            contrast_mask_values = output['contrast_mask_values'].cpu().detach().numpy()
            all_masked_sents_list, all_masked_words_list = create_masks(words_list, sub_words_list, keys_list,contrast_mask,
                                        contrast_mask_values)


            masked_words.extend(all_masked_words_list)
            for i in range(len(all_masked_sents_list)):
                task.append([words_list[i], all_masked_sents_list[i], args.max_num, idxs[i],filter_model,train_dataset.filter_list,lof,label[i].unsqueeze(0),mlm_model])


        task_len = len(task)
        chunk_num = int(np.ceil(task_len/10))
        #task = task[chunk_num*0:chunk_num*1]
        ood_results = []
        for t in tqdm(task):
            res = fill_masks(t)
            ood_results.append(res)


        with open(os.path.join('checkpoints', args.data, 'base_ood_generate_data_0.pkl'),'wb') as f:
            pickle.dump(ood_results,f)
    pbar.close()

