import datetime
import logging
import sys

import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import wandb
from transformers import BertTokenizer, BertModel, AutoConfig
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from data_utils import _load_shard, BertDataset, BertDataset_OOD, OODataset, LLM_OODataset
from tqdm import tqdm
import argparse
import os
# from eval import evaluate
from model.contrast import GenerateModel
from model.contrast_moco import  ContrastiveModelMoco
import constant
import utils
from ood_utils import _tokenize, create_masks, fill_masks

from open_eval import evaluate


class Saver:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)

def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses
    for item in dataset:
        temp = set(item)
        for t in temp:
            count[t] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N / float(count[i])
    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        for v in val:
            weight[idx] +=weight_per_class[v]
    return weight
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--data', type=str, default='fnerd', choices=['BBN', 'fnerd'], help='Dataset.')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--early-stop', type=int, default=10, help='Epoch before early stop.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--update', type=int, default=1, help='Gradient accumulate steps')
parser.add_argument('--warmup', default=2000, type=int, help='Warmup steps.')
parser.add_argument('--contrast',  default=False,type=bool, help='Whether use contrastive model.')
# parser.add_argument('--open_flag',  default=True,type=bool, help='Whether use open loss.')
# parser.add_argument('--weight_flag',  default=True,type=bool, help='Whether use open loss.')
parser.add_argument('--cls_loss',  default=True,type=bool, help='Whether use cls loss.')
parser.add_argument('--layer', default=2, type=int, help='Layer of Graphormer.')
parser.add_argument('--lamb', default=0.0, type=float, help='lambda')
# parser.add_argument('--open_lamda', default=0.0, type=float, help='lambda')
# parser.add_argument('--p_cutoff', default=0.5, type=float, help='lambda')
parser.add_argument('--thre', default=0.0001, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=6, type=int, help='Random seed.')
parser.add_argument('--max_num', type=int, default=200, help='mask word num')
parser.add_argument('--max_epoch', type=int, default=50, help='mask word num')
parser.add_argument('--positive_num', type=int, default=3, help='positive num sample')
parser.add_argument('--knn_num', type=int, default=25, help='positive num sample')
parser.add_argument('--ood_flag', type=bool, default=False, help='ood flag')
parser.add_argument('--llm_propotion', default=0.1, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--plm_propotion', default=0.1, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    print(args)
    utils.seed_torch(args.seed)
    train_name = args.data + '/' + 'ood_type_train.json'
    dev_name = args.data + '/' + 'ood_type_dev.json'
    test_name = args.data + '/' + 'ood_type_test.json'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data_path = os.path.join('data', train_name)
    train_ood_data_path = os.path.join('checkpoints', args.data)
    dev_data_path = os.path.join('data', dev_name)
    test_data_path = os.path.join('data', test_name)
    data_path = os.path.join('data', args.data)
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    train_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=train_data_path,data_name=args.data,args=args)
    ood_dataset = OODataset(device=device, tokenizer=tokenizer,ood_data_path = train_ood_data_path,data_name=args.data,args=args)
    llm_ood_dataset = LLM_OODataset(device=device, tokenizer=tokenizer,ood_data_path = train_ood_data_path,data_name=args.data,args=args)
    dev_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=dev_data_path,data_name=args.data)
    test_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=test_data_path, data_name=args.data,flag='test')

    model = ContrastiveModelMoco.from_pretrained('bert-base-uncased', num_labels=train_dataset.fine_grained_num,
                                                 cls_loss=args.cls_loss,
                                                 contrast_loss=args.contrast,
                                                 lamb=args.lamb,
                                                 queue_size=len(train_dataset), knn_num=args.knn_num, end_k=args.knn_num, ood_num=args.knn_num,
                                                 positive_num=args.positive_num,
                                                 coarse_label_num=train_dataset.coarse_num,
                                                 fine_grained_label_num=train_dataset.fine_grained_num, )
    # model = GenerateModel.from_pretrained('bert-base-uncased', num_labels=num_class, cls_loss=args.cls_loss,
    #                                       contrast_loss=args.contrast, layer=args.layer, data_path=data_path,
    #                                       lamb=args.lamb, threshold=args.thre, tau=args.tau)
    # if args.warmup> 0:
    training_steps = int(len(train_dataset)/args.batch)*args.max_epoch
    args.warmup = int(len(train_dataset)/args.batch)*args.max_epoch*0.1
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
    optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=args.lr,
                    betas=(0.9,0.999),
                    eps=1e-8,
                )
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup, num_training_steps=training_steps
            )
        # optimizer = ScheduledOptim(Adam(model.parameters(),
        #                  lr=args.lr,weight_decay=1e-2),args.lr,n_warmup_steps=args.warmup)
    # else:
    #     optimizer = AdamW(
    #                 optimizer_grouped_parameters,
    #                 lr=args.lr,
    #                 betas=(0.9,0.999),
    #                 eps=1e-8,
    #             )
    g = torch.Generator()
    g.manual_seed(0)
    torch.manual_seed(0)
    weights = make_weights_for_balanced_classes(train_dataset.y_category, train_dataset.answer_num)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),replacement=True,generator=g)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, collate_fn=train_dataset.collate_fn,sampler=sampler)
    train_seq = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, collate_fn=train_dataset.collate_fn)
    dev = DataLoader(dev_dataset, batch_size=args.batch, shuffle=False, collate_fn=dev_dataset.collate_fn)
    test = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=dev_dataset.collate_fn)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch, shuffle=True, collate_fn=ood_dataset.collate_fn)
    llm_ood_loader = DataLoader(llm_ood_dataset, batch_size=args.batch, shuffle=True, collate_fn=llm_ood_dataset.collate_fn)

    model.to(device)
    save = Saver(model, optimizer, None, args)

    best_acc=0
    early_stop_count = 0
    best_f1=0
    best_macro_f1_all_avg=0
    best_macro_f1_id_avg=0
    best_epoch=0
    logger = logger_config(log_path=os.path.join('checkpoints', args.data, 'log.txt'), logging_name='chen')
    output_f = open(os.path.join('checkpoints', args.data, 'aug_output_LLM_propotion.txt'), 'a+')
    time = datetime.datetime.now()

    for epoch in range(args.max_epoch):
        g.manual_seed(epoch)
        ood_loader_iter = iter(ood_loader)
        llm_ood_loader_iter = iter(llm_ood_loader)
        step = 0
        loss = 0
        total_loss=0
        contrast_loss=0
        model.train()
        train_pbar = tqdm(train_dataloader)
        for inputs, label,idxs,id_labels,coarse_labels,fine_grained_labels in train_pbar:
            if (step+1) % 2 == 0:
                try:
                    ood_inputs,ood_labels = next(ood_loader_iter)
                except StopIteration:
                    ood_loader_iter = iter(ood_loader)
                    ood_inputs,ood_labels = next(ood_loader_iter)
            else:
                try:
                    ood_inputs,ood_labels = next(llm_ood_loader_iter)
                except StopIteration:
                    llm_ood_loader_iter = iter(llm_ood_loader)
                    ood_inputs,ood_labels = next(llm_ood_loader_iter)
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
            coarse_positive_sample,fine_grained_positive_sample= train_dataset.generate_positive_samples(coarse_truth,fine_grained_truth)
            positive_samples = train_dataset.collate_fn(fine_grained_positive_sample)
            output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=label,
                           coarse_labels=coarse_labels, fine_grained_labels=fine_grained_labels, return_dict=True,
                           ood_ids=ood_inputs["input_ids"], ood_attention_mask=ood_inputs["attention_mask"],
                           ood_labels=ood_labels,
                           positive_samples=positive_samples, is_train=True, ood_flag=args.ood_flag)
            optimizer.zero_grad()
            output['loss'].backward()
            loss += output['loss'].item()
            if args.contrast == True:
                contrast_loss+=output['hie_sup_loss'].item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()

            step+=1

            train_pbar.set_description('loss:{:.4f}'.format(output['loss'].item()))
        print('train_loss: ' + str(loss/step))
        print('contrast_loss: ' + str(contrast_loss / step))
        train_pbar.close()

        model.eval()
        pbar = tqdm(dev)
        with torch.no_grad():
            truth = []
            pred = []
            logits = None
            logits_mbs=None
            logits_open_cls=None
            for inputs, label,idxs,id_labels,coarse_labels,fine_grained_labels in pbar:

                output = model.evaluate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],labels=label, return_dict=True,is_train=False)
                for l in label:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                if logits != None:
                    logits = torch.cat((logits, output['logits']), dim=0)
                else:
                    logits = output['logits']
                if logits_mbs != None:
                    logits_mbs = torch.cat((logits_mbs, output['logits_mb_cls']), dim=0)
                else:
                    logits_mbs = output['logits_mb_cls']
        pbar.close()
        scores = evaluate(logits, truth, label_dict,map_ids=test_dataset.map_ids,flag='eval',logits_mbs=logits_mbs)

        acc = scores['ACC_ALL']
        macro_f1_all_avg = scores['Macro-F1_ALL']
        macro_f1_id_avg = scores['Macro-F1_ID']
        logger.info('epoch: ' + str(epoch))
        logger.info('dev acc: ' + str(acc))
        logger.info('dev macro f1 all: ' + str(macro_f1_all_avg))
        logger.info('dev macro f1 id: ' + str(macro_f1_id_avg))
        early_stop_count += 1
        if macro_f1_id_avg > best_macro_f1_id_avg:
            best_macro_f1_id_avg = macro_f1_id_avg
            best_epoch = epoch
            best_acc=acc
            save(best_macro_f1_id_avg, best_macro_f1_id_avg,
                 os.path.join('checkpoints', args.data,
                              'checkpoint_best_acc_LLM_lamb_' + str(args.lamb) + '_ood_' + str(
                                  args.ood_flag) + '_contrast_' + str(args.contrast) + '_k_' + str(
                                  args.knn_num) + '_propotion.pt'))
            early_stop_count = 0
        # if early_stop_count > 10:
        #     break


    checkpoint = torch.load(
        os.path.join('checkpoints', args.data,
                     'checkpoint_best_acc_LLM_lamb_' + str(args.lamb) + '_ood_' + str(
                         args.ood_flag) + '_contrast_' + str(args.contrast) + '_k_'+str(args.knn_num)+'_propotion.pt'))
    model.load_state_dict(checkpoint['param'])

    model.eval()

    with torch.no_grad():
        truth = []
        logits = None
        logits_mbs=None
        logits_opens=None
        logits_open_cls=None
        feature_test = None
        pbar = tqdm(test)
        for inputs, label,idxs,id_labels,coarse_labels,fine_grained_labels in pbar:

            output = model.evaluate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],labels=label,
                           return_dict=True,is_train=False)
            if feature_test != None:
                feature_test = torch.cat((feature_test, output['pooled_out']), dim=0)

            else:
                feature_test = output['pooled_out']

            for l in label:
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                if t == []:
                    truth.append([test_dataset.answer_num])
                else:
                    truth.append(t)
            if logits!=None:
                logits=torch.cat((logits,output['logits']),dim=0)
            else:
                logits=output['logits']
            if logits_mbs !=None:
                logits_mbs = torch.cat((logits_mbs, output['logits_mb_cls']), dim=0)
            else:
                logits_mbs = output['logits_mb_cls']
    results = evaluate(logits, truth, label_dict, map_ids=test_dataset.map_ids,train_dataloader=train_dataloader, feature_test=feature_test,model=model, flag='test',logits_mbs=logits_mbs)
    output_f.write("ood_flag=" + str(args.ood_flag) + '\n')
    print(("ood_flag=" + str(args.ood_flag)))
    output_f.write("contrast_loss=" + str(args.contrast) + '\n')
    print(("contrast_loss=" + str(args.contrast)))
    # output_f.write("open_loss=" + str(args.open_flag) + '\n')
    # print(("open_loss=" + str(args.open_flag)))
    output_f.write("lamda=" + str(args.lamb) + '\n')
    print(("lamda=" + str(args.lamb)))
    # output_f.write("p_cutoff=" + str(args.p_cutoff) + '\n')
    # print(("p_cutoff=" + str(args.p_cutoff)))
    # output_f.write("thre=" + str(args.thre) + '\n')
    # output_f.write("open lamda=" + str(args.open_lamda) + '\n')
    # print(("open lamda=" + str(args.open_lamda) + '\n'))
    # print(("thre=" + str(args.thre)))
    output_f.write("seed=" + str(args.seed) + '\n')
    print(("seed=" + str(args.seed)))
    output_f.write("acc_all=" + str(results['ACC_ALL']) + '\n')
    print(("acc_all=" + str(results['ACC_ALL'])))
    output_f.write("f_all=" + str(results['F1_ALL']) + '\n')
    print(("f_all=" + str(results['F1_ALL'])))

    output_f.write("macro_all=" + str(results['Macro-F1_ALL']) + '\n')
    print(("macro_all=" + str(results['Macro-F1_ALL'])))

    output_f.write("micro_all=" + str(results['Micro-F1_ALL']) + '\n')
    print(("micro_all=" + str(results['Micro-F1_ALL'])))

    output_f.write("macro_id=" + str(results['Macro-F1_ID']) + '\n')
    print(("macro_id=" + str(results['Macro-F1_ID'])))

    output_f.write("macro_id_avg=" + str(results['Macro-F1_ID_AVG']) + '\n')
    print(("macro_id_avg=" + str(results['Macro-F1_ID_AVG'])))

    output_f.write("macro_all_avg=" + str(results['Macro-F1_ALL_AVG']) + '\n')
    print(("macro_all_avg=" + str(results['Macro-F1_ALL_AVG'])))

    output_f.write("f_ood=" + str(results['F1_OOD']) + '\n')
    print(("f_ood=" + str(results['F1_OOD']) + '\n'))

    output_f.write("best_epoch=" + str(best_epoch) + '\n')
    print(("best_epoch=" + str(best_epoch) + '\n'))

    output_f.write("llm_propotion=" + str(args.llm_propotion) + '\n')
    print(("llm_propotion=" + str(args.llm_propotion) + '\n'))
    output_f.write("plm_propotion=" + str(args.plm_propotion) + '\n')
    print(("plm_propotion=" + str(args.plm_propotion) + '\n'))
















