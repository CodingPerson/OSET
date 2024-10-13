from transformers import AutoTokenizer
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import json
from collections import defaultdict

np.random.seed(7)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("../../bert-base-uncased")
    with open("bbn_train_types.txt",'r') as f:
        labels = [x.strip() for x in f.readlines()]

    ##这个地方是把label的name转化为bert id
    ##这个地方稍微有点问题，引入还没导入hierarchy
    value_dict = {i: tokenizer.encode(v.lower(), add_special_tokens=False) for i,v in enumerate(labels)}
    torch.save(value_dict, 'bert_value_dict.pt')


    labels_dict={l:i for i,l in enumerate(labels)}
    with open("bbn_types_train_hierachy.txt",'r') as f:
        hierarchy_labels = [x.strip() for x in f.readlines()]
    hiera = defaultdict(list)
    for i,label in enumerate(hierarchy_labels):
        splits = label.split('/')
        if len(splits)== 2:
            hiera[labels_dict[splits[1]]] =[]
        else:

            assert len(splits) == 3
            if labels_dict[splits[1]] not in hiera.keys():
                hiera[labels_dict[splits[1]]]=[labels_dict[splits[2]]]
            else:
                hiera[labels_dict[splits[1]]].append(labels_dict[splits[2]])
    torch.save(hiera, 'slot.pt')


