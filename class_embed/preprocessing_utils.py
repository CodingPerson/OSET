import csv
import itertools
import os
import random
import re
from collections import Counter

import numpy as np
from tqdm import tqdm

from class_utils import DATA_FOLDER_PATH



def clean_html(str):
    left_mark = '&lt;'
    right_mark = '&gt;'
    # for every line find matching left_mark and nearest right_mark
    while True:
        next_left_start = str.find(left_mark)
        if next_left_start == -1:
            break
        next_right_start = str.find(right_mark, next_left_start)
        if next_right_start == -1:
            print("Right mark without Left: " + str)
            break
        # print("Removing " + string[next_left_start: next_right_start + len(right_mark)])
        clean_html.clean_links.append(str[next_left_start: next_right_start + len(right_mark)])
        str = str[:next_left_start] + " " + str[next_right_start + len(right_mark):]
    return str


clean_html.clean_links = []


# mainly for 20news
def clean_email(str):
    return " ".join([s for s in str.split(' ') if "@" not in s])
def clean_html1(str):
    pattern = re.compile(r'<[^>]+>',re.S)
    str = pattern.sub(' ',str)
    return str

def clean_str(str):
    str = str.replace('.','')
    str = clean_html(str)
    str = clean_html1(str)
    str = clean_email(str)

    str = re.sub(r"[^A-Za-z0-9(),\(\)=+.!?\"\']", " ", str)
    str = re.sub(r"\s{2,}", " ", str)
    return str.strip()


def load_clean_text(data_dir):
    text = load_text(data_dir)
    return [clean_str(doc) for doc in text]


def load_text(data_dir):
    with open(os.path.join(data_dir, 'ood_type_train.json'), 'r') as f:
        lines = [eval(line.strip()) for line in tqdm(f)]

        tokens = [line["tokens"] for line in lines]

    return tokens
def load_mention(data_dir):
    with open(os.path.join(data_dir, 'ood_type_train.json'), 'r') as f:
        lines = [eval(line.strip()) for line in tqdm(f)]

        mentions = [line["mentions"] for line in lines]

    return mentions


def load_labels(data_dir):
    with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: int(str(x.strip()).split()[0]), label_file.readlines()))
    return labels
##chenhu
def load_tlabels(data_dir):
    with open(os.path.join(data_dir, 'tlabels.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: str(x.strip()), label_file.readlines()))
    return labels

##chenhu
def load_classnames(data_dir,dataset_name):
    with open(os.path.join(data_dir, dataset_name.lower()+'_types_train_for_embed.txt'), mode='r', encoding='utf-8') as classnames_file:
        class_names = "".join(classnames_file.readlines()).strip().split("\n")
    return class_names

def load_class(data_dir,dataset_name):
    with open(os.path.join(data_dir, dataset_name.lower()+'_types_train_hierachy.txt'), mode='r', encoding='utf-8') as classnames_file:
        labels = [x.strip() for x in classnames_file.readlines()]
        labels_dict = {l: i for i, l in enumerate(labels)}
    return labels_dict


def text_statistics(text, name="default"):
    sz = len(text)

    tmp_text = [s.split(" ") for s in text]
    tmp_list = [len(doc) for doc in tmp_text]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    # print(f"\n### Dataset statistics for {name}: ###")
    # print('# of documents is: {}'.format(sz))
    # print('Document max length: {} (words)'.format(len_max))
    # print('Document average length: {} (words)'.format(len_avg))
    # print('Document length std: {} (words)'.format(len_std))
    # print(f"#######################################")


def load(dataset_name):
    data_dir = os.path.join(DATA_FOLDER_PATH, dataset_name)
    text = load_text(data_dir)
    mentions = load_mention(data_dir)
    class_names = load_classnames(data_dir,dataset_name)
    class_dict = load_class(data_dir,dataset_name)
    # text = [' '.join(s) for s in text]
    #text_statistics(text, "raw_txt")

    #cleaned_text = [clean_str(doc) for doc in text]
    #print(f"Cleaned {len(clean_html.clean_links)} html links")
    #text_statistics(cleaned_text, "cleaned_txt")

    result = {
        "class_names": class_names,
        "raw_text": text,
        "cleaned_text": text,
        "mentions":mentions,
        "class_dict":class_dict
    }
    return result


if __name__ == '__main__':
    data = load('agnews')
