import argparse
import glob
import json
import os
import pickle
import random
from collections import defaultdict, OrderedDict

import numpy as np
import torch

from random import shuffle

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing import Any, Dict, Generator, Optional, Tuple, Union

import constant

from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import Sampler
import utils
from ood_utils import has_english_char

utils.seed_torch(0)
os.environ['PYTHONHASHSEED']=str(0)



def _load_shard(shard_name,context_window_size,do_lower,word2id,coatse2id,fine_grained2id):
      print("Loading {}".format(shard_name))
      idx=0
      with open(shard_name,'r') as f:
        lines = [eval(line.strip()) for line in tqdm(f)]
        #ex_ids = [line["ex_id"] for line in lines]
        mentions = [line["mentions"] for line in lines]
        mention_word=[]
        y_category=[]
        left_context=[]
        right_context=[]
        idx_list=[]
        coarse_label=[]
        fine_grained_label=[]
        for i in range(len(mentions)):
          for mention in mentions[i]:
            if not do_lower:
              mention_word.append(lines[i]["tokens"][int(mention['start']):int(mention['end'])])
            else:
              words=lines[i]["tokens"][int(mention['start']):int(mention['end'])]
              words=[w.lower() for w in words]
              mention_word.append(words)

            temp=[]
            coarse_temp=[]
            fine_grained_temp=[]
            label_name = []
            for iid, y_strs in enumerate(mention["labels"]):
              label_name.append(y_strs)
              if y_strs in word2id.keys():
                temp.append(
                  word2id[y_strs])
              else:
                temp.append(len(word2id))
              if y_strs in coatse2id.keys():
                coarse_temp.append(coatse2id[y_strs])
              if y_strs in fine_grained2id.keys():
                fine_grained_temp.append(fine_grained2id[y_strs])
            temp = list(set(temp))
            if len(temp) == 1:
              temp = temp*2
            assert len(temp) ==2
            #assert len(coarse_temp) != 0
            y_category.append(temp)
            coarse_label.append(coarse_temp)
            if len(coarse_temp) == 1 and len(fine_grained_temp) == 0 and len(label_name) == 1:
              coarse_l = label_name[0]
              coarse_l = coarse_l+"/OTHER"
              fine_grained_temp.append(fine_grained2id[coarse_l])
            fine_grained_label.append(fine_grained_temp)
            left=lines[i]["tokens"][0:int(mention['start'])]
            right=lines[i]["tokens"][int(mention['end']):]
            if not do_lower:
              left=left[-context_window_size:]
              right = right[:context_window_size]
            else:
              left=[w.lower() for w in left]
              right = [w.lower() for w in right]
              left = left[-context_window_size:]
              right = right[:context_window_size]
            left_context.append(left)
            right_context.append(right)
            idx_list.append(idx)
            idx = idx+1

        return left_context,right_context,mention_word,y_category,idx_list,coarse_label,fine_grained_label





def get_id_data(left_context, right_context, mention_word, y_category, idx_list,answer_num):
  id_left_context=[]
  id_right_context=[]
  id_mention_word=[]
  id_y_category=[]
  id_idx_list=[]
  for i in range(len(y_category)):
    if answer_num not in y_category[i]:
      id_y_category.append(y_category[i])
      id_left_context.append(left_context[i])
      id_right_context.append(right_context[i])
      id_mention_word.append(mention_word[i])
      id_idx_list.append(idx_list[i])
  return id_left_context,id_right_context,id_mention_word,id_y_category,id_idx_list


def map_label(fine_grained2id, word2id):
  map_dict=defaultdict()
  for k,v in fine_grained2id.items():
    fine_grained_names = k.split('/')
    coarse_name = '/'+fine_grained_names[1]
    if k in word2id.keys():
      map_dict[v]=[word2id[k],word2id[coarse_name]]
    else:
      map_dict[v] = [word2id[coarse_name]]
  return map_dict
def map_label2(fine_grained2id, word2id):
  map_dict=defaultdict()
  for k,v in fine_grained2id.items():
    fine_grained_names = k.split('-')
    coarse_name = fine_grained_names[0]
    if k in word2id.keys():
      map_dict[v]=[word2id[k],word2id[coarse_name]]
    else:
      map_dict[v] = [word2id[coarse_name]]
  return map_dict
class BertDataset(Dataset):
  def __init__(self, max_token=512, device='cpu', tokenizer=None, data_path=None, do_lower=True, data_name=None,id_flag=True):
    self.device = device
    super(BertDataset, self).__init__()
    self.word2id = constant.load_vocab_dict(constant.TYPE_FILES[data_name])
    self.coatse2id = constant.load_vocab_dict(constant.TYPE_FILES[data_name+'_c'])
    self.fine_grained2id = constant.load_vocab_dict(constant.TYPE_FILES[data_name+'_f'])


    self.left_context, self.right_context, self.mention_word, self.y_category, self.idx_list,self.coarse_labels,self.fine_grained_labels= _load_shard(data_path,
                                                                                                           max_token,
                                                                                                           do_lower,
                                                                                                          self.word2id,self.coatse2id,self.fine_grained2id)
    self.tokenizer = tokenizer
    # self.corpus=[' ']*len(self.word2id)
    # for i in range(len(self.idx_list)):
    #   for y in self.y_category[i]:
    #     if y < len(self.corpus):
    #       sent=' '.join(self.left_context[i]).lower() + ' '.join(self.mention_word[i]).lower() + ' '.join(
    #       self.right_context[i]).lower()
    #       sent_word = self.tokenizer.tokenize(sent)
    #       self.corpus[y] += ' ' + ' '.join(sent_word)
    # vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
    # self.tfidf = vectorizer.fit_transform(self.corpus).toarray()
    # self.vocab = list(vectorizer.get_feature_names_out())
    # print('[TF-IDF]:', self.tfidf.shape)

    # self.tfidf_dict = defaultdict(dict)
    # for i in range(self.tfidf.shape[0]):
    #   for j in range(self.tfidf.shape[1]):
    #     self.tfidf_dict[i][self.vocab[j]] = self.tfidf[i][j]
    # self.filter_list=[]
    # for i in range(self.tfidf.shape[0]):
    #   sorted_dict = list(OrderedDict(sorted(self.tfidf_dict[i].items(), key=lambda item: item[1], reverse=True)).keys())
    #   sorted_dict = [k for k in sorted_dict if '##' not in k][0:100]
    #   self.filter_list.extend(sorted_dict)
    self.max_token = max_token

    self.answer_num = constant.ANSWER_NUM_DICT[data_name]
    self.length = len([y for y in self.y_category if y == [self.answer_num]])
    # if id_flag == True:
    #   self.left_context, self.right_context, self.mention_word, self.y_category, self.idx_list = \
    #     get_id_data(self.left_context, self.right_context, self.mention_word, self.y_category, self.idx_list,self.answer_num)

  def __getitem__(self, item):
    left_context = self.left_context[item]
    right_context = self.right_context[item]
    mention_word = self.mention_word[item]
    label = self.y_category[item]
    coarse_label = self.coarse_labels[item]
    fine_grained_label = self.fine_grained_labels[item]
    idx = self.idx_list[item]
    return {'left_context': left_context, 'right_context': right_context, 'mention_word': mention_word, 'label': label,
            'idx': idx, 'coarse_label': coarse_label, 'fine_grained_label': fine_grained_label}

  def __len__(self):
    return len(self.mention_word)

  ##如果你的数据集中的所有样本都具有相同的大小，你可能可以省略 collate_fn。
  # 然而，在许多情况下，特别是在自然语言处理任务中，文本序列的长度通常是不同的。
  # 在这种情况下，collate_fn 可以确保数据在批次中被正确地对齐和填充，以便输入到模型。
  def tokenizer_word(self,word_list,max_length=0):
    token_list = []
    for word in word_list:
      tokens = self.tokenizer.wordpiece_tokenizer.tokenize(word)
      token_list.extend(tokens)
    for i in range(max_length-len(token_list)):
      token_list.append('[PAD]')
    token_list.append('[SEP]')

    return self.tokenizer.convert_tokens_to_ids(token_list)
  def collate_fn(self, batch):
    mention_length_limit = 20
    if not isinstance(batch, list):
      targets = np.zeros(self.answer_num, np.float32)


      left_seq = batch['left_context']
      right_seq = batch['right_context']
      mention_seq = batch['mention_word']
      label = batch['label']

      idx = batch['idx']
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]

      context = left_seq + mention_seq + right_seq
      mention_context = ['[CLS]']+mention_seq+['[SEP]']+context
      len_after_tokenization = len(
        self.tokenizer_word(mention_context))

      if len_after_tokenization > self.max_token:
        overflow_len = len_after_tokenization - self.max_token
        context = left_seq + mention_seq + right_seq[:-overflow_len]
      mention_context = ['[CLS]']+mention_seq+['[SEP]']+context
      id_ids = self.tokenizer_word(mention_context).to(self.device)
      id_masks = (id_ids !=0).float().to(self.device)


      targets[label] = 1.0

      inputs = defaultdict()

      inputs['input_ids']=id_ids
      inputs['attention_mask'] = id_masks


      targets = torch.from_numpy(targets).to(self.device)



      return inputs, targets, idx

    targets = np.zeros([len(batch), self.answer_num], np.float32)

    id_targets = []
    all_context = []

    idx_list = []
    sentence_len_wp = []
    ood_sent_len=[]
    i = 0
    for b in batch:
      left_seq = b['left_context']
      right_seq = b['right_context']
      mention_seq = b['mention_word']
      label = b['label']

      idx_list.append(b['idx'])
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]
      # mention = ' '.join(mention_seq)
      context = left_seq + mention_seq + right_seq


      mention_context = ['[CLS]'] + mention_seq + ['[SEP]'] + context
      len_after_tokenization = len(
        self.tokenizer_word(mention_context))
      if len_after_tokenization > self.max_token:
        overflow_len = len_after_tokenization - self.max_token
        context = left_seq + mention_seq + right_seq[:-overflow_len]
      mention_context = ['[CLS]'] + mention_seq + ['[SEP]'] + context
      all_context.append(mention_context)


      len_after_tokenization = len(
        self.tokenizer_word(mention_context))
      sentence_len_wp.append(len_after_tokenization)

      id_targets.append(label)
      for answer_ind in label:
          if answer_ind < self.answer_num:
            targets[i, answer_ind] = 1.0
          #ood_labels[i,-1]=1.0
      i = i + 1

    max_len_in_batch = max(sentence_len_wp)

    all_id_ids = []

    for id_context in all_context:
      all_id_ids.append(self.tokenizer_word(id_context,max_len_in_batch))



    targets = torch.from_numpy(targets)
    idx_list = torch.from_numpy(np.array(idx_list))
    all_id_ids=torch.from_numpy(np.array(all_id_ids)).to(self.device)

    id_masks = (all_id_ids != 0).float().to(self.device)

    inputs = defaultdict()

    inputs['input_ids'] = all_id_ids
    inputs['attention_mask'] = id_masks


    return inputs, targets.to(self.device),idx_list.to(self.device)


def load_ood(data_path):
  all_files = os.listdir(data_path)
  all_data=[]
  if 'BBN' in data_path:
    filtered_files = [file for file in all_files if file.startswith('base_ood_generate_data')]
  else:
    filtered_files = [file for file in all_files if file.startswith('base_ood_generate_data')]
  for file in filtered_files:
    with open(os.path.join(data_path,file), 'rb') as f:
      load_data = pickle.load(f)
      all_data.extend(load_data)
  #random.shuffle(all_data)
  all_data = sorted(all_data, key=lambda x: x['idx'])
  return all_data

def load_llm_ood(data_path):
  # all_files = os.listdir(data_path)
  # all_data=[]
  # filtered_files = [file for file in all_files if file.startswith('base_ood_generate_data_new')]
  # for file in filtered_files:
  #   with open(os.path.join(data_path,file), 'rb') as f:
  #     load_data = pickle.load(f)
  #     all_data.extend(load_data)
  # #random.shuffle(all_data)
  # all_data = sorted(all_data, key=lambda x: x['idx'])

  llm_ood_file = os.path.join(data_path,'llm_ood_llama_dpp_valid_w.txt')

  with open(llm_ood_file,'r',encoding='utf-8')  as r:
     lines = r.readlines()
  lines = [line.strip() for line in lines]

  return lines

import math
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
special_words=['[CLS]','[SEP]','[PAD]','.','',"'s",',',"``",'--','$','%','?','&','',"''"]
class TfIdf:
    def __init__(self):
        self.num_docs = 0
        self.vocab = {}
    def filter(self,corpus):
      filter_corpus=[]
      for sent in corpus:
        filter_corpus.append([s for s in sent if s not in stop_words and special_words])
      return filter_corpus
    def add_corpus(self, corpus):
        corpus = self.filter(corpus)
        self._merge_corpus(corpus)

        tfidf_list = []
        for sentence in corpus:
            key_dict = self.get_tfidf(sentence)
            top_k = int(0.2*len(key_dict))
            key_dict = sorted(key_dict.items(), key=lambda x: x[1], reverse=True)
            key_dict = dict(key_dict[0:top_k])
            tfidf_list.append(key_dict)

        return tfidf_list

    def _merge_corpus(self, corpus):
        """
        统计语料库，输出词表，并统计包含每个词的文档数。
        """
        self.num_docs = len(corpus)
        for sentence in corpus:
            words = sentence
            words = set(words)
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0.0) + 1.0

    def _get_idf(self, term):
        """
        计算 IDF 值
        """
        return math.log(self.num_docs / (self.vocab.get(term, 0.0) + 1.0))

    def get_tfidf(self, sentence):
        tfidf = {}
        terms = sentence
        terms_set = set(terms)
        num_terms = len(terms)
        for term in terms_set:
            # 计算 TF 值
            tf = float(terms.count(term)) / num_terms
            # 计算 IDF 值，在实际实现时，可以提前将所有词的 IDF 提前计算好，然后直接使用。
            idf = self._get_idf(term)
            # 计算 TF-IDF 值
            tfidf[term] = tf * idf
        return tfidf

class OODataset(Dataset):
  def __init__(self, max_token=512, device='cpu', tokenizer=None, ood_data_path=None,do_lower=True, data_name=None,args=None):
    self.device = device
    super(OODataset, self).__init__()
    self.args = args


    self.ood_data = load_ood(ood_data_path)
    self.ood_data_filter = ([ood_data for ood_data in self.ood_data  if (ood_data['ood_flag']==True)])
    propotion_num = int(float(args.plm_propotion)*len(self.ood_data_filter))
    self.ood_data_filter=self.ood_data_filter[0:propotion_num]
    self.ood_num = len(self.ood_data_filter)
    self.max_token = max_token
    self.ood_file = open(os.path.join(ood_data_path,'ood_file_new.txt'),'w')
    for od in self.ood_data_filter:
      self.ood_file.write(' '.join(od['replaced_sent'])+'\n')
    self.tokenizer = tokenizer
    self.answer_num = constant.ANSWER_NUM_DICT[data_name]
    self.coarse_num = constant.ANSWER_NUM_DICT[data_name+'_c']
    self.fine_grained_num = constant.ANSWER_NUM_DICT[data_name+'_f']
  def __getitem__(self, item):
    ood_sent = self.ood_data_filter[item]['replaced_sent']
    return {'ood_sent': ood_sent}
  def tokenizer_word(self,word_list,max_length=0):
    token_list = []
    for word in word_list:
      tokens = self.tokenizer.wordpiece_tokenizer.tokenize(word)
      token_list.extend(tokens)
    if len(token_list) > max_length:
      token_list = token_list[0:max_length-1]
    for i in range(max_length-len(token_list)):
      token_list.append('[PAD]')

    token_list.append('[SEP]')

    return self.tokenizer.convert_tokens_to_ids(token_list)
  def __len__(self):
    return len(self.ood_data_filter)
  def collate_fn(self, batch):
    mention_length_limit = 20
    if not isinstance(batch, list):
      ood_labels = np.zeros(self.answer_num, np.float32)
      ood_sent = batch['ood_sent']
      ood_sent = ood_sent[0:-1]
      ood_ids = self.tokenizer_word(ood_sent).to(self.device)
      ood_masks = (ood_ids != 0).float().to(self.device)
      ood_input = defaultdict()

      ood_input['input_ids'] = ood_ids
      ood_input['attention_mask'] = ood_masks


      ood_labels = torch.from_numpy(ood_labels).to(self.device)
      return ood_input, ood_labels


    ood_labels = np.zeros([len(batch), self.answer_num], np.float32)

    all_ood_context=[]

    ood_sent_len=[]
    i = 0
    for b in batch:
      ood_sent = b['ood_sent']
      ood_sent=ood_sent[0:-1]
      all_ood_context.append(ood_sent)
      ood_len = len(
        self.tokenizer_word(ood_sent))
      ood_sent_len.append(ood_len)
      i = i + 1
    max_len_ood_in_batch = max(ood_sent_len)
    all_ood_ids = []
    for ood_context in all_ood_context:
      all_ood_ids.append(self.tokenizer_word(ood_context,max_len_ood_in_batch))
    all_ood_ids = torch.from_numpy(np.array(all_ood_ids)).to(self.device)

    ood_labels = torch.from_numpy(ood_labels).to(self.device)
    ood_masks = (all_ood_ids != 0).float().to(self.device)
    ood_input = defaultdict()
    ood_input['input_ids'] = all_ood_ids
    ood_input['attention_mask'] = ood_masks

    return ood_input,ood_labels
class LLM_OODataset(Dataset):
  def __init__(self, max_token=510, device='cpu', tokenizer=None, ood_data_path=None,do_lower=True, data_name=None,args=None):
    self.device = device
    super(LLM_OODataset, self).__init__()
    self.args = args


    self.ood_data = load_llm_ood(ood_data_path)
    #self.ood_data_filter = ([ood_data for ood_data in self.ood_data  if (ood_data['ood_flag']==True)])
    propotion_num = int(float(args.llm_propotion) * len(self.ood_data))
    self.ood_data = self.ood_data[0:propotion_num]
    self.ood_num = len(self.ood_data)
    self.max_token = max_token

    self.tokenizer = tokenizer
    self.answer_num = constant.ANSWER_NUM_DICT[data_name]
    self.coarse_num = constant.ANSWER_NUM_DICT[data_name+'_c']
    self.fine_grained_num = constant.ANSWER_NUM_DICT[data_name+'_f']
  def __getitem__(self, item):
    ood_sent = self.ood_data[item]
    return {'ood_sent': ood_sent}

  def tokenizer_word(self,word_list,max_length=0):
    token_list = []
    for word in word_list:
      tokens = self.tokenizer.wordpiece_tokenizer.tokenize(word)
      token_list.extend(tokens)
    if len(token_list) > max_length:
      token_list = token_list[0:max_length-1]
    for i in range(max_length-len(token_list)):
      token_list.append('[PAD]')

    token_list.append('[SEP]')

    return self.tokenizer.convert_tokens_to_ids(token_list)
  def __len__(self):
    return len(self.ood_data)
  def collate_fn(self, batch):
    mention_length_limit = 20
    if not isinstance(batch, list):
      ood_labels = np.zeros(self.answer_num, np.float32)
      ood_sent = batch['ood_sent'].split(' ')
      ood_sent = ood_sent[0:-1]
      ood_ids = self.tokenizer_word(ood_sent).to(self.device)
      ood_masks = (ood_ids != 0).float().to(self.device)
      ood_input = defaultdict()

      ood_input['input_ids'] = ood_ids
      ood_input['attention_mask'] = ood_masks


      ood_labels = torch.from_numpy(ood_labels).to(self.device)
      return ood_input, ood_labels


    ood_labels = np.zeros([len(batch), self.answer_num], np.float32)

    all_ood_context=[]

    ood_sent_len=[]
    i = 0
    for b in batch:
      ood_sent = b['ood_sent'].split(' ')
      ood_sent=ood_sent[0:-1]
      all_ood_context.append(ood_sent)
      ood_len = len(
        self.tokenizer_word(ood_sent))
      ood_sent_len.append(ood_len)
      i = i + 1
    max_len_ood_in_batch = max(ood_sent_len)
    if max_len_ood_in_batch > self.max_token:
      max_len_ood_in_batch=self.max_token
    all_ood_ids = []
    for ood_context in all_ood_context:
      all_ood_ids.append(self.tokenizer_word(ood_context,max_len_ood_in_batch))
    all_ood_ids = torch.from_numpy(np.array(all_ood_ids)).to(self.device)

    ood_labels = torch.from_numpy(ood_labels).to(self.device)
    ood_masks = (all_ood_ids != 0).float().to(self.device)
    ood_input = defaultdict()
    ood_input['input_ids'] = all_ood_ids
    ood_input['attention_mask'] = ood_masks

    return ood_input,ood_labels
class BertDataset_OOD(Dataset):
  def __init__(self, max_token=512, device='cpu', tokenizer=None, ood_data_path =None,data_path=None,do_lower=True, data_name=None,args=None,flag=None):
    self.device = device
    super(BertDataset_OOD, self).__init__()
    self.args = args
    self.word2id,_ = constant.load_vocab_dict(constant.TYPE_FILES[data_name])
    self.coatse2id,_ = constant.load_vocab_dict(constant.TYPE_FILES[data_name+'_c'])
    self.fine_grained2id,_ = constant.load_vocab_dict(constant.TYPE_FILES[data_name+'_f'])

    self.left_context, self.right_context, self.mention_word, self.y_category, self.idx_list,self.coarse_labels,self.fine_grained_labels = _load_shard(data_path,
                                                                                                           max_token,
                                                                                                           do_lower,
                                                                                                          self.word2id,self.coatse2id,self.fine_grained2id)
    #self.ood_data = load_ood(ood_data_path)
    #self.ood_data_filter = ([ood_data for ood_data in self.ood_data  if (ood_data['ood_flag']==True)])
   # self.ood_num = len(self.ood_data_filter)
    #self.known_file = open(os.path.join(ood_data_path,'known_file.txt'),'w')
    self.context=[]
    for i in range(len(self.mention_word)):
      context = self.left_context[i]+self.mention_word[i]+self.right_context[i]
      self.context.append(context)
    if data_name == 'BBN':
      self.map_ids = map_label(self.fine_grained2id, self.word2id)
    else:
      self.map_ids = map_label2(self.fine_grained2id, self.word2id)
    self.max_token = max_token
    self.tokenizer = tokenizer
    self.answer_num = constant.ANSWER_NUM_DICT[data_name]
    self.coarse_num = constant.ANSWER_NUM_DICT[data_name+'_c']
    self.fine_grained_num = constant.ANSWER_NUM_DICT[data_name+'_f']
    self.length = len([y for y in self.y_category if y == [self.answer_num]])
    self.corpus = [[] for _ in range(len(self.word2id))]
    for i in range(len(self.idx_list)):
      for y in self.y_category[i]:
        if y < len(self.corpus):
          sent = self.left_context[i] + self.mention_word[i] + self.right_context[i]
          sent = [s.lower() for s in sent]
          self.corpus[y].extend(sent)
    self.tfidf = TfIdf()
    self.tfidf_values = self.tfidf.add_corpus(self.corpus)

    self.filter_list = set()
    for tfidf_dict in self.tfidf_values:
        tfidf_dict = dict(sorted(tfidf_dict.items(),key = lambda item:item[1],reverse=True)[0:int(len(tfidf_dict.items())*0.1)])
        self.filter_list = self.filter_list.union(set(tfidf_dict.keys()))
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    self.filter_list = [item for item in self.filter_list if item not in stop_words and has_english_char(item)]
    if flag == None:
      self.coarse_negative_dataset,self.fine_grained_negative_dataset,self.class_num= self.create_negative_dataset()


  def __getitem__(self, item):
    left_context = self.left_context[item]
    right_context = self.right_context[item]
    mention_word = self.mention_word[item]
    label = self.y_category[item]
    coarse_label = self.coarse_labels[item]
    fine_grained_label = self.fine_grained_labels[item]
    idx = self.idx_list[item]
    #ood_sent = self.ood_data_filter[item%self.ood_num]['replaced_sent']
    #ood_sent = self.ood_data[idx]['replaced_sent']
    return {'left_context': left_context, 'right_context': right_context, 'mention_word': mention_word, 'label': label,
            'idx': idx,'coarse_label':coarse_label,'fine_grained_label':fine_grained_label}
  def tokenizer_word(self,word_list,max_length=0):
    token_list = []
    for word in word_list:
      tokens = self.tokenizer.wordpiece_tokenizer.tokenize(word)
      token_list.extend(tokens)
    for i in range(max_length-len(token_list)):
      token_list.append('[PAD]')
    token_list.append('[SEP]')

    return self.tokenizer.convert_tokens_to_ids(token_list)
  def __len__(self):
    return len(self.mention_word)

  ##如果你的数据集中的所有样本都具有相同的大小，你可能可以省略 collate_fn。
  # 然而，在许多情况下，特别是在自然语言处理任务中，文本序列的长度通常是不同的。
  # 在这种情况下，collate_fn 可以确保数据在批次中被正确地对齐和填充，以便输入到模型。
  def collate_fn(self, batch):
    mention_length_limit = 20
    if not isinstance(batch, list):
      targets = np.zeros(self.answer_num, np.float32)
      coarse_targets = np.zeros(self.coarse_num, np.float32)
      fine_grained_targets = np.zeros(self.fine_grained_num, np.float32)

      left_seq = batch['left_context']
      right_seq = batch['right_context']
      mention_seq = batch['mention_word']
      label = batch['label']
      coarse_label = batch['coarse_label']
      fine_grained_label = batch['fine_grained_label']
      id_targets = np.array(label)
      # ood_targets = np.array([self.answer_num] *2)
      # ood_labels = np.zeros(self.answer_num, np.float32)
      idx = batch['idx']
      # ood_sent = batch['ood_sent']
      # ood_sent = ood_sent[0:-1]
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]
      # mention = ' '.join(mention_seq)
      # context = ' '.join(left_seq + mention_seq + right_seq)
      # ood_sent = ood_sent[1:-1]
      # ood_sent_str = ' '.join(ood_sent)
      context = left_seq + mention_seq + right_seq
      mention_context = ['[CLS]']+mention_seq+['[SEP]']+context
      len_after_tokenization = len(
        self.tokenizer_word(mention_context))

      if len_after_tokenization > self.max_token:
        overflow_len = len_after_tokenization - self.max_token
        context = left_seq + mention_seq + right_seq[:-overflow_len]
      mention_context = ['[CLS]']+mention_seq+['[SEP]']+context
      id_ids = self.tokenizer_word(mention_context).to(self.device)
      id_masks = (id_ids !=0).float().to(self.device)
      # ood_ids = self.tokenizer_word(ood_sent).to(self.device)
      # ood_masks = (ood_ids != 0).float().to(self.device)
      # len_after_tokenization = len(input_ids)
      # if label < self.answer_num:
      targets[label] = 1.0
      coarse_targets[coarse_label]=1.0
      fine_grained_targets[fine_grained_label]=1.0
      inputs = defaultdict()
      # ood_input = defaultdict()
      inputs['input_ids']=id_ids
      inputs['attention_mask'] = id_masks
      # ood_input['input_ids'] = ood_ids
      # ood_input['attention_mask'] = ood_masks
      targets = torch.from_numpy(targets).to(self.device)
      coarse_targets = torch.from_numpy(coarse_targets).to(self.device)
      fine_grained_targets = torch.from_numpy(fine_grained_targets).to(self.device)
      # ood_targets = torch.from_numpy(ood_targets).to(self.device)
      id_targets = torch.from_numpy(id_targets).to(self.device)
      # ood_labels = torch.from_numpy(ood_labels).to(self.device)
      return inputs, targets, idx,id_targets,coarse_targets,fine_grained_targets

    targets = np.zeros([len(batch), self.answer_num], np.float32)
    coarse_targets = np.zeros([len(batch), self.coarse_num], np.float32)
    fine_grained_targets = np.zeros([len(batch), self.fine_grained_num], np.float32)
    # ood_targets = np.full((len(batch),2),self.answer_num)
    # ood_labels = np.zeros([len(batch), self.answer_num], np.float32)
    id_targets = []
    all_context = []
    # all_ood_context=[]
    idx_list = []
    sentence_len_wp = []
    # ood_sent_len=[]
    i = 0
    for b in batch:
      left_seq = b['left_context']
      right_seq = b['right_context']
      mention_seq = b['mention_word']
      label = b['label']
      coarse_label = b['coarse_label']
      fine_grained_label = b['fine_grained_label']
      # ood_sent = b['ood_sent']
      idx_list.append(b['idx'])
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]
      # mention = ' '.join(mention_seq)
      context = left_seq + mention_seq + right_seq
      # ood_sent=ood_sent[0:-1]
      # ood_sent_str = ' '.join(ood_sent)

      mention_context = ['[CLS]'] + mention_seq + ['[SEP]'] + context
      len_after_tokenization = len(
        self.tokenizer_word(mention_context))
      if len_after_tokenization > self.max_token:
        overflow_len = len_after_tokenization - self.max_token
        context = left_seq + mention_seq + right_seq[:-overflow_len]
      mention_context = ['[CLS]'] + mention_seq + ['[SEP]'] + context
      all_context.append(mention_context)
      # all_ood_context.append(ood_sent)

      len_after_tokenization = len(
        self.tokenizer_word(mention_context))
      sentence_len_wp.append(len_after_tokenization)
      # ood_len = len(
      #   self.tokenizer_word(ood_sent))
      # ood_sent_len.append(ood_len)
      id_targets.append(label)
      for answer_ind in label:
        if answer_ind < self.answer_num:
          targets[i, answer_ind] = 1.0
          #ood_labels[i,-1]=1.0
      for c_ind in coarse_label:
          coarse_targets[i, c_ind] = 1.0
          #ood_labels[i,-1]=1.0
      for f_ind in fine_grained_label:
          fine_grained_targets[i, f_ind] = 1.0
          #ood_labels[i,-1]=1.0
      i = i + 1
    id_targets = np.array(id_targets)
    max_len_in_batch = max(sentence_len_wp)
    # max_len_ood_in_batch = max(ood_sent_len)
    all_id_ids = []
    # all_ood_ids = []
    for id_context in all_context:
      all_id_ids.append(self.tokenizer_word(id_context,max_len_in_batch))
    # for ood_context in all_ood_context:
    #   all_ood_ids.append(self.tokenizer_word(ood_context,max_len_ood_in_batch))


    targets = torch.from_numpy(targets).to(self.device)
    coarse_targets = torch.from_numpy(coarse_targets).to(self.device)
    fine_grained_targets = torch.from_numpy(fine_grained_targets).to(self.device)
    idx_list = torch.from_numpy(np.array(idx_list)).to(self.device)
    all_id_ids=torch.from_numpy(np.array(all_id_ids)).to(self.device)
    # all_ood_ids = torch.from_numpy(np.array(all_ood_ids))
    # ood_targets = torch.from_numpy(ood_targets).to(self.device)
    id_targets = torch.from_numpy(id_targets).to(self.device)
    # ood_labels = torch.from_numpy(ood_labels).to(self.device)
    id_masks = (all_id_ids != 0).float().to(self.device)
    # ood_masks = (all_ood_ids != 0).float().to(self.device)
    inputs = defaultdict()
    # ood_input = defaultdict()
    inputs['input_ids'] = all_id_ids
    inputs['attention_mask'] = id_masks
    # ood_input['input_ids'] = all_ood_ids
    # ood_input['attention_mask'] = ood_masks

    return inputs, targets, idx_list,id_targets,coarse_targets,fine_grained_targets

  def create_negative_dataset(self):
    coarse_negative_dataset = {}
    fine_grained_negative_dataset = {}
    class_num=[0]*len(self.fine_grained2id)
    data = zip(self.left_context, self.right_context, self.mention_word, self.y_category, self.idx_list,self.coarse_labels,self.fine_grained_labels)

    for line in data:
      coarse_label = line[5][0]
      fine_grained_label = line[6][0]
      line_dict = {'left_context': line[0], 'right_context': line[1], 'mention_word': line[2], 'label': line[3],
       'idx': line[4], 'coarse_label': line[5], 'fine_grained_label': line[6]}
      if coarse_label not in coarse_negative_dataset.keys():
        coarse_negative_dataset[coarse_label] = [line_dict]
      else:
        coarse_negative_dataset[coarse_label].append(line_dict)

      class_num[int(fine_grained_label)] +=1
      if fine_grained_label not in fine_grained_negative_dataset.keys():
        fine_grained_negative_dataset[fine_grained_label] = [line_dict]
      else:
        fine_grained_negative_dataset[fine_grained_label].append(line_dict)

    return coarse_negative_dataset,fine_grained_negative_dataset,class_num
  def generate_positive_samples(self,coarse_labels,fine_grained_labels):
    positive_num = self.args.positive_num  # 3
    # positive_num = 16
    coarse_positive_sample = []
    for index in range(len(coarse_labels)):
      coarse_label = int(coarse_labels[index][0])
      ##这个地方有可能采样的数量大于list中的数量，所以需要改成可重复的采样
      g_index = range(len(self.coarse_negative_dataset[coarse_label]))
      choose_indexs = np.random.choice(g_index, positive_num)
      coarse_positive_sample.extend([self.coarse_negative_dataset[coarse_label][i] for i in choose_indexs])

    fine_grained_positive_sample = []
    for index in range(len(fine_grained_labels)):
      fine_grained_label = int(fine_grained_labels[index][0])
      ##这个地方有可能采样的数量大于list中的数量，所以需要改成可重复的采样
      g_index = range(len(self.fine_grained_negative_dataset[fine_grained_label]))
      choose_indexs = np.random.choice(g_index, positive_num)
      fine_grained_positive_sample.extend([self.fine_grained_negative_dataset[fine_grained_label][i] for i in choose_indexs])

    return coarse_positive_sample,fine_grained_positive_sample


