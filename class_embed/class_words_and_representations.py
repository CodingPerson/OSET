

import argparse
import copy
import gc
import itertools
import math
import os
import pickle
import time
from collections import defaultdict, Counter

import numpy
import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score


from preprocessing_utils import load_clean_text, load_tlabels, load_classnames
from tqdm import tqdm

from static_representations import handle_sentence
import pickle as pk
from class_utils import (INTERMEDIATE_DATA_FOLDER_PATH, MODELS,weight_sentence,
                   cosine_similarity_embedding, cosine_similarity_embeddings,
                   evaluate_predictions, tensor_to_numpy, DATA_FOLDER_PATH, ndcg_at_k)


def MinmaxNormalization(mylist):
    if len(mylist) == 1:
        return np.array([1])
    max=np.max(mylist)
    min=np.min(mylist)
    new_list=[]
    for x in mylist:
        new_list.append((x-min)/(max-min))
    return np.array(new_list)


def CountWords_Embeddings(document_statics,document_words,static_word_representations):
    words_id_field = []
    words_embeddings_field=[]
    for doc_ws in document_statics:
        words_id_field.extend(doc_ws)
    words_id_field = list(set(words_id_field))
    for id in words_id_field:
        words_embeddings_field.append(static_word_representations[id])
    return words_id_field,words_embeddings_field



def main(dataset_name, confidence_threshold ,random_state ,lm_type ,layer ,attention_mechanism):
    inter_data_dir = os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name)
    static_repr_path = os.path.join(inter_data_dir, f"static_repr_lm-bbu.pk")
    with open(static_repr_path, "rb") as f:
        vocab = pickle.load(f)
        static_word_representations = vocab["static_word_representations"]
        word_to_index = vocab["word_to_index"]
        vocab_words = vocab["vocab_words"]
    with open(os.path.join(inter_data_dir, f"tokenization_lm-bbu.pk"), "rb") as f:
        tokenization_info = pickle.load(f)["tokenization_info"]
    with open(os.path.join(INTERMEDIATE_DATA_FOLDER_PATH, dataset_name, f"document_repr_lm-bbu.pk"), "rb") as f:
        dictionary = pickle.load(f)
        document_representations = dictionary["document_representations"]
        class_representations = dictionary["class_representations"]

        static_class_representations = dictionary["static_class_representations"]

        document_statics = dictionary["document_statics"]
        class_words = dictionary["class_words"]

        document_context=dictionary["document_context"]

        document_words = dictionary['document_key_tokens']

        document_word_weights = dictionary['document_tokens_weights']

        document_word_embeddings=dictionary['document_tokens_embeddings']

        document_all_words = dictionary['document_all_tokens']

    epoch = 40

    cur_class_representations=[[class_representations[i]] for i in range(len(class_representations))]

    finished_class = [i for i in range(len(class_representations))]
    finished_document = [i for i in range(len(tokenization_info))]


    words_id_field,words_embedding_field=CountWords_Embeddings(document_statics,document_words,static_word_representations)
    exist_words = []
    for i in range(len(class_words)):
        for j in class_words[i]:
            exist_words.append(j)
    start = time.time()
    for itra in range(epoch):
        print("iteration num ："+str(itra))

        if len(finished_class) == 0:
            print("class itera stop ！")


        for i in range(len(class_representations)):
            class_representations[i] = np.array(class_representations[i])
        for i in range(len(static_class_representations)):
            static_class_representations[i] = np.array(static_class_representations[i])

        for i in range(len(class_representations)):
            class_representations[i] = np.array(class_representations[i])
        for i in range(len(static_class_representations)):
            static_class_representations[i] = np.array(static_class_representations[i])





        cluster_similarities=[]
        cluster_nearest_words=[]


        for i in range(len(static_class_representations)):
            cluster_similarities.append(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field)))
            cluster_nearest_words.append(np.argsort(-np.array(cosine_similarity_embeddings([static_class_representations[i]], np.array(words_embedding_field))), axis=1))

        exist_words=[]
        for i in range(len(class_words)):
            for j in class_words[i]:
                exist_words.append(j)


        cur_weights=[[] for i in range(len(class_representations))]

        cur_index = [-1 for i in range(len(class_representations))]
        extended_words = ["" for i in range(len(class_representations))]
        for i in range(len(cluster_nearest_words)):
            if i not in finished_class:
                continue
            new_class_words=cluster_nearest_words[i][0]

            new_index = 0
            for j in range(len(new_class_words)):

                if vocab_words[words_id_field[new_class_words[j]]] not in exist_words:
                    new_index=new_class_words[j]
                    cur_index[i] = new_index

                    extended_words[i]=vocab_words[words_id_field[new_class_words[j]]]
                    class_words[i].append(vocab_words[words_id_field[new_class_words[j]]])
                    exist_words.append(vocab_words[words_id_field[new_class_words[j]]])
                    break
            cur_class_representations[i].append(words_embedding_field[new_index])



        for i in range(len(cur_class_representations)):
            for j in range(len(cur_class_representations[i])):
                cur_weights[i].append(cosine_similarity_embedding(cur_class_representations[i][j],static_class_representations[i]))

        new_class_representations = []
        for i in range(len(cur_class_representations)):
            if i in finished_class:
                new_class_representations.append(
                    np.average(cur_class_representations[i], weights=MinmaxNormalization(cur_weights[i]), axis=0))
            else:
                new_class_representations.append(class_representations[i])

        class_representations = new_class_representations

        if itra <=5 :
            continue
        cluster_similarities = []
        cluster_nearest_words = []
        for i in range(len(static_class_representations)):
            cluster_similarities.append(cosine_similarity_embeddings([class_representations[i]], np.array(words_embedding_field)))
            cluster_nearest_words.append(np.argsort(-np.array(cosine_similarity_embeddings([class_representations[i]], np.array(words_embedding_field))),axis=1))

        for i in range(len(cluster_nearest_words)):
            if i not in finished_class:
                continue
            length = int(len(class_words[i]))
            new_class_words=cluster_nearest_words[i][0][0:length]
            num = 0
            for j in range(len(new_class_words)):
                if vocab_words[words_id_field[new_class_words[j]]] not in class_words[i]:
                    num = num+1
                    if num >= length/4:
                        finished_class.remove(i)
                        cur_class_representations[i].pop()
                        class_words[i].pop()
                        print("finish " + str(i))
                        break
    with open(os.path.join('../data',dataset_name, f"class_words_representations.pk"), "wb") as f:
        pk.dump({
                "static_class_representations": static_class_representations,
                "class_words": class_words,
                "class_representations": np.array(class_representations),
        }, f, protocol=4)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="fnerd")
    parser.add_argument("--confidence_threshold", default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lm_type", type=str, default='bbu')
    parser.add_argument("--layer", type=int, default=12)
    ##chenhu
    parser.add_argument("--attention_mechanism", type=str, default="norm_1_2")
    args = parser.parse_args()
    print(vars(args))
    main(args.dataset_name, args.confidence_threshold, args.random_state, args.lm_type, args.layer,
         args.attention_mechanism)