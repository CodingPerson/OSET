import argparse
import copy
import os

import numpy as np
import openai
from openai import OpenAI
import json
import random

# from transformers.models.qwen2 import Qwen2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import constant
from data_utils import BertDataset_OOD, OODataset

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch

from model.contrast_moco import ContrastiveModelMoco
from model.dpp_map import fast_map_dpp
from ood_utils import compute_ood_batch

# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key='EMPTY'
openai_api_base="http://10.134.104.85:8000/v1"
#client = ZhipuAI(api_key="779e65dd2f4682db29115a7a34f8f46c.4FVUKxMrYehjTBgi")
client = OpenAI(api_key=openai_api_key,base_url=openai_api_base)
# os.environ["HTTP_PROXY"]="http://127.0.0.1:10809"
# os.environ["HTTPS_PROXY"]="http://127.0.0.1:10809"
def cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)
    return res_data
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
parser.add_argument('--weight_flag',  default=True,type=bool, help='Whether use open loss.')
parser.add_argument('--cls_loss',  default=True,type=bool, help='Whether use cls loss.')
parser.add_argument('--layer', default=2, type=int, help='Layer of Graphormer.')
parser.add_argument('--lamb', default=0.0, type=float, help='lambda')
parser.add_argument('--open_lamda', default=1.0, type=float, help='lambda')
parser.add_argument('--p_cutoff', default=0.5, type=float, help='lambda')
parser.add_argument('--thre', default=0.0001, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
parser.add_argument('--seed', default=6, type=int, help='Random seed.')
parser.add_argument('--max_num', type=int, default=200, help='mask word num')
parser.add_argument('--max_epoch', type=int, default=50, help='mask word num')
parser.add_argument('--positive_num', type=int, default=3, help='positive num sample')
parser.add_argument('--extra', default='replace', choices=['acc,f1,replace'], help='An extra string in the name of checkpoint.')
parser.add_argument('--topk', type=float, default=1, help='demenstration top')
parser.add_argument('--scale_factor', type=float, default=0.5, help='demenstration top')


def prompt_format(ins):
    import json
    mention = ins[0]
    context = ins[1]
    mention_index = context.index(mention)
    mention_length = len(mention)

    new_context = context[0:mention_index]+' [ENT]'+mention+'[\ENT]'+context[mention_index+mention_length:]+'\n'


    demenstration = "Sentence:\n"
    demenstration+=new_context
    demenstration+="Entity in above sentence:\n"
    demenstration+=mention


    return demenstration
def get_middle_string(main_string,sub_string1,sub_string2):
    results=[]
    start_index=0
    while True:
        start_index = main_string.find(sub_string1,start_index)
        if start_index == -1:
            break
        start_index += len(sub_string1)
        end_index = main_string.find(sub_string2,start_index)
        if end_index == -1:
            break
        results.append(main_string[start_index:end_index].strip())
        start_index = end_index+len(sub_string2)
    return results
def process_synsetic_samples(raw_sample):
    raw_sample = raw_sample.strip()
    all_sentences = get_middle_string(raw_sample,'Sentence:','Entity in above sentence:')
    filter_sentence = []
    filter_mentions = []
    inputs=[]
    for sentence in all_sentences:
        if '[ENT]' in sentence and '[\ENT]' in sentence:
            mentions = get_middle_string(sentence,'[ENT]','[\ENT]')
            filter_sentence.append(sentence)
            filter_mentions.append(mentions)

    for i in range(len(filter_sentence)):
        mentions = copy.deepcopy(filter_mentions[i])
        cur_sentence = copy.deepcopy(filter_sentence[i])
        cur_tokens = cur_sentence.split(' ')
        context = [item for item in cur_tokens if '[ENT]' not in item and '[\ENT]' not in item]
        for mention in mentions:
            str_token = '[CLS] '+mention+' [SEP] '+' '.join(context)+' [SEP]'
            inputs.append(str_token)
    # if raw_sample.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
    #     raw_sample = raw_sample[3:]
    # elif raw_sample.startswith(
    #         ("10. ", "11. ", "12. ", "13. ", "14. ", "15. ", "16. ", "17. ", "18. ", "19. ", "20. ")):
    #     raw_sample = raw_sample[4:]
    return inputs
def get_ood_score(mention_context,model,ood_loader):
    ood_pbar = tqdm(ood_loader)
    logits = None
    logits_mbs=None
    pooled_outs=None
    model.eval()
    with torch.no_grad():
        for ood_inputs,ood_labels in ood_pbar:
            output=model.evaluate(input_ids=ood_inputs["input_ids"], attention_mask=ood_inputs["attention_mask"],labels=ood_labels, return_dict=True,is_train=False)
            if logits != None:
                logits = torch.cat((logits, output['logits']), dim=0)
            else:
                logits = output['logits']
            if logits_mbs != None:
                logits_mbs = torch.cat((logits_mbs, output['logits_mb_cls']), dim=0)
            else:
                logits_mbs = output['logits_mb_cls']
            if pooled_outs != None:
                pooled_outs = torch.cat((pooled_outs, output['pooled_out']), dim=0)
            else:
                pooled_outs = output['pooled_out']
    ood_score_dict = compute_ood_batch(logits,origin_logits_mb=logits_mbs, device=args.device)


    score_list=[]
    for index in range(len(mention_context)):
        mention_context_item = mention_context[index]
        mention_ood_score = ood_score_dict['ova'][index]
        score_list.append(mention_ood_score)

    combine_list = list(zip(mention_context,score_list))
    sorted_combine_list = sorted(enumerate(combine_list),key = lambda x:x[1][1],reverse=True)
    sorted_mention_context = [item[1][0] for item in sorted_combine_list]
    sorted_mention_score = [item[1][1] for item in sorted_combine_list]
    sorted_mention_index = [item[0] for item in sorted_combine_list]

    return sorted_mention_context,sorted_mention_score,sorted_mention_index,pooled_outs



def get_kernel(sorted_mention_score, candidates_embed, scale_factor):
    near_reps = candidates_embed
    # normalize first
    # embed = embed / np.linalg.norm(embed)
    near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

    # rel_scores = np.matmul(embed, near_reps.T)[0]
    # to make kernel-matrix non-negative
    rel_scores = (sorted_mention_score + 1) / 2
    # to prevent overflow error
    rel_scores -= rel_scores.max()
    # to balance relevance and diversity
    rel_scores = np.exp(rel_scores / (2 * scale_factor))
    sim_matrix = np.matmul(near_reps, near_reps.T)
    # to make kernel-matrix non-negative
    sim_matrix = (sim_matrix + 1) / 2
    kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
    return near_reps, rel_scores, kernel_matrix
if __name__ == '__main__':
    args = parser.parse_args()

    checkpoint = torch.load(os.path.join('checkpoints', args.data, 'checkpoint_best_acc_PLM_lamb_0.0_ood_True_contrast_True.pt'),
                            map_location=lambda storage, loc: storage.cuda(0))

    train_ood_data_path = os.path.join('checkpoints', args.data)


    device = args.device
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_name = args.data + '/' + 'ood_type_train.json'
    train_data_path = os.path.join('data', train_name)
    data_path = os.path.join('data', args.data)
    ood_dataset = OODataset(device=args.device, tokenizer=tokenizer,ood_data_path = train_ood_data_path,data_name=args.data,args=args)
    train_dataset = BertDataset_OOD(device=device, tokenizer=tokenizer, data_path=train_data_path, data_name=args.data,
                                    args=args)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch, shuffle=False, collate_fn=ood_dataset.collate_fn)
    ood_file = open(os.path.join(train_ood_data_path, 'ood_file_new.txt'), 'r')
    ood_lines = ood_file.readlines()
    model = ContrastiveModelMoco.from_pretrained('bert-base-uncased',num_labels=train_dataset.fine_grained_num,cls_loss=args.cls_loss,
                                          contrast_loss=args.contrast,
                                          lamb=args.lamb,
                                            queue_size=len(train_dataset),knn_num=25,end_k=25,ood_num=25,positive_num=args.positive_num,
                                            coarse_label_num=train_dataset.coarse_num,fine_grained_label_num = train_dataset.fine_grained_num,)
    model.load_state_dict(checkpoint['param'])

    model.to(device)
    model.eval()


    filter_txt = open(args.data+'_filter_ids.txt','r')
    filter_line = filter_txt.readlines()[0]
    filter_ids = [int(f) for f in filter_line.split(' ')]
    logit_bias = {key:-100 for key in filter_ids}
    mention_context = []
    for ood_line in ood_lines:
        splits = ood_line.split('[SEP]')
        entity_mention = splits[0].split('[CLS]')[1]
        context = splits[1].split('[PAD]')[0]
        mention_context.append([entity_mention,context])


    sorted_mention_context,sorted_mention_score,sorted_mention_index,pooled_outs = get_ood_score(mention_context,model,ood_loader)
    topk= int(args.topk*len(mention_context))
    sorted_mention_index_top_k = sorted_mention_index[0:topk]
    sorted_mention_score_top_k = sorted_mention_score[0:topk]
    candidate_embed = pooled_outs[sorted_mention_index_top_k].cpu().detach().numpy()
    sorted_mention_context_topk = sorted_mention_context[0:topk]
    sorted_mention_score_top_k = np.array(sorted_mention_score_top_k)
    choose_results=[]
    exit_index=[]
    while(1):

        near_reps, rel_scores, kernel_matrix=get_kernel(sorted_mention_score_top_k, candidate_embed, args.scale_factor)
        if kernel_matrix.shape[0] <= 5:
            choose_results.append(sorted_mention_context_topk)
            break
        map_results = fast_map_dpp(kernel_matrix, 5)
        map_context = [item for i,item in enumerate(sorted_mention_context_topk) if i in map_results]
        exit_index.extend(map_results)
        choose_results.append(map_context)
        exist_mention_context_topk = [item for i,item in enumerate(sorted_mention_context_topk) if i not in map_results]
        exist_mention_index = [item for item in range(candidate_embed.shape[0]) if item not in map_results]
        candidate_embed = candidate_embed[exist_mention_index]
        sorted_mention_score_top_k = sorted_mention_score_top_k[exist_mention_index]
        sorted_mention_context_topk = copy.deepcopy(exist_mention_context_topk)

    fine_grained2id,_ = constant.load_vocab_dict(constant.TYPE_FILES[args.data + '_e'])
    fine_grained_keys = fine_grained2id.keys()
    fine_grained_keys = [key.lower() for key in fine_grained_keys]
    prompt_suffix ="Please write 20 more sentences, with the same format and similar entity types as above, but with different topics, domains, and diversified entities and semantics. The types of entities in generated sentences are NOT"
    #prompt_suffix = "Please Write 20 more sentences, with the same format as above, but a different topic, domain, and diversified in entities and semantics. The types of entities in generated sentence are NOT"
    prompt_suffix += ' '.join(fine_grained_keys)
    prompt_suffix += "\nDesired format:\nSentence:sentence\nEntity in above sentence:entity\n"
    # prompt_suffix+='.'
    # mention_context_list = cut_list(mention_context,5)
    print("===== Start querying =====")
    all_num=0
    exit_msg = set()
    with open(os.path.join(train_ood_data_path, 'llm_ood_llama_dpp_valid.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '[CLS]' in line and '[SEP]' in line:
                exit_msg.add(line.strip())
                all_num += 1
    f.close()
    for seed in range(1, 10000):
        random.seed(seed)
        prompt = ""
        sampled_ins = choose_results[seed%len(choose_results)-1]
        for ins in sampled_ins:
            prompt += prompt_format(ins)
            prompt += '\n'
        prompt += prompt_suffix

        response = client.chat.completions.create(
            model="openchat",
            messages=[
                # {"role": "system", "content": "You are an intelligent writing assistant."},
                {"role": "user", "content": prompt},
            ],
            seed=seed,
            # prompt=prompt,
            temperature=0.9,
            n=1,
            logit_bias=logit_bias,
            presence_penalty=0.6,
            # max_tokens=3000,
            # stop=['<|eot_id|>']
            top_p=0.9,
        )

        message = response.choices[0].message.content
        # print(message)
        message = process_synsetic_samples(message)
        if message == []:
            print("seed: " + str(seed))
            print(message)
            continue
        replaced_logits_list = []
        replaced_logits_mb_list = []
        replaced_pooled_out_list = []
        for i in range(len(message)):
            msg = message[i]
            # msg = [item for item in msg if '[ENT]' not in item and '[\ENT]' not in item]
            msg_temp = copy.deepcopy(msg)
            # msg_temp = np.array(msg_temp)

            msg_temp_ids = tokenizer.encode(msg_temp, add_special_tokens=False,max_length=512)
            msg_temp_ids = torch.from_numpy(np.array(msg_temp_ids)).to(device)
            msg_temp_ids = msg_temp_ids.unsqueeze(0)
            msg_temp_masks = (msg_temp_ids != 0)
            msg_temp_masks = msg_temp_masks.to(device)
            # replaced_sent = ' '.join(masked_sent_temp)
            # replaced_inputs = tokenizer(
            #     replaced_sent,
            #     return_tensors="pt",
            # ).to(0)
            label = torch.zeros((1,train_dataset.fine_grained_num))
            replaced_output = model.evaluate(input_ids=msg_temp_ids,
                                             attention_mask=msg_temp_masks,
                                             labels=label,
                                             return_dict=True, is_train=False)
            replaced_logits = replaced_output['logits']
            replaced_logits_mb = replaced_output['logits_mb_cls']
            replaced_logits_copy = copy.deepcopy(replaced_logits.detach())
            replaced_logits_mb_copy = copy.deepcopy(replaced_logits_mb.detach())
            replaced_pooled_out = replaced_output['pooled_out']
            replaced_pooled_out_copy = copy.deepcopy(replaced_pooled_out.detach())
            replaced_logits_list.append(replaced_logits_copy)
            replaced_logits_mb_list.append(replaced_logits_mb_copy)
            replaced_pooled_out_list.append(replaced_pooled_out_copy)
        ood_score_dict = compute_ood_batch(torch.cat(replaced_logits_list, dim=0),
                                           origin_logits_mb=torch.cat(replaced_logits_mb_list, dim=0),device=args.device)
        valid_message = []
        for i in range(len(message)):
            if ood_score_dict['ova'][i] > 0.5  and message[i] not in exit_msg:
                valid_message.append(message[i])
                exit_msg.add(message[i])

        print("===== completed query seed {} prompt =====".format(seed))
        all_num +=  len(valid_message)

        with open(os.path.join(train_ood_data_path,'llm_ood_llama_dpp_valid.txt'),'a+') as f:
            # json_str = json.dumps(response)
            for msg in valid_message:
                if '[CLS]' in line and '[SEP]' in line:
                    f.write(msg)
                    f.write("\n")
        if len(exit_msg) > int(len(train_dataset)):
            break
        with open(os.path.join(train_ood_data_path, 'llm_ood_llama_dpp_valid.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '[CLS]' in line and '[SEP]' in line:
                    exit_msg.add(line.strip())
                    all_num += 1
        f.close()