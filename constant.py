import torch
import json
import os

from typing import Dict, Optional

import utils

utils.seed_torch(0)
os.environ['PYTHONHASHSEED']=str(0)
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

ANSWER_NUM_DICT = {"ufet": 10331, "onto": 89, "figer": 113, "BBN": 27,"fnerd":37,"BBN_c":7,"fnerd_c":4,"BBN_f":27,"fnerd_f":33}

# Specify paths here
BASE_PATH = "/"
FILE_ROOT = "data/"
EXP_ROOT = "save_model"
ONTOLOGY_DIR = "../data/ontology"

TYPE_FILES = {

    "fnerd": os.path.join(FILE_ROOT, "fnerd/fnerd_types_train_hierachy.txt"),
    "BBN": os.path.join(FILE_ROOT, "BBN/bbn_types_train_hierachy.txt"),
    "BBN_c": os.path.join(FILE_ROOT, "BBN/bbn_types_train_coarse.txt"),
    "BBN_f": os.path.join(FILE_ROOT, "BBN/bbn_types_train_fine_grained.txt"),
    "BBN_e": os.path.join(FILE_ROOT, "BBN/bbn_types_train_for_embed.txt"),
    "fnerd_c": os.path.join(FILE_ROOT, "fnerd/fnerd_types_train_coarse.txt"),
    "fnerd_f": os.path.join(FILE_ROOT, "fnerd/fnerd_types_train_fine_grained.txt"),
    "fnerd_e": os.path.join(FILE_ROOT, "fnerd/fnerd_types_train_for_embed.txt"),
}


def load_vocab_dict(
  vocab_file_name: str,
  vocab_max_size: Optional[int] = None,
  start_vocab_count: Optional[int] = None,
  common_vocab_file_name: Optional[str] = None
):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    file_content = dict(zip(text, range(0, len(text))))
  return file_content,text


