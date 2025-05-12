# Our Framework

## Environment

* Computational platform: Pytorch 1.13.1, NVIDIA RTX A6000 GPU, CUDA Version 12.4
*  Development language: Python 3.8
* Libraries are listed in requirements.txt, which can be installed via the command `pip install -r requirements.txt`.

## Datasets

We construct two benchmark datasets of the OSET task based on existing fine-grained entity typing datasets (BBN, Few-NERD), which are provided in the folder `data`. The dataset statistics are shown as follows:

| **Dataset**                        | **BBN** | **Few-NERD** |
| ---------------------------------- | ------- | ------------ |
| **Known types**                    | 27      | 37           |
| **Unknown types**                  | 20      | 37           |
| **Training instances**             | 7996    | 35193        |
| **Validation instances**           | 2431    | 31729        |
| **Known-type testing instances**   | 961     | 16176        |
| **Unknown-type testing instances** | 961     | 16176        |


The specific type allocation result for known and unknown types is shown as follows:
| **BBN** | **Few-NERD** |
| --- | --- |
| **Known Types in BBN:**<br>/person, /organization, /location, /gpe, /work_of_art, /product, /animal, /organization/corporation, /organization/educational, /organization/hotel, /organization/government, /organization/hospital, /organization/museum, /organization/political, /organization/religious, /location/continent, /location/lake_sea_ocean, /location/region, /location/river, /gpe/city, /gpe/country, /gpe/state_province, /work_of_art/book, /work_of_art/play, /work_of_art/song, /product/vehicle, /product/weapon | **Known Types in Few-NERD:**<br>/person, /organization, /location, /building, /person/actor, /person/artist_author, /person/athlete, /person/director, /person/other, /person/politician, /person/scholar, /person/soldier, /organization/company, /organization/education, /organization/government, /organization/media_newspaper, /organization/other, /organization/political_party, /organization/religion, /organization/show_organization, /organization/sports_league, /organization/sports_team, /location/bodies_of_water, /location/gpe, /location/island, /location/mountain, /location/other, /location/park, /location/road_railway_highway_transit, /building/airport, /building/hospital, /building/hotel, /building/library, /building/other, /building/restaurant, /building/sports_facility, /building/theater |
| **Unknown Types in BBN:**<br>/contact_info, /event, /facility, /disease, /game, /language, /law, /plant, /substance, /contact_info/url, /event/hurricane, /event/war, /facility/airport, /facility/attraction, /facility/bridge, /facility/building, /facility/highway_street, /substance/chemical, /substance/drug, /substance/food | **Unknown Types in Few-NERD:**<br>/art, /event, /other, /product, /art/broadcast_program, /art/film, /art/music, /art/other, /art/painting, /art/written_art, /event/attack_battle_war_military_conflict, /event/disaster, /event/election, /event/other, /event/protest, /event/sports_event, /other/astronomy_thing, /other/award, /other/biology_thing, /other/chemical_thing, /other/currency, /other/disease, /other/educational_degree, /other/god, /other/language, /other/law, /other/living_thing, /other/medical, /product/airplane, /product/car, /product/food, /product/game, /product/other, /product/ship, /product/software, /product/train, /product/weapon |
  
## Quickly Reproduce

We provide all generated pseudo unknown-type instances in the folder `checkpoints` (You should extract the compressed files with a `.zip` extension). Based on these instances, you can run the following command to reproduce the results

```
python train_weight_llm.py --data <dataset_name> --unknown_flag True --contrast True --lamb 0.1
```



## Reproduce of Our Framework

### PLM-Based Generation Module

#### keyword extractor

Employ the attribute value representation module to generate semantics-enhanced type representations

```
sh class_embed/run_type_name.sh
```

Train the keyword extractor

```
python train_aug.py --data <dataset_name>
```

#### iterative substitution

Train the unified open-set classifier using  known-type instances to calculate the unknown-type probability


```
python train_weight.py --data <dataset_name>
```

Perform iterative substitutions across all known-type instances

```
python plm_generate.py --data <dataset_name>
```

### LLM-Based Generation Module

Train the unified open-set classifier using  known-type instances and pseudo unknown-type instances derived from the PLM-based generation module

```
python train_weight.py ---data <dataset_name> --unknown_flag True
```

Incorporate a prompt along with selected demonstrations to aid the LLM in generating pseudo unknown-type instances

```
python llm_generate.py --data <dataset_name> 
```

### Classifier Optimization

Train the unified open-set classifier using  known-type instances and pseudo unknown-type instances derived from the above two generation modules

```
python train_weight_llm.py --data <dataset_name> --unknown_flag True --contrast True --lamb 0.1
```
## Descriptions of Baseline Methods
### Fine-grained entity typing
For fine-grained entity typing methods, we incorporate a threshold mechanism that enables their identification of unknown-type instances.
1. **UFET** [1] utilizes BiLSTM and GloVe for context encoding, while employing GloVe and CharCNN for entity mention encoding. During inference, it learns a type label matrix to estimate each type's probability for multi-label classification based on a predefined threshold (i.e., 0.5). To enable UFET to detect unknown-type instances, we classify instances whose probabilities of all types fall below the threshold as unknown types.
2. **Box4Types** [2] employs box embeddings to jointly represent entity mentions and entity types, thus effectively capturing their intersections. Based on these intersections, conditional probabilities for each type are computed to perform the final type prediction based on a predefined threshold (i.e., 0.5). Similarly, we regard instances whose conditional probabilities for all types fall below the threshold as unknown types.
3. **UniST** [3] exploits type semantics to learn a joint semantic embedding space for both entity mentions and types. Depending on the similarities between mentions and types, types with similarities above a certain threshold are given as the final prediction, where the threshold is tuned on the validation set. Similar to the above, we regard instances with all similarities below this threshold as unknown types.
4. **CASENT** [4] is a seq2seq model designed for entity typing that predicts types with calibrated confidence scores. To enable it the ability to detect unknown-type instances, we classify instances as unknown types if all of their type scores fall below a predefined global threshold, which is estimated on the validation set.
5. **ChatGPT** [5] is a well-known generative large language model that has achieved success in many NLP applications. We regard it as a baseline and evaluate its performance of addressing the OSET task with a tailor-designed prompt as follows: "_You are an AI assistant who specializes entity types. Your task is as follows: according to the sentence, predict the entity type of entity mention in the sentence. If the predicted type belongs to the known types supported by the system, return the corresponding known type, otherwise return "unknown". The supported known types include: {known types}. Only provide one type from above known types or "unknown" and do not give the explanation_".
7. **Llama3** [6] is a large language model released by Meta, pre-trained on 15 trillion tokens. Similar to ChatGPT, we evaluate its performance for OSET  with the same prompt as **ChatGPT**.

### Open-set text classification 
As entity typing can be viewed as a text classification task to some extent, we also apply several open-set text classification methods to the OSET task for a comprehensive comparison. During inference of these methods, we convert the predicted type label into a type path via obtaining the corresponding types from the top level of the taxonomy to the level of the predicted type label.
1. **MSP** [7] is a softmax prediction probability baseline that classifies known-type instances based on the maximum softmax probabilities and detects those of unknown types with a predefined threshold (i.e., 0.5).
2. **SEG** [8] utilizes a Gaussian mixture distribution to learn representations of instances and injects dynamic type semantic information into Gaussian means to enhance the detection of unknown-type instances.
3. **SCL** [9] is a supervised contrastive learning method, which learns discriminative representations for classifying known-type instances and detecting those of unknown types. Besides, it also adopts an adversarial augmentation mechanism to derive pseudo various views of known-type instances in the latent space, thus enhancing the performance.
4. **APRL** [10] learns representations by maximizing variance between reciprocal points and known-type instances, and then leverages a learnable margin to constrain open space. However, directly applying this method from computer vision to our task significantly declines performance. Accordingly, we introduce a pre-training step using known-type instances and detect unknown-type instances with a probability threshold (i.e., 0.5). 
5. **SELFSUP** [11] trains a discriminative classifier by constructing synthetic outliers via self-supervision, enhancing the capability of classifying known-type instances and detecting unknown-type instances.

### References
[1] Eunsol Choi, Omer Levy, Yejin Choi, and Luke Zettlemoyer. 2018. Ultra-Fine Entity Typing. In ACL. 87–96  
[2] Yasumasa Onoe, Michael Boratko, Andrew Mccallum, and Greg Durrett. 2021. Modeling Fine-Grained Entity Types with Box Embeddings. In ACL-IJCNLP. 2051–2064.  
[3] James Y Huang, Bangzheng Li, Jiashu Xu, and Muhao Chen. 2022. Unified Semantic Typing with Meaningful Label Inference. In NAACL-HLT. 2642–2654.  
[4] Yanlin Feng, Adithya Pratapa, and David R Mortensen. 2023. Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing. In EMNLP-Findings. 15550–15560.  
[5] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In NIPS. 27730–27744.  
[6] AI@Meta. 2024. Llama 3 Model Card. (2024). https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md.  
[7] Dan Hendrycks and Kevin Gimpel. 2017. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In ICLR.  
[8] Guangfeng Yan, Lu Fan, Qimai Li, Han Liu, Xiaotong Zhang, Xiao-Ming Wu, and Albert YS Lam. 2020. Unknown intent detection using Gaussian mixture model with an application to zero-shot intent classification. In ACL. 1050–1060.  
[9] Zhiyuan Zeng, Keqing He, Yuanmeng Yan, Zijun Liu, Yanan Wu, Hong Xu, Huixing Jiang, and Weiran Xu. 2021. Modeling Discriminative Representations for Out-of-Domain Detection with Supervised Contrastive Learning. In ACL-IJCNLP. 870–878.  
[10] Guangyao Chen, Peixi Peng, Xiangqian Wang, and Yonghong Tian. 2021. Adversarial reciprocal points learning for open set recognition. IEEE TPAMI 44, 11 (2021), 8065–8081.  
[11] Li-Ming Zhan, Haowen Liang, Bo Liu, Lu Fan, Xiao-Ming Wu, and Albert YS Lam. 2021. Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training. In ACL-IJCNLP. 3521–3532.  
