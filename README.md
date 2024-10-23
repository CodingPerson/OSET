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

## Quickly Reproduce

We provide all generated pseudo unknown-type instances in the folder `checkpoints`. Based on these instances, you can run the following command to reproduce the results

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

 

