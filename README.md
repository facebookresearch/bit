# BiT

This repository contains the training code of BiT introduced in our work: "[BiT: Robustly Binarized Multi-distilled Transformer](https://arxiv.org/abs/2205.13016)"

In this work, we identify a series of improvements which enables binary transformers at a much higher accuracy than what was possible previously. These include a two-set binarization scheme, a novel elastic binary activation function with learned parameters, and a multi-step distilation method. These approaches allow for the first time, fully binarized transformer models that are at a practical level of accuracy, approaching a full-precision BERT baseline on the GLUE language understanding benchmark within as little as 5.9%.

<div align=center>
<img width=60% src="https://github.com/facebookresearch/bit/blob/main/overview.jpg"/>
</div>


## Citation

If you find our code useful for your research, please consider citing:
    
    @article{liu2022bit,
    title={BiT: Robustly Binarized Multi-distilled Transformer},
    author={Liu, Zechun and Oguz, Barlas and Pappu, Aasish and Xiao, Lin and Yih, Scott and Li, Meng and Krishnamoorthi, Raghuraman and Mehdad, Yashar},
    journal={arXiv preprint arXiv:2205.13016},
    year={2022}
    }
    
## Run

### 1. Requirements:
* python 3.6, pytorch 1.7.1
    
### 2. Data:
* Download [GLUE dataset](https://github.com/nyu-mll/GLUE-baselines) and [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)

  For data augmentation on GLUE, please follow the instruction in [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).


### 3. Pretrained models:
* Download pretrained models from hugging face model zoo.
  | Dataset | Finetuned full-precision model |
  | --- | --- |
  | MNLI | [bert-base-uncased-MNLI](https://huggingface.co/textattack/bert-base-uncased-MNLI) |
  | QQP | [bert-base-uncased-QQP](https://huggingface.co/textattack/bert-base-uncased-QQP) |
  | QNLI | [bert-base-uncased-QNLI](https://huggingface.co/textattack/bert-base-uncased-QNLI) |
  | SST-2 | [bert-base-uncased-SST-2](https://huggingface.co/textattack/bert-base-uncased-SST-2) |
  | CoLA | [bert-base-uncased-CoLA](https://huggingface.co/textattack/bert-base-uncased-CoLA) | 
  | STS-B | [bert-base-uncased-STS-B](https://huggingface.co/textattack/bert-base-uncased-STS-B) |
  | MRPC | [bert-base-uncased-MRPC](https://huggingface.co/textattack/bert-base-uncased-MRPC) | 
  | RTE | [bert-base-uncased-RTE](https://huggingface.co/textattack/bert-base-uncased-RTE) |
  | Squad v1 | [bert-base-uncased-squad-v1](https://huggingface.co/csarron/bert-base-uncased-squad-v1) |

### 4. Steps to run:
* Specify the num_bits, data path and the pre-trained model path in scrips/run.sh file. 
* Run `bash scripts/run_glue.sh GLUE_dataset` or Run `bash scrips/run_squad.sh` .

  E.g., `bash scripts/run_glue.sh MNLI` for running the MNLI dataset in GLUE dataset.

## Models

### 1. GLUE dataset

(1) Without data augmentation

| Method | #Bits | Size (M)| FLOPs (G) | MNLI m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT |  32-32-32 |  418 | 22.5 | 84.9/85.5 | 91.4 | 92.1 | 93.2 | 59.7 | 90.1 | 86.3 | 72.2 | 83.9 |
| BinaryBert | 1-1-4 | 16.5 | 1.5 | 83.9/84.2 | 91.2 | 90.9 | 92.3 | 44.4 | 87.2 | 83.3 | 65.3 | 79.9 |
| BinaryBert | 1-1-2 | 16.5 | 0.8 | 62.7/63.9 | 79.9 | 52.6 | 82.5 | 14.6 | 6.5 | 68.3 | 52.7 | 53.7 |
| BinaryBert | 1-1-1 | 16.5 | 0.4 | 35.6/35.3 | 66.2 | 51.5 | 53.2 | 0 | 6.1 | 68.3 | 52.7 | 41.0 |
| BiBert | 1-1-1 | 13.4 | 0.4 | 66.1/67.5 | 84.8 | 72.6 | 88.7 | 25.4 | 33.6 | 72.5 | 57.4 | 63.2 |
| **BiT** \* | 1-1-4 | 13.4 | 1.5 | 83.6/84.4 | 87.8 | 91.3 | 91.5 | 42.0 | 86.3 | 86.8 | 66.4 |79.5 |
| **BiT** \*| 1-1-2 | 13.4 | 0.8 | 82.1/82.5 | 87.1 | 89.3 | 90.8 | 32.1 | 82.2 | 78.4 | 58.1 | 75.0 |
| **BiT** \*| 1-1-1 | 13.4 | 0.4 | 77.1/77.5 | 82.9 | 85.7 | 87.7 | 25.1 | 71.1 | 79.7 | 58.8 | 71.0 |
| **BiT** | **1-1-1** | **13.4** | **0.4** | [**79.5/79.4**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ETcn4hMvP0JAqUTYJv8HYoMBwgCpsw1z1sJC8swDgre2bA?e=Y9pPHc) | [**85.4**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EUS3ZzGid-xPoL5XBnwz-FABgzdRy2n6ml3AM_SfyXK8IQ?e=aZbYPf) | [**86.4**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EbeT2rWvmQFHutjtrHHQcMkBi1k_84adFi7MqflY0SosDQ?e=SPI5xA) | [**89.9**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ES1khipUjbFJkVKJscO2OncBvuWm6xDZzDeSMyWk5STLFQ?e=yoDfXv) | [**32.9**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EalT9Gpo-RRDvQwtOq_gpFIB3E9NbkkmW6zD0s7xyl2FdA?e=ZW7png) | [**72**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ER2owFhqoedErs304GY01fQBpjESj-23EYPWYw2-BhXBLA?e=mYWw3u) | [**79.9**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EVKtA_9BftBKpdOUP7-fim8BNVugn_kJGIoC4Wrok8pOEw?e=bvzERS) | [**62.1**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EbVxl1v5K2NNszc-tf5BtIcBUhbyb5zH8iUWfQy9jOhYMQ?e=IeVO8t)| **73.5**|


(2) With data augmentation

| Method | #Bits | Size (M)| FLOPs (G) | MNLI m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BinaryBert | 1-1-2 | 16.5 | 0.8 | 62.7/63.9* | 79.9* | 51.0 | 89.6 | 33.0 | 11.4 | 71.0 | 55.9 | 57.6 |
| BinaryBert | 1-1-1 | 16.5 | 0.4 | 35.6/35.3\* | 66.2\* | 66.1 | 78.3 | 7.3 | 22.1 | 69.3 | 57.7 | 48.7 |
| BiBert | 1-1-1 | 13.4 | 0.4 | 66.1/67.5\* | 84.8\* | 76.0 | 90.9 | 37.8 | 56.7 | 78.8 | 61.0 | 68.8 |
| **BiT** \*| 1-1-2 | 13.4 | 0.8 | 82.1/82.5\* | 87.1\* | 88.8 | 92.5 | 43.2 | 86.3 | 90.4 | 72.9 | 80.4 | 
| **BiT** \*| 1-1-1 | 13.4 | 0.4 | 77.1/77.5\* | 82.9\* | 85.0 | 91.5 | 32.0 | 84.1 | 88.0 | 67.5 | 76.0 |
| **BiT** | **1-1-1** | **13.4** | **0.4** |[**79.5/79.4***](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ETcn4hMvP0JAqUTYJv8HYoMBwgCpsw1z1sJC8swDgre2bA?e=Y9pPHc) | [**85.4***](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EUS3ZzGid-xPoL5XBnwz-FABgzdRy2n6ml3AM_SfyXK8IQ?e=aZbYPf) | [**86.5**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EZNPctCyEuRDkU5WMQEMbcIBB6pJWIQMoumx9-qkNJvQcw?e=A0BJzb) | [**92.3**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/Ee9SYLwlu_lCs0QmjdENy68BI6F72TOAlz-0b2sKVxO_Gw?e=88MXkV) | [**38.2**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EWC82bqkeAhEnEfYnK25744Bj-JNxJ3104F9-fAg-_5zbQ?e=Ee09vW) | [**84.2**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/Ea2QjooiWAVHs6TVCSebhmMBezUQ7xVorIuIiSlfGdevHA?e=CCLTeT) | [**88**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EbsmVMBB9VlNlV9Lt-esAm0Ba7ojT4zPY8-39jRPusouGg?e=WwvaNQ) | [**69.7**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EdtcPWWAKeRFml64PgNkoBUBQikuT8O_3l2cXuapRcvIcw?e=islH5G) | **78.0** |


### 2. SQuAD dataset

| Method | #Bits | SQuADv1.1 em/f1 |
| --- | --- | --- |
| BERT |  32-32-32 |  82.6/89.7 |
| BinaryBert | 1-1-4 | 77.9/85.8 |
| BinaryBert | 1-1-2 | 72.3/81.8 |
| BinaryBert | 1-1-1 | 1.5/8.2 |
| BiBert | 1-1-1 | 8.5/18.9 |
| **BiT** | **1-1-1** | [**63.1/74.9**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ES7JcZVGjShPmBn7GLxr0SABK2iDb7PqNj3QQrWiWp20PQ?e=54d6Kp) |

## Acknowledgement

The original code is borrowed from [BinaryBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BinaryBERT).

## Contact

Zechun Liu, Reality Labs, Meta Inc (liuzechun0216 at gmail.com)

## License
BiT is CC-BY-NC 4.0 licensed as of now.

