AMTL
---
## Requirements
Ensure you have the following packages installed:

* Python (tested on 3.7.4)
* CUDA (tested on 11.6)
* [PyTorch](http://pytorch.org/) (tested on 1.12.1+cu113)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.20.1)
* numpy (tested on 1.21.6)
* [spacy](https://spacy.io/) (tested on 3.3.3)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm
* wandb

---
## Dataset
Organize your dataset files as follows:
```
AMTL
 |-- dataset
 |    |-- redocred
 |    |    |-- train_revised.json        
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json 
 |-- meta
 |   |-- redocred_rel2id_4_Na.json
 |   |-- rel2id.json
 |-- scripts
 |-- checkpoint
 |-- result
```

---
## Training and Evaluation

Use the following commands to train and evaluate the model.

#### On ReDocRED Dataset 
- **Using RoBERTa**:
    ```bash
    # Training
    bash scripts/train_roberta_seeds_ReDocRED_MT.sh
    
    # Evaluation
    bash scripts/test_roberta_seeds_ReDocRED_MT.sh
    ```
  
- **Using BERT**:
    ```bash
    # Training
    bash scripts/train_bert_seeds_ReDocRED_MT.sh
    
    # Evaluation
    bash scripts/test_bert_seeds_ReDocRED_MT.sh
    ```
---

Note: This code is partially based on the code of [ATLOP](https://github.com/wzhouad/ATLOP) and [HingeABL](https://github.com/Jize-W/HingeABL).
