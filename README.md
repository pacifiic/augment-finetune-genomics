# Preference-Based Fine-Tuning of Genomic Sequence Models for Personal Expression Prediction with Data Augmentation

This repository contains the official implementation for the study:

> **Preference-Based Fine-Tuning of Genomic Sequence Models for Personal Expression Prediction with Data Augmentation**

It provides preprocessing pipelines, training scripts, and evaluation tools for fine-tuning Enformer-based architectures on both real and simulated genomic data.

---

## 📘 Overview

This project introduces a **hybrid fine-tuning framework** that integrates real RNA-seq–based regression with simulated genome–based preference learning.  
The method leverages **LoRA-based adapters** for parameter-efficient fine-tuning of Enformer models and uses population-level genetic diversity simulated via `sim1000G` to enhance cross-individual generalization.

---
## 📁 Project Structure
A high-level overview of the repository structure is shown below:
```bash
augment-finetune-genomics/
├── environment_setup/                 # Conda environments for preprocessing & fine-tuning
│   ├── env_vcf_cleaning/
│   └── env_finetuning_enformer/
│
├── data_preprocessing/                # All scripts to reproduce real & synthetic datasets
│   ├── 0.vcf_cleaning/
│   ├── 1.preparing_real_sequences/
│   └── 2.preparing_virtual_sequences/
│
├── finetuning_enformer/               # Enformer fine-tuning implementation (LoRA + DeepSpeed)
│
├── evaluate_testset.py                # The test-set evaluation script
│
└── data/                              # Preprocessed datasets required for training
```

This structure provides a clear separation of preprocessing, training, and evaluation components, allowing users to reproduce all results end-to-end.

---
## ⚙️ Environment Setup
To run the codes in finetuning_enformer, refer to the [`environment_setup`](./environment_setup) directory for installation and environment configuration instructions.

---
## 🧬 Data Preparation

All datasets used in this study—both real and synthetic—can be reproduced using the scripts in the
[`data_preprocessing`](./data_preprocessing) directory.
A detailed step-by-step guide is provided in **`data_preprocessing/README.md`**, and users should consult that document when regenerating the dataset.

For convenience, many intermediate files and final processed outputs are already included in the `data/` directory.
You may simply point the fields in `train_enformer/config.py` to the corresponding files in `data/` without rerunning the full preprocessing pipeline.

- **Synthetic sequences**: fully processed haplotype FASTA files can be downloaded from Zenodo  
  https://zenodo.org/records/17637352  
  After downloading, convert them into embedding vectors using  
  `data_preprocessing/2.preparing_virtual_sequences/9.fasta_to_vector.py`.

- **Real sequences**: must be generated locally (cannot be redistributed).  
  Use the scripts in  
  `data_preprocessing/0.vcf_cleaning/` and  
  `data_preprocessing/1.preparing_real_sequences/`  
  to create personalized FASTA sequences, then convert them into embeddings with  
  `data_preprocessing/1.preparing_real_sequences/3.fasta_to_vector.py`.


---
  
## 🚀 Running Experiments
First, download
https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/precomputed/tf_gammas.pt
and place it in:
```bash
finetuning_enformer/enformer_pytorch_for_lora/precomputed/tf_gammas.pt
```
The main fine-tuning experiments are located in the finetuning_enformer directory.
Before launching experiments, please update all dataset and checkpoint paths in:
```bash
train_enformer/config.py
```
This file contains clearly organized fields for all file paths required during training,
including real-data inputs, synthetic-data inputs. You must replace these with paths valid for your system.
All required input files are located in the data/ directory, making it easy to match
each field in config.py with the corresponding dataset.
In particular, the fine-tuning pipeline requires the outputs of:
```bash
data_preprocessing/1.preparing_real_sequences/3.fasta_to_vector.py
data_preprocessing/2.preparing_virtual_sequences/9.fasta_to_vector.py
```


To launch all experiments in the suite simultaneously (as run in our study on 4×H100 GPUs):
```bash
deepspeed --num_nodes 1 --num_gpus 4 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --module train_enformer.main -- --suite
```
Environment variables MASTER_ADDR and MASTER_PORT can be set manually if needed
(e.g., when multiple distributed jobs are running on the same machine).

This command automatically executes all experiment phases, including:

• Real-only regression

• Real–synthetic alternating fine-tuning

• Real-only fine-tuning combining regression and Bradley–Terry preference objectives


## 📊 Evaluation
After completing a training experiment, you can evaluate the model on the test set by running:
```bash
python evaluation_on_testset.py
```
This script loads the best-performing checkpoint (selected based on validation Pearson correlation), computes per-gene Pearson and Spearman correlations between predicted and observed expression values across individuals, and outputs both gene-level and summary metrics.


## 📄 Citation & Licenses

This repository integrates components from several open-source projects under the MIT License:
```bash
Copyright (c) 2018 Kipoi team
Licensed under the MIT License.
Source: https://github.com/kipoi/kipoiseq

Copyright (c) 2021 Phil Wang
Licensed under the MIT License.
Source: https://github.com/lucidrains/enformer-pytorch

Copyright (c) 2023 ni-lab
Licensed under the MIT License.
Source: https://github.com/ni-lab/personalized-expression-benchmark
```

All real individual genomic sequences used in this study were reconstructed locally from [E-GEUV-1](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-GEUV-1) and are not redistributed in this repository.
Synthetic genomes were generated using sim1000G, which simulates haplotypes based on population-level genetic variation and does not reproduce any individual's real genome.