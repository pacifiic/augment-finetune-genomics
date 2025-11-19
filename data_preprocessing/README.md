## 🧬 Data Preparation

This repository provides all scripts necessary to reproduce both **real individual FASTA sequences** and **simulated (synthetic) FASTA sequences**, which are used as inputs to the Enformer fine-tuning pipeline.  
You may either **recreate the datasets from scratch** or **use the ready-made files** available in the `data/` directory and on Zenodo.

---

## 📌 1. Synthetic Sequences (Recommended: Download from Zenodo)

Fully processed synthetic haplotype FASTA files used in our experiments can be downloaded from:

📦 **Zenodo record:**  
https://zenodo.org/records/17637352

These sequences were originally generated using the following pipeline:

```
data_preprocessing/2.preparing_virtual_sequences/
  ├── 0.vcf_preprocessing.sh
  ├── 1.run_sim1000g.r
  ├── 2.adjust_vcf_length.sh
  ├── 7.vcf_to_fasta.py
  ├── 8.adjust_fasta_length.py
```

> ⚠ **Note:**  
> You do *not* need to run these scripts yourself unless you want to regenerate the synthetic genomes.  
> The identical final FASTA sequences are already available on Zenodo.

> 🧪 **Environment requirement:**  
> Scripts that convert VCFs to FASTA (`7.vcf_to_fasta.py`) require the  
> **`environment_setup/env_vcf_cleaning`** Conda environment.

### ➤ Vectorization required before fine-tuning
To use the synthetic sequences for Enformer fine-tuning, convert them to embedding vectors via:

```
data_preprocessing/2.preparing_virtual_sequences/9.fasta_to_vector.py
```

This produces PyTorch vector files (`torch.save`) ready for input to the training pipeline.

> 🧪 **Environment requirement:**  
> The vectorization script (`9.fasta_to_vector.py`) must be run inside  
> **`environment_setup/env_finetuning_enformer`**.

---

## 📌 2. Real Individual Sequences (Must Be Generated Locally)

Real haplotype sequences derived from GEUVADIS cannot be redistributed.  
Therefore, users must generate them locally using the provided preprocessing steps.

Run the following pipeline:

```
data_preprocessing/0.vcf_cleaning/
    1.vcf_cleaning.sh

data_preprocessing/1.preparing_real_sequences/
    0.download_ref_fasta.sh
    1.vcf_to_fasta.py
    2.adjust_fasta_length.py
```

These steps:

1. Clean the real-individual VCF files  
2. Download the reference genome (hg19)  
3. Substitute individual-specific variants to produce personalized haplotypes  
4. Normalize FASTA length to 196,608 bp

> 🧪 **Environment requirement:**  
> Both `1.vcf_cleaning.sh` and `1.vcf_to_fasta.py` require the  
> **`environment_setup/env_vcf_cleaning`** Conda environment.

### ➤ Vectorization required before fine-tuning
After preparing the finalized real haplotype FASTA sequences:

```
data_preprocessing/1.preparing_real_sequences/3.fasta_to_vector.py
```

This generates the real-individual embedding vectors used by `finetuning_enformer`.

> 🧪 **Environment requirement:**  
> Run `3.fasta_to_vector.py` using  
> **`environment_setup/env_finetuning_enformer`**.

---

## 📌 3. Files Already Provided in `data/` (No Execution Needed)

The following preprocessing steps were used in the original study to generate pseudo-expression labels and beta-parameter metadata.  
However, **all resulting files are already included under `data/`**, so these scripts do *not* need to be executed again:

```
data_preprocessing/1.preparing_real_sequences/4.make_beta_params.py

data_preprocessing/2.preparing_virtual_sequences/
    3.preprocessing_before_prediction.py
    4.PrediXcan.sh
    5.preprocessing_after_prediction.py
    6.merge_prediction_out.py
    10.make_beta_params.py
```

These scripts were part of the full synthetic-data augmentation pipeline involving PrediXcan pseudo-labeling.  
You may directly use the prepared datasets from the repository.

---

## ✔️ Summary

| Task | Real Data | Synthetic Data |
|------|-----------|----------------|
| FASTA generation | Must run scripts locally | **Download directly from Zenodo** (recommended) |
| Required environment | `env_vcf_cleaning` | `env_vcf_cleaning` (only if regenerating) |
| Vectorization required | `3.fasta_to_vector.py` (env: `env_finetuning_enformer`) | `9.fasta_to_vector.py` (env: `env_finetuning_enformer`) |
| Ready-to-use files in `data/` | Beta parameters, pseudo-labels | Beta parameters, PrediXcan outputs |

After running the required vectorization steps, the outputs can be directly consumed by the fine-tuning scripts inside:

```
finetuning_enformer/
```
