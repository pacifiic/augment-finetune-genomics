# 🧩 Environment Overview

This directory contains multiple Conda environments used across the full data-processing and Enformer fine-tuning pipeline.  
Each subdirectory includes its own detailed README explaining installation and usage.

---

## 📁 Available Environments

### **1. `env_vcf_cleaning/`**
Environment for:

- Cleaning and preprocessing VCF files  
- Converting cleaned VCFs into haplotype-specific FASTA sequences  
  (for both **real** and **synthetic** individuals)

This environment supports scripts such as:

- `data_preprocessing/0.vcf_cleaning/1.vcf_cleaning.sh`
- `data_preprocessing/1.preparing_real_sequences/1.vcf_to_fasta.py`
- `data_preprocessing/2.preparing_virtual_sequences/7.vcf_to_fasta.py`

See the directory-specific README for details.

---

### **2. `env_finetuning_enformer/`**
Environment for:

- Running all scripts in `finetuning_enformer/`
- Running vectorization utilities:  
  - `data_preprocessing/1.preparing_real_sequences/3.fasta_to_vector.py`  
  - `data_preprocessing/2.preparing_virtual_sequences/9.fasta_to_vector.py`  

Includes PyTorch Enformer, LoRA (PEFT), DeepSpeed, Transformers, etc.  
Detailed installation steps are provided in the subdirectory README.

---

## 🧬 sim1000G Environment (R)

Synthetic individuals were generated using **sim1000G**, executed separately because the R environment required only minimal dependencies.

- **sim1000G version:**  
  Downloaded from CRAN Archive:  
  https://cran.r-project.org/src/contrib/Archive/sim1000G/sim1000G_1.40.tar.gz  

- **R session used:**
  ```
  R version 4.5.1 (2025-06-13) -- "Great Square Root"
  Platform: x86_64-conda-linux-gnu
  ```

- **Script:**  
  `data_preprocessing/2.preparing_virtual_sequences/1.run_sim1000g.r`

Since this step relies on base R + sim1000G only, we do not maintain a dedicated Conda environment here; simply installing sim1000G in R 4.5.1 was sufficient.

---

## 🧪 PrediXcan & AlphaGenome

PrediXcan and AlphaGenome are used as external baseline models.  
Their installation and usage are best handled via the official repositories:

- **PrediXcan:** https://github.com/hakyimlab/MetaXcan  
- **AlphaGenome:** https://github.com/google-deepmind/alphagenome  

Each provides its own environment setup instructions, so they are not bundled here.

---

## ✔️ Summary

- This directory only provides **two primary Conda environments**:
  - `env_vcf_cleaning`: VCF → FASTA preprocessing
  - `env_finetuning_enformer`: vectorization + Enformer fine-tuning

- The synthetic population generation (sim1000G) was run in a lightweight R environment, not a Conda-managed one.

- External tools such as PrediXcan and AlphaGenome rely on their official installation guides.
