# 🧩 Environment: VCF Cleaning & FASTA Generation

This environment is designed specifically to support all scripts involved in **VCF cleaning** and **FASTA sequence generation** for both real and synthetic individuals.
The Conda environment defined in **`vcf_cleaning.yml`** provides all necessary tools (e.g., `bcftools`, `htslib`, Python packages) to run the preprocessing pipeline reliably.

---

## 📁 Relevant Scripts Supported by This Environment

The following scripts rely on the `vcf_cleaning.yml` environment:

### **1. VCF Cleaning**
- `data_preprocessing/0.vcf_cleaning/1.vcf_cleaning.sh`  
  Performs initial cleanup, indexing, and filtering of VCF files.

### **2. Real Individual FASTA Generation**
- `data_preprocessing/1.preparing_real_sequences/1.vcf_to_fasta.py`  
  Converts cleaned real-individual VCF files into personalized haplotype FASTA sequences.

### **3. Synthetic Individual FASTA Generation**
- `data_preprocessing/2.preparing_virtual_sequences/7.vcf_to_fasta.py`  
  Converts sim1000G-generated synthetic VCFs into haplotype FASTA sequences.

---

## 🚀 Create the Conda Environment

Simply run:

```bash
conda env create -f vcf_cleaning.yml
conda activate vcf_cleaning
```