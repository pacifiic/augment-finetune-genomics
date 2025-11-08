#!/bin/bash

# ==============================
# Configuration
# ==============================
TRAIN_SAMPLE_FILE="/data/2.train_sample_file_295.txt"
GENE_LIST="/data/1.chr22_genes_list.txt"
INPUT_VCF="/path/to/vcf_dir_from_[0.vcf_cleaning]_[1.vcf_cleaning.sh]/chr22_step6.vcf.gz"
OUTDIR="./output_dir"
WINDOW=120000

# ==============================
# Prepare output directory
# ==============================
mkdir -p "$OUTDIR"

# ==============================
# Step 1: Extract regions per gene
# ==============================
while IFS=, read -r ENSG CHR TSS SYMBOL STRAND; do
    [[ "$ENSG" == "FileName" || "$ENSG" == "" ]] && continue

    start=$((TSS - WINDOW))
    end=$((TSS + WINDOW))
    outvcf="${OUTDIR}/${ENSG}.vcf.gz"

    echo "▶ Processing $ENSG ($SYMBOL) → $outvcf"

    bcftools view -r chr${CHR}:${start}-${end} -Oz -o "$outvcf" "$INPUT_VCF"
    tabix -p vcf "$outvcf"
done < "$GENE_LIST"

# ==============================
# Step 2: Filter to training samples
# ==============================
cd "$OUTDIR" || exit

for f in *.vcf.gz; do
    out="${f%.vcf.gz}.training_samples.vcf.gz"
    echo "▶ Filtering $f → $out"
    bcftools view -S "$TRAIN_SAMPLE_FILE" -Oz -o "$out" "$f"
    tabix -p vcf "$out"
done
