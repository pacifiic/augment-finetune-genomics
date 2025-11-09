GENE_LIST="/path/to/data/1.chr22_genes_list.txt"
INPUT_VCF="/path/to/data_preprocessing/1.vcf_cleaning.sh/out"
OUTDIR="/path/to/gene_vcf"
WINDOW=98304

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

#!/bin/bash

VCF_DIR="/path/to/gene_vcf"
OUTPUT_DIR="/path/to/pred_196608"

MODEL_DB="/path/to/ElasticNet/elastic_net_models/en_Cells_EBV-transformed_lymphocytes.db"
LIFTOVER="/path/to/hg19ToHg38.over.chain.gz"
PREDICT_SCRIPT="/path/to/MetaXcan-master/software/Predict.py"

for vcf_file in "$VCF_DIR"/*.vcf.gz; do
    gene_id=$(basename "$vcf_file" ".vcf.gz")
    prediction_output="${OUTPUT_DIR}/predicted_expression_${gene_id}.txt"
    prediction_summary="${OUTPUT_DIR}/prediction_summary_${gene_id}.txt"

    echo "Processing $gene_id ..."

    python3 "$PREDICT_SCRIPT" \
        --model_db_path "$MODEL_DB" \
        --model_db_snp_key varID \
        --vcf_genotypes "$vcf_file" \
        --vcf_mode genotyped \
        --on_the_fly_mapping METADATA "{}_{}_{}_{}_b38" \
        --liftover "$LIFTOVER" \
        --prediction_output "$prediction_output" \
        --prediction_summary_output "$prediction_summary"
done
