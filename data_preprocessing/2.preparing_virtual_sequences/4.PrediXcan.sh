 cd "/path/to/3.preprocessing_before_prediction/output_dir"
 for f in *.vcf; do
     echo "▶ Compressing and indexing $f"
     bgzip -c "$f" > "${f}.gz"
     tabix -p vcf "${f}.gz"
 done

VCF_DIR="/path/to/3.preprocessing_before_prediction/output_dir"
OUTPUT_DIR="/path/to/output"
# wget https://zenodo.org/record/3519321/files/elastic_net_eqtl.tar?download=1
# wget https://hgdownload.soe.ucsc.edu/gbdb/hg19/liftOver/hg19ToHg38.over.chain.gz
# git clone https://github.com/hakyimlab/MetaXcan.git
MODEL_DB="/path/to/ElasticNet/elastic_net_models/en_Cells_EBV-transformed_lymphocytes.db"
LIFTOVER="/path/to/hg19ToHg38.over.chain.gz"
PREDICT_SCRIPT="/path/to/MetaXcan-master/software/Predict.py"


for vcf_file in "$VCF_DIR"/*.vcf.gz; do
    gene_id=$(basename "$vcf_file" ".vcf.gz")

    # 출력 파일 경로 설정
    prediction_output="${OUTPUT_DIR}/predicted_expression_${gene_id}.txt"
    prediction_summary="${OUTPUT_DIR}/prediction_summary_${gene_id}.txt"

    echo "Processing $gene_id ..."

    python3 "$PREDICT_SCRIPT" \
        --model_db_path "$MODEL_DB" \
        --model_db_snp_key varID \
        --vcf_genotypes "$vcf_file" \
        --vcf_mode genotyped \
        --on_the_fly_mapping METADATA "chr{}_{}_{}_{}_b38" \
        --liftover "$LIFTOVER" \
        --prediction_output "$prediction_output" \
        --prediction_summary_output "$prediction_summary"
done