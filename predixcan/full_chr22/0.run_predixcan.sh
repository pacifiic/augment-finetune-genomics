#!/bin/bash
python3 /path/to/MetaXcan-master/software/Predict.py \
--model_db_path /path/to/ElasticNet/elastic_net_models/en_Cells_EBV-transformed_lymphocytes.db \
--model_db_snp_key varID \
--vcf_genotypes /path/to/data_preprocessing/1.vcf_cleaning.sh/out \
--vcf_mode imputed \
--on_the_fly_mapping METADATA "{}_{}_{}_{}_b38" \
--liftover /path/to/hg19ToHg38.over.chain.gz \
--prediction_output /path/to/output/predicted_expression.txt \
--prediction_summary_output /path/to/output/prediction_summary.txt