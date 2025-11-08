#!/bin/bash
RENAME_FILE="/data/0.rename.txt"
GENE_LIST="/data/1.chr22_genes_list.txt"
INPUT_DIR="/path/to/1.run_sim1000g.r/output_dir"
OUT_DIR1="/path/to/output_dir_1"
OUT_DIR2="/path/to/output_dir_2"
WINDOW=98304
mkdir -p "$OUTDIR"
while IFS=, read -r ENSG CHR TSS SYMBOL STRAND; do

    # Skip empty lines and header
    [[ -z "$ENSG" ]] && continue
    [[ "$ENSG" == "FileName" ]] && continue

    # Define region (TSS ± WINDOW)
    start=$((TSS - WINDOW))
    if [ $start -lt 1 ]; then start=1; fi
    end=$((TSS + WINDOW))

    # Input / Output paths
    infile="${INPUT_DIR}/${ENSG}_simulated_chr22.vcf.gz"
    outfile="${OUT_DIR1}/${ENSG}.vcf.gz"

    # Check input file existence
    if [ ! -f "$infile" ]; then
        echo "⚠️ Input file not found for $ENSG ($infile)"
        continue
    fi

    # Processing message
    echo "▶ Processing $ENSG ($SYMBOL): ${CHR}:${start}-${end} → $outfile"

    # Extract region and index
    bcftools view -r ${CHR}:${start}-${end} -Oz -o "$outfile" "$infile"
    tabix -p vcf "$outfile"

done < "$GENE_LIST"

mkdir -p "$OUT_DIR2"

for file in ${OUT_DIR1}/*.vcf.gz; do
    base=$(basename "$file" .vcf.gz)
    out="${OUT_DIR2}/${base}.reheader.vcf.gz"

    echo "▶ Processing $base ..."
    bcftools annotate --rename-chrs "$RENAME_FILE" -Oz -o "$out" "$file"
    tabix -p vcf "$out"
done

echo "✅ All VCFs reheadered to $OUT_DIR2"
