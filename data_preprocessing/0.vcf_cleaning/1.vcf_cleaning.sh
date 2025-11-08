#!/bin/bash
# Reference: https://www.biostars.org/p/385427/
# (This implementation was inspired by the discussion in the Biostars thread above.)

# VCFtools (0.1.17)
# bcftools 1.6
# vt version(2015.11.10) build(1) Channel(bioconda)
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
faidx -x hg19.fa
wget https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/001/E-GEUV-1/Files/E-GEUV-1/genotypes/GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz
zcat GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.vcf.gz \
  | awk '{
      if ($0 ~ /^##fileformat=VCFv4.1/) {
          print $0
          print "##FILTER=<ID=PASS,Description=\"All filters passed\">"
      }
      else if ($0 ~ /^##FORMAT=<ID=GL,/) {
          print "##FORMAT=<ID=GL,Number=G,Type=Float,Description=\"Genotype Likelihoods\">"
          print "##FORMAT=<ID=PP,Number=1,Type=String,Description=\"Placeholder for PP format\">"
          print "##FORMAT=<ID=BD,Number=1,Type=String,Description=\"Placeholder for BD format\">"
      }
      else if ($0 ~ /^##INFO=<ID=GERP,/) {
          print $0
          print "##contig=<ID=chr22>"
      }
      else {
          print $0
      }
  }' \
  | bgzip > GEUVADIS.chr22.reheader.vcf.gz

tabix -p vcf GEUVADIS.chr22.reheader.vcf.gz

bcftools annotate \
  --rename-chrs /data/0.rename.txt \
  -Oz \
  -o GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.chr22.vcf.gz \
  GEUVADIS.chr22.reheader.vcf.gz

tabix -p vcf GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.chr22.vcf.gz

bcftools norm -d both --threads=32 GEUVADIS.chr22.PH1PH2_465.IMPFRQFILT_BIALLELIC_PH.annotv2.genotypes.chr22.vcf.gz -Ov  -o chr22_step1.vcf.gz
bcftools view -e 'ALT="-" | REF="-"' chr22_step1.vcf.gz -Ov -o chr22_step2.vcf.gz
tabix -p vcf chr22_step2.vcf.gz
vt decompose -s chr22_step2.vcf.gz -o chr22_step3.vcf.gz
vt normalize chr22_step3.vcf.gz -r hg19.fa -o chr22_step4.vcf.gz

vcftools --gzvcf chr22_step4.vcf.gz --minGQ 90 --out chr22_step5 --recode

bcftools filter -i 'QUAL>30' chr22_step5.recode.vcf -o chr22_step6.vcf
bgzip chr22_step6.vcf
tabix -p vcf chr22_step6.vcf.gz