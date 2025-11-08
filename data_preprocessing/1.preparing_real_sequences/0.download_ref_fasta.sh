#!/bin/bash
mkdir data/ref_fasta && cd data/ref_fasta
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz
faidx -x hg19.fa
rm *random* *hap* *Un* # cleanup unnecessary files
for file in *.fa; do faidx -x $file; done
rm *random* *hap* *Un* # cleanup unnecessary files