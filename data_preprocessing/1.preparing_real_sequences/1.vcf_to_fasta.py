# Copyright (c) 2023 ni-lab
# Licensed under the MIT License.
# Source: https://github.com/ni-lab/personalized-expression-benchmark

# This implementation was developed with reference to
# https://github.com/ni-lab/personalized-expression-benchmark/blob/main/consensus/make_consensus_enformer.py
"""
python3 1.vcf_to_fasta.py ref_fasta_dir  \
        /data/1.chr22_genes_list.txt     \
        /data/5.individual_id_sorted.txt \
        -o output_path

If you split `/data/5.individual_id_sorted.txt` and use the parts as input, the computational load can be distributed across the compute nodes.
"""
from optparse import OptionParser
import multiprocessing as mp
import pyfaidx
from itertools import product

REF_DIR = "ref"
INDS = "inds"
vcf_dir = "/path/to/0.vcf_cleaning/1.vcf_cleaning.sh/vcf_dir"
SEQUENCE_LENGTH = 220000

def get_vcf(chr):
    return f"{vcf_dir}/chr{chr}_step6.vcf.gz"


def get_items(file):
    with open(file, "r") as f:
        return f.read().splitlines()


def get_sample_files(sample, gene_id):
    return f"{options.out_dir}/{INDS}/{sample}/{gene_id}.1pIu.fa", f"{options.out_dir}/{INDS}/{sample}/{gene_id}.2pIu.fa"

def get_index_files(sample, gene_id):
    return f"{options.out_dir}/{INDS}/{sample}/{gene_id}.1pIu.fai", f"{options.out_dir}/{INDS}/{sample}/{gene_id}.2pIu.fai"

def generate_ref(ref_fasta_dir, gene):
    gene_id, chr, tss, _, strand = gene.split(",")
    print(f"#### Starting reference fasta for {gene_id} ####")
    out_file = f"{options.out_dir}/{REF_DIR}/{gene_id}.fa"
    with open(out_file, "w") as f:
        start, end = int(tss) - SEQUENCE_LENGTH // 2, int(tss) + SEQUENCE_LENGTH // 2 - 1
        ref_command = f"samtools faidx {ref_fasta_dir}/chr{chr}.fa chr{chr}:{start}-{end} > {out_file}"
        subprocess.run(ref_command, shell=True)

import os
import subprocess
import pyfaidx

import os
import subprocess
import pyfaidx

def generate_consensus(pair):
    gene, sample = pair
    gene_id, chr, tss, _, strand = gene.split(",")
    out1, out2 = get_sample_files(sample, gene_id)
    ind1, ind2 = get_index_files(sample, gene_id)

    print(f"#### Starting consensus fasta for {gene_id}, Sample {sample} ####")

    # Check if the files already exist
    if not os.path.exists(out1) or os.path.getsize(out1) == 0:
        hap1 = f"bcftools consensus -s {sample} -f {options.out_dir}/{REF_DIR}/{gene_id}.fa -I -H 1pIu {get_vcf(chr)} > {out1}"
        subprocess.run(hap1, shell=True)
        pyfaidx.Faidx(out1, rebuild=False)
    else:
        # Check if .fai index file exists
        if not os.path.exists(f"{out1}.fai"):
            pyfaidx.Faidx(out1, rebuild=False)
        else:
            print(f"{out1} and its index already exist. Skipping...")

    if not os.path.exists(out2) or os.path.getsize(out2) == 0:
        hap2 = f"bcftools consensus -s {sample} -f {options.out_dir}/{REF_DIR}/{gene_id}.fa -I -H 2pIu {get_vcf(chr)} > {out2}"
        subprocess.run(hap2, shell=True)
        pyfaidx.Faidx(out2, rebuild=False)
    else:
        # Check if .fai index file exists
        if not os.path.exists(f"{out2}.fai"):
            pyfaidx.Faidx(out2, rebuild=False)
        else:
            print(f"{out2} and its index already exist. Skipping...")

def make_dirs(samples):
    for sample in samples:
        if not os.path.exists(f"{options.out_dir}/{INDS}/{sample}"):
            os.makedirs(f"{options.out_dir}/{INDS}/{sample}")

if __name__ == "__main__":
    """
    Create individual fasta sequences

    Arguments:
    - ref_fasta_dir: reference fasta directory
    
            mkdir data/ref_fasta && cd data/ref_fasta
            wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
            gunzip hg19.fa.gz 
            faidx -x hg19.fa 
            rm *random* *hap* *Un* # cleanup unnecessary files
            for file in *.fa; do faidx -x $file; done
            rm *random* *hap* *Un* # cleanup unnecessary files
    
    - genes_csv: file containing Ensembl gene IDs, chromosome, TSS position, gene symbol, and strand
    
            path: /data/1.chr22_genes_list.txt
            
    - sample_file: file containing individuals names
            
            path: /data/5.individual_id_sorted.txt
    """
    usage = "usage: %prog [options] <ref_fasta_dir> <genes_csv> <sample_file>"
    parser = OptionParser(usage)
    parser.add_option("-o", dest="out_dir",
                      default='consensus/seqs',
                      type=str,
                      help="Output directory for predictions [Default: %default]")
    (options, args) = parser.parse_args()

    num_expected_args = 3
    if len(args) != num_expected_args:
        parser.error(
            "Incorrect number of arguments, expected {} arguments but got {}".format(num_expected_args, len(args)))

    # Setup
    ref_fasta_dir = args[0]
    genes_file = args[1]
    sample_file = args[2]

    if not os.path.exists(options.out_dir):
        os.makedirs(options.out_dir)
    if not os.path.exists(options.out_dir + "/" + REF_DIR):
        os.makedirs(options.out_dir + "/" + REF_DIR)
    if not os.path.exists(options.out_dir + "/" + INDS):
        os.makedirs(options.out_dir + "/" + INDS)

    genes = get_items(genes_file)
    for gene in genes:
        generate_ref(ref_fasta_dir, gene)
    samples = get_items(sample_file)
    # make sample directories
    make_dirs(samples)
    pool = mp.Pool(processes=mp.cpu_count())
    with pool:
        pairs = product(genes, samples)
        pool.map(generate_consensus, pairs)
