# Copyright (c) 2023 ni-lab
# Licensed under the MIT License.
# Source: https://github.com/ni-lab/personalized-expression-benchmark

# This implementation was developed with reference to
# https://github.com/ni-lab/personalized-expression-benchmark/blob/main/consensus/make_consensus_enformer.py
"""
python3 7.vcf_to_fasta.py ref_fasta_dir  \
        /data/1.chr22_genes_list.txt     \
        /data/11.samples.txt \
        -o output_path
        
If you split `/data/11.samples.txt` and use the parts as input, the computational load can be distributed across the compute nodes.
"""
from optparse import OptionParser
import subprocess
import os
import multiprocessing as mp
import pyfaidx
from itertools import product

REF_DIR = "ref"
INDS = "inds"
vcf_dir = "/path/to/2.adjust_vcf_length.sh/OUR_DIR2"
"""
IMPORTANT:
Unlike the previous code, you must use OUT_DIR2 from 2.adjust_vcf_length.sh as the input 
— not OUT_DIR1.

Since the SEQUENCE_LENGTH is sufficiently large, 
the generated FASTA file should be created without Ns.
If many Ns are included in the virtual sequence, 
the performance will deteriorate.
"""
SEQUENCE_LENGTH = 196608
INTERVAL = 114688


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
    start, end = int(tss) - SEQUENCE_LENGTH // 2, int(tss) + SEQUENCE_LENGTH // 2 - 1
    ref_command = f"samtools faidx {ref_fasta_dir}/chr{chr}.fa {chr}:{start}-{end} > {out_file}"
    subprocess.run(ref_command, shell=True)


def generate_consensus(pair):
    gene, sample = pair
    gene_id, chr, tss, _, strand = gene.split(",")
    out1, out2 = get_sample_files(sample, gene_id)
    ind1, ind2 = get_index_files(sample, gene_id)

    vcf_path = f"{vcf_dir}/{gene_id}.vcf.gz"  # gene-specific VCF 경로
    print(f"#### Starting consensus fasta for {gene_id}, Sample {sample} ####")

    # Haplotype 1
    if not os.path.exists(out1) or os.path.getsize(out1) == 0:
        hap1 = f"bcftools consensus -s {sample} -f {options.out_dir}/{REF_DIR}/{gene_id}.fa -I -H 1pIu {vcf_path} > {out1}"
        subprocess.run(hap1, shell=True)
    else:
        if not os.path.exists(f"{out1}.fai"):
            pyfaidx.Faidx(out1, rebuild=False)

    # Haplotype 2
    if not os.path.exists(out2) or os.path.getsize(out2) == 0:
        hap2 = f"bcftools consensus -s {sample} -f {options.out_dir}/{REF_DIR}/{gene_id}.fa -I -H 2pIu {vcf_path} > {out2}"
        subprocess.run(hap2, shell=True)
    else:
        if not os.path.exists(f"{out2}.fai"):
            pyfaidx.Faidx(out2, rebuild=False)


def make_dirs(samples):
    for sample in samples:
        if not os.path.exists(f"{options.out_dir}/{INDS}/{sample}"):
            os.makedirs(f"{options.out_dir}/{INDS}/{sample}")


if __name__ == "__main__":
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
    samples = get_items(sample_file)

    for gene in genes:
        generate_ref(ref_fasta_dir, gene)

    make_dirs(samples)

    pool = mp.Pool(processes=mp.cpu_count())
    with pool:
        pairs = product(genes, samples)
        pool.map(generate_consensus, pairs)

