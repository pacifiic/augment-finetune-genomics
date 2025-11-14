# -*- coding: utf-8 -*-
"""Centralized configuration for paths and defaults (vector-only)."""
import os

# -----------------------------
# Paths & Constants (edit as needed)
# -----------------------------
SEQUENCE_LENGTH = 393216 // 2

# Real data (vector .pt)
PROCESSED_VECTOR_DIR = '/path/to/data_preprocessing/1.preparing_real_sequences/4.fasta_to_vector/output_dir'

TRAIN_GENES_FILE = '/path/to/data/7.gene_id_chr22.txt'
TRAIN_SAMPLES_FILE = 'path/to/data/2.train_samples_file_84.txt'
VALID_SAMPLES_FILE = '/path/to/data/3.valid_samples_file_42.txt'
TARGETS_CSV = '/path/to/data/6.geuvadis_peer_normalized_filtered.csv'

# Virtual data
VIRTUAL_VECTOR_DIR = '/path/to/data_preprocessing/2.preparing_virtual_sequences/9.fasta_to_vector/output_dir'
PSEUDO_LABEL_PATH = '/path/to/data/10.merged_virtual_sequence_pseudo_label.txt'

DEFAULT_SAVE_ROOT_DIR = '/path/to/default_save_root_dir'

# Beta params (genes)
VIRTUAL_BETA_PARAMS_PATH = '/path/to/data/12.virtual_beta_params.csv'
REAL_BETA_PARAMS_PATH = '/path/to/data/8.real_beta_params.csv'

EUROPEAN_INDIVIDUAL_COUNT = 295

# Training defaults
DEFAULT_LR = 1e-4
DEFAULT_WD = 1e-3
DEFAULT_ZERO_STAGE = 1
DEFAULT_WARMUP_STEPS = 1000