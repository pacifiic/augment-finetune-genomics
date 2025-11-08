"""
For perfect reproducibility, it is recommended to run using a single NVIDIA H100 GPU.
"""
# Copyright (c) 2018 Kipoi team
# Licensed under the MIT License.
# Source: https://github.com/kipoi/kipoiseq
import torch
import numpy as np
import os
import glob
from enformer_pytorch_for_lora.modeling_enformer import from_pretrained
from torch.cuda.amp import autocast

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 393216 // 2

# DNA alphabet definition
DNA = ['A', 'C', 'G', 'T']


def _get_alphabet_dict(alphabet):
    return {l: i for i, l in enumerate(alphabet)}


def tokenize(seq, alphabet=DNA, neutral_alphabet=["N", "."]):
    alphabet_dict = _get_alphabet_dict(alphabet)
    return np.array([alphabet_dict.get(base, -1 if base == 'N' else -2) for base in seq])


def token2one_hot(tokens, alphabet_size=4, neutral_value=.25, neutral_char_value=[0.25] * 4, dtype=None):
    arr = np.zeros((len(tokens), alphabet_size), dtype=dtype)
    tokens_range = np.arange(len(tokens), dtype=int)
    arr[tokens_range[tokens >= 0], tokens[tokens >= 0]] = 1
    arr[tokens_range[tokens == -2], :] = neutral_char_value
    return arr


def one_hot_dna(seq, alphabet=DNA, neutral_alphabet=['N', '.'], neutral_value=.25, dtype=np.float32):
    tokens = tokenize(seq, alphabet, neutral_alphabet)
    return token2one_hot(tokens, len(alphabet), neutral_value, dtype=dtype)


def one_hot_encode(sequence):
    return one_hot_dna(sequence).astype(np.float32)


def load_preprocessed_sequence(file_path):
    with open(file_path, 'r') as f:
        sequence = f.read().strip()
    return sequence


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load Enformer model
    enformer = from_pretrained('EleutherAI/enformer-official-rough')
    enformer.to(device)
    enformer.eval()

    # 2) Set input/output directories
    input_dir = "/path/to/2.adjust_fasta_length/output_dir"
    output_dir = "/path/to/output_dir"

    # 3) Traverse all folders
    pattern = os.path.join(input_dir, "*", "*.fa")
    fa_files = glob.glob(pattern)

    # 4) Process each fa file
    for fa_path in fa_files:
        sub_dir, filename = os.path.split(fa_path)
        individual_id = os.path.basename(sub_dir)
        gene_name_replicate, _ = os.path.splitext(filename)

        individual_out_dir = os.path.join(output_dir, individual_id)
        os.makedirs(individual_out_dir, exist_ok=True)

        out_filename = f"{gene_name_replicate}.pt"
        out_path = os.path.join(individual_out_dir, out_filename)

        if os.path.exists(out_path):
            print(f"Already exists, skipping: {out_path}")
            continue

        # 5) Load sequence + One-hot encoding
        sequence = load_preprocessed_sequence(fa_path)
        one_hot_np = one_hot_encode(sequence)
        one_hot = torch.from_numpy(one_hot_np)
        one_hot = one_hot.unsqueeze(0).to(device)

        # 6) Get tensor after Conv Tower
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            conv_output = enformer.get_conv_output(one_hot)

        # 7) Save (move to CPU)
        conv_output = conv_output.squeeze(0).cpu()
        torch.save(conv_output, out_path)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
