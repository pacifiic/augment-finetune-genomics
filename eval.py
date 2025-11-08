#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn

EXP_DIRS = [
    "/path/to/exp01_real_regression100_seed42",
    "/path/to/exp02_real_virtual_mix_seed42/phase5_real30",
    "/path/to/exp03_real_only_mix_seed42/phase5_real30",
    "/path/to/exp01_real_regression100_seed427",
    "/path/to/exp02_real_virtual_mix_seed47/phase5_real30",
    "/path/to/exp03_real_only_mix_seed47/phase5_real30",
    "/path/to/exp01_real_regression100_seed52",
    "/path/to/exp02_real_virtual_mix_seed52/phase5_real30",
    "/path/to/exp03_real_only_mix_seed52/phase5_real30",
]

processed_vector_dir = '/path/to/data_preprocessing/preparing_real_sequences/4.fasta_to_vector/output_dir'
train_genes_file     = '/path/to/data/7.gene_id_chr22.txt'
valid_samples_file   = '/path/to/data/3.valid_samples_file_42.txt'
test_samples_file    = '/path/to/data/4.test_samples_file_84.txt'
targets_csv_path     = '/path/to/data/6.geuvadis_peer_normalized_filtered.csv'

# Common output directory
OUTPUT_DIR = '/path/to/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

from enformer_pytorch_for_lora.modeling_enformer import from_pretrained
from enformer_pytorch_for_lora.finetune import HeadAdapterWrapper_For_Lymphocyte_Vector_Input
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# Model freezing utilities
# =========================
def _freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def freeze_stem_and_conv_tower(enf: nn.Module):
    for name, module in enf.named_modules():
        if name.split(".")[-1] in ("stem", "conv_tower"):
            _freeze_module(module)

class EnformerPEFTWrapper(nn.Module):
    def __init__(self, enformer_model):
        super().__init__()
        self.enformer = enformer_model
    def forward(self, input_ids=None, seq=None, **kwargs):
        return self.enformer(seq)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.enformer, name)

def build_dual_enformer_for_vector(use_lora=True, lora_rank=32):
    e1 = from_pretrained("EleutherAI/enformer-official-rough")
    e2 = from_pretrained("EleutherAI/enformer-official-rough")
    freeze_stem_and_conv_tower(e1)
    freeze_stem_and_conv_tower(e2)
    if use_lora:
        cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.1,
            bias="none", target_modules=["to_q","to_k","to_v","to_out"],
            task_type=TaskType.FEATURE_EXTRACTION
        )
        w1, w2 = EnformerPEFTWrapper(e1), EnformerPEFTWrapper(e2)
        p1, p2 = get_peft_model(w1, cfg), get_peft_model(w2, cfg)
        return HeadAdapterWrapper_For_Lymphocyte_Vector_Input(enformer1=p1, enformer2=p2)
    return HeadAdapterWrapper_For_Lymphocyte_Vector_Input(enformer1=e1, enformer2=e2)

# =========================
# Dataset
# =========================
def safe_load_tensor(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)

class ConvOutputDataset(torch.utils.data.Dataset):
    """Dataset that loads precomputed convolutional outputs and corresponding expression targets."""
    def __init__(self, genes_file, samples_file, vector_dir, targets_df):
        with open(samples_file) as f:
            self.samples = [l.strip() for l in f if l.strip()]
        with open(genes_file) as f:
            self.genes = [l.strip() for l in f if l.strip()]
        self.vector_dir = vector_dir
        self.targets_df = targets_df

    def __len__(self):
        return len(self.genes) * len(self.samples) * 2

    def __getitem__(self, idx):
        si = (idx // 2) // len(self.genes)
        gi = (idx // 2) % len(self.genes)
        gene = self.genes[gi]
        sample = self.samples[si]
        if idx % 2 == 0:
            f1 = f"{self.vector_dir}/{sample}/{gene}.1pIu.pt"
            f2 = f"{self.vector_dir}/{sample}/{gene}.2pIu.pt"
        else:
            f1 = f"{self.vector_dir}/{sample}/{gene}.2pIu.pt"
            f2 = f"{self.vector_dir}/{sample}/{gene}.1pIu.pt"
        s1, s2 = safe_load_tensor(f1), safe_load_tensor(f2)
        t = torch.tensor(self.targets_df.loc[sample, gene].astype(np.float32))
        return s1, s2, t

def create_loader(genes_file, samples_file, vector_dir, targets_df):
    return DataLoader(
        ConvOutputDataset(genes_file, samples_file, vector_dir, targets_df),
        batch_size=1, shuffle=False, num_workers=4, prefetch_factor=2,
        pin_memory=torch.cuda.is_available()
    )

# =========================
# Checkpoint utilities
# =========================
def _extract_state_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("module", "state_dict", "model", "network", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, (torch.Tensor, nn.Parameter)) for v in obj.values()):
            return obj
    return obj

def _strip_prefix(state_dict, prefix):
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }

def find_best_checkpoint(base_dir):
    """Find the checkpoint with the best recorded Pearson correlation."""
    if not os.path.isdir(base_dir):
        return None, -1.0
    best_val = -1.0
    best_ckpt = None
    for ckpt in os.listdir(base_dir):
        if not ckpt.startswith("checkpoint_"):
            continue
        ckpt_dir = os.path.join(base_dir, ckpt)
        if not os.path.isdir(ckpt_dir):
            continue
        for f in os.listdir(ckpt_dir):
            if f.startswith("metadata_") and f.endswith(".json"):
                try:
                    with open(os.path.join(ckpt_dir, f)) as jf:
                        meta = json.load(jf)
                    pearson = meta.get("Pearson_correlation", -1.0)
                except Exception:
                    pearson = -1.0
                if pearson > best_val:
                    best_val = pearson
                    gs = [d for d in os.listdir(ckpt_dir) if d.startswith("global_step")]
                    if gs:
                        gs.sort()
                        best_ckpt = os.path.join(ckpt_dir, gs[-1], "mp_rank_00_model_states.pt")
    return best_ckpt, best_val

# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_per_gene(model, loader, device, genes, samples):
    model.eval()
    preds, targs = [], []
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    for s1, s2, t in loader:
        s1, s2 = s1.to(device, dtype), s2.to(device, dtype)
        t = t.to(device).float()
        out = model(s1, s2)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        preds.extend(out.detach().cpu().view(-1).tolist())
        targs.extend(t.cpu().view(-1).tolist())

    preds = np.array(preds).reshape(-1, 2).mean(1)
    targs = np.array(targs).reshape(-1, 2).mean(1)

    S, G = len(samples), len(genes)
    preds_mat = preds.reshape(S, G)
    targs_mat = targs.reshape(S, G)

    rows = []
    rhos_p, rhos_s = [], []
    for j, g in enumerate(genes):
        p, t = preds_mat[:, j], targs_mat[:, j]
        if np.isfinite(p).sum() >= 2 and np.isfinite(t).sum() >= 2:
            pear, spear = pearsonr(p, t)[0], spearmanr(p, t)[0]
        else:
            pear, spear = np.nan, np.nan
        rows.append({"gene": g, "pearson": pear, "spearman": spear, "n_samples": S})
        if not np.isnan(pear):
            rhos_p.append(pear)
        if not np.isnan(spear):
            rhos_s.append(spear)

    rows.append({
        "gene": "MEAN",
        "pearson": np.mean(rhos_p) if rhos_p else np.nan,
        "spearman": np.mean(rhos_s) if rhos_s else np.nan,
        "n_samples": S
    })
    return pd.DataFrame(rows)

# =========================
# Run evaluation for one experiment
# =========================
def run_one_experiment(EXP_DIR: str, device, targets_df_all, genes):
    exp_name = os.path.basename(EXP_DIR.rstrip("/"))
    best_ckpt, best_val = find_best_checkpoint(EXP_DIR)
    print(f"\n[BEST] {exp_name} | ckpt={best_ckpt} | Pearson(meta)={best_val:.4f}")

    if best_ckpt is None:
        print(f"[SKIP] No checkpoint found: {EXP_DIR}")
        return

    # Build and load model
    model = build_dual_enformer_for_vector(use_lora=True, lora_rank=32).to(device)
    obj = torch.load(best_ckpt, map_location=device)
    sd = _extract_state_dict(obj)

    loaded = False
    for prefix in ["module.", "model.", "network.", "", "enformer."]:
        try_sd = _strip_prefix(sd, prefix)
        try:
            missing, unexpected = model.load_state_dict(try_sd, strict=False)
            print(f"[CKPT] Loaded with prefix '{prefix}' | missing={len(missing)} unexpected={len(unexpected)}")
            loaded = True
            break
        except Exception:
            continue
    if not loaded:
        model.load_state_dict(sd, strict=False)
        print("[CKPT] Forced load with strict=False")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16, device=device)
        print("[DTYPE] Using bfloat16")
    else:
        model = model.to(dtype=torch.float32, device=device)
        print("[DTYPE] Using float32")

    # Evaluate on validation and test splits
    splits = {
        "validation": valid_samples_file,
        "test":       test_samples_file
    }
    for split_name, samples_path in splits.items():
        with open(samples_path) as f:
            samples = [l.strip() for l in f if l.strip()]
        targets_sub = targets_df_all.loc[samples, genes]
        loader = create_loader(train_genes_file, samples_path, processed_vector_dir, targets_sub)
        df = evaluate_per_gene(model, loader, device, genes, samples)

        # Save in requested format: {exp_name}_{split}.csv
        out = os.path.join(OUTPUT_DIR, f"{exp_name}_{split_name}.csv")
        df.to_csv(out, index=False)
        print(f"[RESULT] {split_name} -> {out}")

# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets_df_all = pd.read_csv(targets_csv_path, index_col=0)
    with open(train_genes_file) as f:
        genes = [l.strip() for l in f if l.strip()]

    print(f"[INFO] device={device} | #genes={len(genes)} | OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] Starting evaluation of {len(EXP_DIRS)} experiments")

    for exp_dir in EXP_DIRS:
        run_one_experiment(exp_dir, device, targets_df_all, genes)

if __name__ == "__main__":
    main()
