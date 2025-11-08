# -*- coding: utf-8 -*-
from typing import Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

from . import config
from .utils import safe_load_tensor

# Load once per worker
_targets_df = pd.read_csv(config.TARGETS_CSV, index_col=0)

class RealRegressionDataset(Dataset):
    def __init__(self, genes_file: str, samples_file: str, vector_dir: str):
        with open(samples_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]
        with open(genes_file, "r") as f:
            self.genes = [line.strip() for line in f if line.strip()]
        self.vector_dir = vector_dir

    def __len__(self):
        return len(self.genes) * len(self.samples) * 2

    def __getitem__(self, idx):
        sample_idx = (idx // 2) // len(self.genes)
        gene_idx = (idx // 2) % len(self.genes)
        gene = self.genes[gene_idx]
        sample = self.samples[sample_idx]
        if idx % 2 == 0:
            f1 = f"{self.vector_dir}/{sample}/{gene}.1pIu.pt"
            f2 = f"{self.vector_dir}/{sample}/{gene}.2pIu.pt"
        else:
            f1 = f"{self.vector_dir}/{sample}/{gene}.2pIu.pt"
            f2 = f"{self.vector_dir}/{sample}/{gene}.1pIu.pt"
        s1 = safe_load_tensor(f1)
        s2 = safe_load_tensor(f2)
        y = torch.tensor(_targets_df.loc[sample, gene].astype(np.float32))
        return s1, s2, y


class RealPairDataset(Dataset):
    def __init__(self, genes_file: str, samples_file: str, data_dir: str, pairs_per_epoch: Optional[int] = None):
        with open(samples_file, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]
        with open(genes_file, "r") as f:
            self.genes = [line.strip() for line in f if line.strip()]
        self.data_dir = data_dir
        if pairs_per_epoch is None:
            pairs_per_epoch = len(self.samples) * len(self.genes) * 2
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def _load_pair(self, sample: str, gene: str):
        f1 = f"{self.data_dir}/{sample}/{gene}.1pIu.pt"
        f2 = f"{self.data_dir}/{sample}/{gene}.2pIu.pt"
        s1 = safe_load_tensor(f1)
        s2 = safe_load_tensor(f2)
        y = float(_targets_df.loc[sample, gene])
        return s1, s2, torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        g = random.choice(self.genes)
        s1 = random.choice(self.samples)
        s2 = random.choice(self.samples)
        while s2 == s1:
            s2 = random.choice(self.samples)
        a1, a2, yA = self._load_pair(s1, g)
        b1, b2, yB = self._load_pair(s2, g)
        return a1, a2, yA, b1, b2, yB, g

class VirtualRegressionDataset(Dataset):
    def __init__(self, genes_file, virtual_vector_dir, pseudo_label_path,
                 n_virtual_samples=1000, samples_prefix="sample",
                 pairs_per_epoch=None):
        with open(genes_file, "r") as f:
            self.genes = [line.strip() for line in f if line.strip()]

        self.samples = [f"{samples_prefix}{i}" for i in range(1, n_virtual_samples + 1)]
        self.vector_dir = virtual_vector_dir

        self.pseudo_df = pd.read_csv(pseudo_label_path, sep="\t")
        self.pseudo_df.set_index(["IID", "hap"], inplace=True)

        if pairs_per_epoch is None:
            pairs_per_epoch = config.EUROPEAN_INDIVIDUAL_COUNT * len(self.genes) * 2
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        g = random.choice(self.genes)

        s = random.choice(self.samples)
        h = random.randint(1, 2)

        f = f"{self.vector_dir}/{s}/{g}.{h}pIu.pt"
        seq = safe_load_tensor(f)
        y = float(self.pseudo_df.loc[(s, h), g])

        return seq, torch.tensor(y, dtype=torch.float32), g

class VirtualPairDataset(Dataset):
    def __init__(self, genes_file: str, virtual_vector_dir: str, pseudo_label_path: str,
                 n_virtual_samples: int = 1000, pairs_per_epoch: Optional[int] = None):
        with open(genes_file, "r") as f:
            self.genes = [line.strip() for line in f if line.strip()]
        self.samples = [f"sample{i}" for i in range(1, n_virtual_samples + 1)]
        self.vector_dir = virtual_vector_dir
        self.pseudo_df = pd.read_csv(pseudo_label_path, sep="\t")
        self.pseudo_df.set_index(["IID", "hap"], inplace=True)
        if pairs_per_epoch is None:
            pairs_per_epoch = config.EUROPEAN_INDIVIDUAL_COUNT * len(self.genes) * 2
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        g = random.choice(self.genes)
        while True:
            s1 = random.choice(self.samples)
            s2 = random.choice(self.samples)
            h1 = random.randint(1, 2)
            h2 = random.randint(1, 2)
            if not (s1 == s2 and h1 == h2):
                break
        fA = f"{self.vector_dir}/{s1}/{g}.{h1}pIu.pt"
        fB = f"{self.vector_dir}/{s2}/{g}.{h2}pIu.pt"
        seqA = safe_load_tensor(fA)
        seqB = safe_load_tensor(fB)
        yA = float(self.pseudo_df.loc[(s1, h1), g])
        yB = float(self.pseudo_df.loc[(s2, h2), g])
        return seqA, torch.tensor(yA, dtype=torch.float32), seqB, torch.tensor(yB, dtype=torch.float32), g
