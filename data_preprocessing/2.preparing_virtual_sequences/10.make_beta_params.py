#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python 10.make_beta_params.py                               \
--pseudo /data/10.merged_virtual_sequence_pseudo_label.csv  \
--genes  /data/7.gene_id_chr22.txt                          \
--out    /data/12.virtual_beta_params.csv                   \
--pairs  200000 --low 0.1 --high 0.9 --seed 42
"""

import argparse
import numpy as np
import pandas as pd

def logit(p: float) -> float:
    return np.log(p / (1.0 - p))

def sample_abs_diffs(vals: np.ndarray, num_pairs: int, rng: np.random.Generator) -> np.ndarray:
    n = len(vals)
    if n < 2:
        return np.array([0.0], dtype=np.float64)
    i = rng.integers(0, n, size=num_pairs, endpoint=False)
    j = rng.integers(0, n, size=num_pairs, endpoint=False)
    same = (i == j)
    if np.any(same):  # 같은 인덱스 보정
        j[same] = (j[same] + 1) % n
    diffs = np.abs(vals[i] - vals[j]).astype(np.float64, copy=False)
    return diffs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--high", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.genes, "r") as f:
        genes = [line.strip() for line in f if line.strip()]

    head = pd.read_csv(args.pseudo, sep="\t", nrows=0)
    cols_available = set(head.columns)
    required_cols = {"IID", "hap"}
    missing_req = required_cols - cols_available

    genes_present = [g for g in genes if g in cols_available]
    genes_missing = [g for g in genes if g not in cols_available]

    usecols = ["IID", "hap"] + genes_present

    df = pd.read_csv(args.pseudo, sep="\t", usecols=usecols, low_memory=False)
    for g in genes_present:
        df[g] = pd.to_numeric(df[g], errors="coerce").astype("float32")
    df = df.dropna(subset=["IID", "hap"])
    df = df.set_index(["IID", "hap"])

    L_low  = logit(args.low)
    L_high = logit(args.high)

    rows = []
    for idx, g in enumerate(genes_present, 1):
        s = df[g].dropna()
        vals = s.to_numpy(dtype=np.float64, copy=False)
        n = len(vals)
        if n < 2:
            center, k = float(0.0), float(1.0)
            d10 = d50 = d90 = float(0.0)
            mean_d = std_d = float(0.0)
            n_pairs = 0
        else:
            n_pairs = min(args.pairs, n * (n - 1) // 2)
            diffs = sample_abs_diffs(vals, n_pairs, rng)
            d10, d50, d90 = np.quantile(diffs, [0.10, 0.50, 0.90])
            mean_d = float(diffs.mean())
            std_d  = float(diffs.std(ddof=0))
            if d90 > d10 + 1e-12:
                k = float((L_high - L_low) / (d90 - d10))
                center = float(d10 - (L_low / k))
            else:
                k = 1.0
                center = float(d50)
        rows.append({
            "gene": g,
            "center": center,
            "k": k,
            "d10": float(d10),
            "d50": float(d50),
            "d90": float(d90),
            "mean_absdiff": float(mean_d),
            "std_absdiff": float(std_d),
            "n_rows": int(n),
            "n_pairs_sampled": int(n_pairs)
        })

        if idx % 200 == 0:
            print(f"[{idx}/{len(genes_present)}] processed...")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] saved: {args.out} (genes={len(genes_present)})")

if __name__ == "__main__":
    main()

