#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-gene DPO Beta Schedule (center, k) for matrix CSV (rows = samples, cols = genes)
- Input: expression matrix CSV (first column = sample ID; remaining columns = genes)
- Input: gene list file (txt, one gene_id per line)
- Output: CSV (gene, center, k, d10, d50, d90, mean_absdiff, std_absdiff, n_rows, n_pairs_sampled)

python beta_params_matrix.py                                    \
  --matrix /data/6.geuvadis_peer_normalized_chr22_filtered.csv  \
  --genes  /data/7.gene_id_ch22.txt                             \
  --out    /data/8.real_beta_params.txt                         \
  --pairs  200000 --low 0.1 --high 0.9 --seed 42
"""

import argparse
import numpy as np
import pandas as pd

def logit(p: float) -> float:
    return np.log(p / (1.0 - p))

def sample_abs_diffs(vals: np.ndarray, num_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly sample (i, j) pairs from vals and return |vals[i] - vals[j]|."""
    n = len(vals)
    if n < 2:
        return np.array([0.0], dtype=np.float64)
    i = rng.integers(0, n, size=num_pairs, endpoint=False)
    j = rng.integers(0, n, size=num_pairs, endpoint=False)
    same = (i == j)
    if np.any(same):
        j[same] = (j[same] + 1) % n
    return np.abs(vals[i] - vals[j]).astype(np.float64, copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="path to expression matrix CSV (first column = sample ID)")
    ap.add_argument("--genes",  required=True, help="gene list file (txt, one gene_id per line)")
    ap.add_argument("--out",    required=True, help="output CSV path")
    ap.add_argument("--pairs", type=int, default=200_000, help="number of random pairs per gene")
    ap.add_argument("--low",   type=float, default=0.10, help="lower sigmoid target probability (default 0.10)")
    ap.add_argument("--high",  type=float, default=0.90, help="upper sigmoid target probability (default 0.90)")
    ap.add_argument("--seed",  type=int, default=42, help="random seed")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.genes, "r") as f:
        gene_list = [line.strip() for line in f if line.strip()]
    gene_set = set(gene_list)

    df = pd.read_csv(args.matrix, index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    base_to_full = {}
    for c in df.columns:
        base = c.split(".", 1)[0]
        if base not in base_to_full:
            base_to_full[base] = c

    genes_present = []
    genes_missing = []
    for g in gene_list:
        if g in df.columns:
            genes_present.append(g)
        elif g in base_to_full:
            genes_present.append(base_to_full[g])
        else:
            genes_missing.append(g)

    if genes_missing:
        print(f"[WARN] {len(genes_missing)} genes not found in matrix are skipped. e.g. {genes_missing[:5]}")

    for g in genes_present:
        df[g] = pd.to_numeric(df[g], errors="coerce")
    L_low  = logit(args.low)
    L_high = logit(args.high)

    rows = []
    for idx, gcol in enumerate(genes_present, 1):
        s = df[gcol].dropna()
        vals = s.to_numpy(dtype=np.float64, copy=False)
        n = len(vals)

        if n < 2:
            center = 0.0
            k = 1.0
            d10 = d50 = d90 = 0.0
            mean_d = 0.0
            std_d  = 0.0
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
            "gene": gcol.split(".", 1)[0],           # 결과 gene은 버전 제거한 base ID로 기록
            "center": float(center),
            "k": float(k),
            "d10": float(d10),
            "d50": float(d50),
            "d90": float(d90),
            "mean_absdiff": float(mean_d),
            "std_absdiff": float(std_d),
            "n_rows": int(n),
            "n_pairs_sampled": int(n_pairs),
        })

        if idx % 200 == 0:
            print(f"[{idx}/{len(genes_present)}] processed...")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"[DONE] saved: {args.out} (genes={len(out_df)})")

if __name__ == "__main__":
    main()
