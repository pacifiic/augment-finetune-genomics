# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

PREDIXCAN_PATH = "/media/leelabsg-storage0/moonwon/prediXcanDB/output/predicted_expression_chr22_var_im_1016_retry_for_check.txt"
TARGETS_PATH   = "/media/leelabsg-storage0/moonwon/recomb/chr22_csv_directory/targets_df.csv"
GENES_FILE     = "/media/leelabsg-storage0/moonwon/recomb/chr22_csv_directory/genes_file.txt"
VALID_SAMPLES  = "/media/leelabsg-storage0/moonwon/3.valid_samples_file_42.txt"
TEST_SAMPLES   = "/media/leelabsg-storage0/moonwon/4.test_samples_file_84.txt"
OUTPUT_DIR     = "/media/leelabsg-storage0/moonwon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load PrediXcan ----
pred_df = pd.read_csv(PREDIXCAN_PATH, sep="\t")
pred_df.columns = [c.strip() for c in pred_df.columns]
pred_df.columns = [c.split(".")[0] if c.startswith("ENSG") else c for c in pred_df.columns]

if "IID" not in pred_df.columns:
    candidates = [c for c in pred_df.columns if c.lower() == "iid"]
    if candidates:
        pred_df.rename(columns={candidates[0]: "IID"}, inplace=True)
    else:
        raise KeyError("IID column not found in PrediXcan output.")
pred_df = pred_df.set_index("IID")

# ---- Load targets ----
targets_df = pd.read_csv(TARGETS_PATH, index_col=0)
with open(GENES_FILE) as f:
    gene_list = [g.strip() for g in f if g.strip()]

pred_genes = [c for c in pred_df.columns if c.startswith("ENSG")]
common_genes = [g for g in gene_list if g in pred_genes and g in targets_df.columns]
if len(common_genes) == 0:
    raise ValueError("No common genes found between PrediXcan and targets_df.")

common_samples = [s for s in pred_df.index if s in targets_df.index]
print(f"[INFO] Common genes: {len(common_genes)} | Common samples: {len(common_samples)}")

def evaluate_split(split_name, sample_file):
    with open(sample_file) as f:
        samples = [s.strip() for s in f if s.strip()]
    samples = [s for s in samples if s in common_samples]
    print(f"[SPLIT] {split_name}: {len(samples)} samples")

    pred_sub = pred_df.loc[samples, common_genes]
    targ_sub = targets_df.loc[samples, common_genes]

    rows, pearsons, spearmans = [], [], []
    for g in common_genes:
        p, t = pred_sub[g].to_numpy(), targ_sub[g].to_numpy()
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() >= 2:
            pear = pearsonr(p[mask], t[mask])[0]
            spear = spearmanr(p[mask], t[mask])[0]
        else:
            pear, spear = np.nan, np.nan
        rows.append({"gene": g, "pearson": pear, "spearman": spear, "n_samples": len(samples)})
        if not np.isnan(pear): pearsons.append(pear)
        if not np.isnan(spear): spearmans.append(spear)

    rows.append({
        "gene": "MEAN",
        "pearson": np.mean(pearsons) if pearsons else np.nan,
        "spearman": np.mean(spearmans) if spearmans else np.nan,
        "n_samples": len(samples)
    })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"per_gene_corr__predixcan__{split_name}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"[RESULT] Saved -> {out_path}")
    return df_out

if __name__ == "__main__":
    evaluate_split("validation", VALID_SAMPLES)
    evaluate_split("test", TEST_SAMPLES)
