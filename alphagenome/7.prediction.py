# ~/new_moon/pcc3.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PRED_DIR   = "/path/to/6.alphagenome/output"
TARGET_CSV = "/data/13.geuvadis_peer_normalized.csv"
GENE_LIST  = "/path/to/data/1.chr22_genes_list.txt"

OUT_DIR    = "/path/to/out_dir"
os.makedirs(OUT_DIR, exist_ok=True)

POLYA_TRACKS = [
    "rna_polya_plus_rna-seq_unstranded",
    "rna_polya_plus_rna-seq_+",
    "rna_polya_plus_rna-seq_-",
]
TOTAL_TRACKS = [
    "rna_total_rna-seq_unstranded",
    "rna_total_rna-seq_+",
    "rna_total_rna-seq_-",
]

def read_gene_strand_map(path):

    m = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                m[parts[0]] = parts[4]
    return m

def pick_track(columns, gene_strand, candidates):
    """
    Select tracks from candidates according to priority.
    Strand matching is prioritized:
      gene_strand == "+" → use *_+
      gene_strand == "-" → use *_-
      fallback: *_unstranded
    """

    if gene_strand == "+":
        for c in candidates:
            if c.endswith("_+") and c in columns:
                return c
    elif gene_strand == "-":
        for c in candidates:
            if c.endswith("_-") and c in columns:
                return c
    # fallback
    for c in candidates:
        if "unstranded" in c and c in columns:
            return c
    return None

def load_predictions(pred_dir, gene2strand, candidates):
    """
    From each individual CSV file under pred_dir, select columns that match the given track candidates,
    and construct a (individual × gene) prediction matrix.
    Average the values of hap1 and hap2.
    """
    tables = []
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue

        records = []
        for _, row in df.iterrows():
            gene_id = row["gene_id"]
            strand = gene2strand.get(gene_id, "?")
            track_col = pick_track(df.columns, strand, candidates)
            if track_col and track_col in row:
                records.append({
                    "individual": row["individual"],
                    "gene_id": gene_id,
                    "pred": row[track_col]
                })
        if records:
            grp = pd.DataFrame(records).groupby(["individual", "gene_id"], as_index=False)["pred"].mean()
            tables.append(grp)

    if not tables:
        raise RuntimeError(f"Could not find prediction CSV files for the candidate tracks {candidates} in {pred_dir}.")

    pred_df = pd.concat(tables, ignore_index=True)
    pred_mat = pred_df.pivot(index="individual", columns="gene_id", values="pred").sort_index(axis=0).sort_index(axis=1)
    return pred_mat

def align_and_compute_pcc(pred_mat, targ_mat):
    """
    Compute gene-wise PCC after aligning common individuals and genes.
    """
    common_inds  = sorted(set(pred_mat.index)  & set(targ_mat.index))
    common_genes = sorted(set(pred_mat.columns) & set(targ_mat.columns))

    if not common_inds or not common_genes:
        raise RuntimeError("Could not find common individuals or genes.")

    P = pred_mat.loc[common_inds, common_genes].to_numpy(dtype=float)
    T = targ_mat.loc[common_inds, common_genes].to_numpy(dtype=float)

    pccs = []
    for j in range(len(common_genes)):
        p = P[:, j]
        t = T[:, j]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() >= 2 and (p[ok].std() > 0) and (t[ok].std() > 0):
            r, _ = pearsonr(p[ok], t[ok])
            pccs.append(r)
        else:
            pccs.append(np.nan)

    per_gene = pd.Series(pccs, index=common_genes, name="pearson").sort_index()
    mean_pearson = float(np.nanmean(per_gene.values))
    return mean_pearson, per_gene, common_inds, common_genes

def main():
    targ_df = pd.read_csv(TARGET_CSV, index_col=0)
    targ_mat = targ_df.sort_index(axis=0).sort_index(axis=1)

    gene2strand = read_gene_strand_map(GENE_LIST)

    summary = []

    for label, candidates in [("polyA", POLYA_TRACKS), ("total", TOTAL_TRACKS)]:
        try:
            pred_mat = load_predictions(PRED_DIR, gene2strand, candidates)
            mean_pcc, per_gene_pcc, inds, genes = align_and_compute_pcc(pred_mat, targ_mat)
        except RuntimeError as e:
            print(f"[WARN] {label}: {e}")
            continue

        out_csv = os.path.join(OUT_DIR, f"per_gene_pcc_{label}.csv")
        per_gene_pcc.to_csv(out_csv, header=True)

        print(f"[DONE] {label}: mean PCC={mean_pcc:.4f}, {len(genes)} genes, {len(inds)} individuals")
        summary.append({"track_type": label, "mean_pearson": mean_pcc, "num_genes": len(genes), "num_inds": len(inds)})

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "summary_polyA_vs_total.csv"), index=False)
        print(f"[INFO] Summary saved -> {os.path.join(OUT_DIR, 'summary_polyA_vs_total.csv')}")

if __name__ == "__main__":
    main()

