# ~/new_moon/alpha3.py
import os
import numpy as np
import pandas as pd

from alphagenome.models import dna_client

GENE_LIST_PATH   = "/path/to/data/1.chr22_genes_list.txt"
INDS_ROOT        = "/path/to/5.adjust_fasta_length_merge/out_dir"
ID_VALIDATION_TXT = "/path/to/data/4.test_samples_file_84.txt"
OUTPUT_DIR       = "/path/to/out_dir"

API_KEY = "add your api_key"
ONTOLOGY = "EFO:0002784"  # GM12878 (B lymphoblastoid cell line; B lymphoblastoid)
REQUESTED_OUTPUT = dna_client.OutputType.RNA_SEQ

TARGET_LEN = 1048576
CENTER_HALF_WINDOW_BP = 128 * 5  # = 640bp

def read_gene_strand_map(txt_path: str) -> dict:
    gene2strand = {}
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            if len(parts) < 5:
                continue
            gene_id = parts[0]
            strand = parts[4]
            gene2strand[gene_id] = strand
    return gene2strand

def load_fa_make_upper(path: str) -> str:
    with open(path, "r") as f:
        seq = f.read().strip()
    seq = seq.upper()
    return "".join([ch if ch in "ACGTN" else "N" for ch in seq])

def pad_or_center_crop_to_length(seq: str, target_len: int) -> str:
    L = len(seq)
    if L == target_len:
        return seq
    else:
        raise ValueError

def center_window_mean(values_2d: np.ndarray, half_window_bp: int) -> np.ndarray:
    L, T = values_2d.shape
    mid = L // 2
    s = max(0, mid - half_window_bp)
    e = min(L, mid + half_window_bp)
    return values_2d[s:e, :].mean(axis=0)

def make_track_keys_from_metadata(meta_df: pd.DataFrame) -> list:
    keys = []
    for _, row in meta_df.iterrows():
        assay = str(row["Assay title"]).lower().replace(" ", "_")
        strand = str(row["strand"]).strip()
        if strand in {"+", "-"}:
            key = f"{assay}_{strand}"
        else:
            key = f"{assay}_unstranded"
        keys.append(key)
    return keys


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dna_model = dna_client.create(API_KEY)

    gene2strand = read_gene_strand_map(GENE_LIST_PATH)

    with open(ID_VALIDATION_TXT, "r") as f:
        valid_inds = {line.strip() for line in f if line.strip()}

    for indiv in sorted(os.listdir(INDS_ROOT)):
        if indiv not in valid_inds:
            continue

        indiv_dir = os.path.join(INDS_ROOT, indiv)
        if not os.path.isdir(indiv_dir):
            continue

        rows = []
        print(f"[INFO] Processing {indiv} ...")

        for fname in sorted(os.listdir(indiv_dir)):
            if not fname.endswith(".fa"):
                continue
            try:
                gene_id, hap_part, _ = fname.split(".")
            except Exception:
                continue
            if not (hap_part.endswith("pIu") and hap_part[0] in "12"):
                continue
            haplotype = int(hap_part[0])

            fa_path = os.path.join(indiv_dir, fname)
            seq = load_fa_make_upper(fa_path)
            seq_fixed = pad_or_center_crop_to_length(seq, TARGET_LEN)

            try:
                output = dna_model.predict_sequence(
                    sequence=seq_fixed,
                    requested_outputs=[REQUESTED_OUTPUT],
                    ontology_terms=[ONTOLOGY],
                )
            except Exception as e:
                print(f"[WARN] predict_sequence failed for {indiv} {gene_id} hap{haplotype}: {e}")
                continue

            vals = output.rna_seq.values
            meta = output.rna_seq.metadata
            track_keys = make_track_keys_from_metadata(meta)
            means = center_window_mean(vals, CENTER_HALF_WINDOW_BP)

            row = {
                "individual": indiv,
                "gene_id": gene_id,
                "haplotype": haplotype,
                "gene_strand": gene2strand.get(gene_id, "?"),
                "sequence_len_used": len(seq_fixed),
                "center_half_window_bp": CENTER_HALF_WINDOW_BP,
                "ontology": ONTOLOGY,
            }
            for key, m in zip(track_keys, means):
                row[f"rna_{key}"] = float(m)

            rows.append(row)

        out_csv = os.path.join(OUTPUT_DIR, f"{indiv}.csv")
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[DONE] {indiv}: {len(rows)} rows -> {out_csv}")
        else:
            print(f"[DONE] {indiv}: no rows (skipped or failed).")

if __name__ == "__main__":
    main()

