# /media/leelabsg-storage1/moonwon/personalized-expression-benchmark/consensus/merge2.py
import os

IN_BASE  = "/path/to/1.alphagenome_make_fasta_right_half/output_dir"
OUT_BASE = "/path/to/output_dir_for_right_half"

TARGET_LEN = 524288 - 1  # 524,287

def read_fasta_one_line(path: str) -> str:
    with open(path, "r") as f:
        seq = "".join(line.strip() for line in f if not line.startswith(">"))
    return seq.upper()

def adjust_left_pole(seq: str, target_len: int) -> str:
    n = len(seq)
    if n == target_len:
        return seq
    elif n < target_len:
        pad = "N" * (target_len - n)
        return seq + pad
    else:
        return seq[:target_len]

def process_individual(indiv: str):
    in_dir  = os.path.join(IN_BASE, indiv)
    out_dir = os.path.join(OUT_BASE, indiv)
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if not fname.endswith(".fa"):
            continue
        in_path  = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        seq = read_fasta_one_line(in_path)
        adj = adjust_left_pole(seq, TARGET_LEN)

        with open(out_path, "w") as out:
            out.write(adj)

        print(f"[OK] {indiv}/{fname} ({len(seq)} → {len(adj)})")

def main():
    for indiv in sorted(os.listdir(IN_BASE)):
        if os.path.isdir(os.path.join(IN_BASE, indiv)):
            process_individual(indiv)

if __name__ == "__main__":
    main()
