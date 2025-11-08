# /media/leelabsg-storage1/moonwon/personalized-expression-benchmark/consensus/mergemerge.py
import os


LEFT_BASE  = "/path/to/3.adjust_fasta_length_left_half/out_dir"
RIGHT_BASE = "/path/to/4.adjust_fasta_length_right_half/out_dir"
MERGE_BASE = "/path/to/out_dir"

FINAL_LEN = 1_048_576

def read_seq(path: str) -> str:
    with open(path) as f:
        return "".join(line.strip() for line in f)

def process_individual(indiv: str):
    left_dir  = os.path.join(LEFT_BASE, indiv)
    right_dir = os.path.join(RIGHT_BASE, indiv)
    out_dir   = os.path.join(MERGE_BASE, indiv)

    if not (os.path.isdir(left_dir) and os.path.isdir(right_dir)):
        print(f"[SKIP] {indiv}: one of left/right does not exist")
        return

    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(left_dir):
        if not fname.endswith(".fa"):
            continue
        left_path  = os.path.join(left_dir, fname)
        right_path = os.path.join(right_dir, fname)
        if not os.path.isfile(right_path):
            print(f"[WARN] {right_path} does not exist")
            continue

        left_seq  = read_seq(left_path)
        right_seq = read_seq(right_path)
        merged    = left_seq + right_seq

        if len(merged) != FINAL_LEN:
            print(f"[ERR] {indiv}/{fname}: len={len(merged)}, expected_length={FINAL_LEN}")
            continue

        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as out:
            out.write(merged)

        print(f"[OK] {indiv}/{fname} saved! ({len(merged)}bp)")

def main():
    for indiv in sorted(os.listdir(LEFT_BASE)):
        if os.path.isdir(os.path.join(LEFT_BASE, indiv)):
            process_individual(indiv)

if __name__ == "__main__":
    main()
