"""
python3 2.adjust_fasta_length.py /data/5.individual_id_sorted.txt
"""
import os
import sys
import glob

def adjust_fasta_sequence(fasta_path, base_dir, out_base_dir, target_length=196608):
    with open(fasta_path, "r") as f:
        lines = f.readlines()

    # Remove headers and create sequence without line breaks
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

    seq_len = len(sequence)
    diff = target_length - seq_len

    if diff > 0:
        # Pad with N (equally at the front/back)
        left_pad = (diff + 1) // 2
        right_pad = diff // 2
        sequence = "N" * left_pad + sequence + "N" * right_pad
        print("Padding applied")
    elif diff < 0:
        diff = -diff
        left_trim = diff // 2
        right_trim = (diff + 1) // 2
        sequence = sequence[left_trim: seq_len - right_trim]

    assert len(sequence) == target_length, f"Adjusted length is {len(sequence)}, expected {target_length}"

    # Build output path
    rel_path = os.path.relpath(fasta_path, start=base_dir)
    out_path = os.path.join(out_base_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sequence = sequence.upper()
    # Save sequence as a single line
    with open(out_path, "w") as f:
        f.write(sequence + "\n")

    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python adjust_fasta_by_ids.py <id_file>")
        sys.exit(1)

    id_file = sys.argv[1]
    base_dir = "/path/to/1.vcf_to_fasta/output_dir"
    out_base_dir = "/path/to/output_dir"

    with open(id_file, "r") as f:
        individual_ids = [line.strip() for line in f if line.strip()]

    adjusted_files = []

    for ind_id in individual_ids:
        fasta_files = glob.glob(f"{base_dir}/{ind_id}/*.fa")
        for fasta_path in fasta_files:
            try:
                out_path = adjust_fasta_sequence(fasta_path, base_dir, out_base_dir)
                adjusted_files.append(out_path)
                print(f"✔ Saved: {out_path}")
            except Exception as e:
                print(f"❌ Failed: {fasta_path} - {e}")
