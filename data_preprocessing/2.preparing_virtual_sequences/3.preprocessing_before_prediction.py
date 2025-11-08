import os
import gzip

"""
Important:
You must use OUT_DIR1 from 2.adjust_vcf_length.sh as the input — not OUT_DIR2.

Regarding the complexity of the workflow, I sincerely apologize for any inconvenience.
An improved version will be uploaded in the near future.
"""
input_dir = "/path/to/2.adjust_vcf_length.sh/OUT_DIR1"
output_dir = "/path/to/output_dir"
os.makedirs(output_dir, exist_ok=True)
"""
When predicting gene expression with PrediXcan,
to create a setting where gene expression is predicted per haplotype, 
each individual column in the VCF file is constructed to contain two copies of a single haplotype.
"""
def process_vcf_file(input_path, output_path):
    with gzip.open(input_path, "rt") as f:
        lines = f.readlines()

    header_lines = []
    sample_names = []
    data_lines = []

    for line in lines:
        if line.startswith("##"):
            header_lines.append(line)
        elif line.startswith("#CHROM"):
            header_parts = line.strip().split("\t")
            sample_names = header_parts[9:]
            header_lines.append(line)
        else:
            data_lines.append(line.strip())

    # hap1/hap2 이름 생성
    new_sample_names = []
    for s in sample_names:
        hap1_name = f"{s}_hap1"
        hap2_name = f"{s}_hap2"
        new_sample_names.extend([hap1_name, hap2_name])

    # 새 헤더
    new_header = []
    for line in header_lines:
        if line.startswith("#CHROM"):
            parts = line.strip().split("\t")
            new_line = "\t".join(parts[:9] + new_sample_names) + "\n"
            new_header.append(new_line)
        else:
            new_header.append(line if line.endswith("\n") else line + "\n")

    new_data_lines = []
    for line in data_lines:
        parts = line.strip().split("\t")
        fixed_fields = parts[:9]
        genotypes = parts[9:]

        new_genotypes = []
        for gt in genotypes:
            if "|" in gt:
                hap1, hap2 = gt.split("|")
            elif "/" in gt:
                hap1, hap2 = gt.split("/")
            else:
                hap1, hap2 = ".", "."

            new_genotypes.append(f"{hap1}|{hap1}")
            new_genotypes.append(f"{hap2}|{hap2}")

        new_line = "\t".join(fixed_fields + new_genotypes) + "\n"
        new_data_lines.append(new_line)

    with open(output_path, "w") as out:
        out.writelines(new_header)
        out.writelines(new_data_lines)

if __name__ == "__main__":
    for filename in os.listdir(input_dir):
        if filename.endswith(".vcf.gz"):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace(".vcf.gz", ".vcf")
            output_path = os.path.join(output_dir, output_filename)
            print(f"Processing {filename} -> {output_path}")
            process_vcf_file(input_path, output_path)