import os
import pandas as pd

input_dir = "/path/to/1.preprocessing_after_prediction/pred_196608_cleaned"
output_file = "/path/to/merged_pred_196608_cleaned.csv"

all_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

merged_df = None

for file in sorted(all_files):
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path, sep="\t")

    gene_id = file.split("_")[-1].replace(".txt", "")

    df["sample"] = df["IID"].str.replace("_hap[12]", "", regex=True)
    df["hap"] = df["IID"].str.extract(r"hap(\d)").astype(int)

    df_gene = df[["sample", "hap", gene_id]]

    # merge
    if merged_df is None:
        merged_df = df_gene
    else:
        merged_df = pd.merge(merged_df, df_gene, on=["sample", "hap"], how="outer")

# ------------------------------
# 🔹 Sort IIDs numerically
# ------------------------------
# Extract numeric part from sample names (e.g., "sample23" → 23)

merged_df["sample_num"] = merged_df["sample"].str.extract(r"(\d+)").astype(int)

merged_df = merged_df.sort_values(by=["sample_num", "hap"]).drop(columns="sample_num")

merged_df = merged_df.rename(columns={"sample": "IID"})

merged_df.to_csv(output_file, sep="\t", index=False)
print(f"✅ Merged file saved to: {output_file}")
