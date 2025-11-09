import os
import pandas as pd

input_dir = "/path/to/predixcan/0.run_predixcan.sh/pred_196608"
output_dir = "/path/to/pred_196608_cleaned"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    gene_id = filename.replace("predicted_expression_", "").replace(".txt", "")

    df = pd.read_csv(input_path, sep="\t")

    gene_col = [c for c in df.columns if c.startswith(gene_id)][0]

    new_gene_col = gene_col.split(".")[0]

    out_df = df[["IID", gene_col]].copy()
    out_df.rename(columns={gene_col: new_gene_col}, inplace=True)

    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Processed {filename} → {output_path}")
