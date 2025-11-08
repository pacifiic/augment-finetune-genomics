import pandas as pd

expr_file = "/data/13.geuvadis_peer_normalized.csv"
ids_file  = "/data/2.train_samples_file_295.txt"
genes_file = "/data/7.gene_id_ch22.txt"
out_file  = "/data/6.geuvadis_peer_normalized_filtered.csv"

df = pd.read_csv(expr_file, index_col=0)

with open(ids_file) as f:
    keep_ids = [line.strip() for line in f if line.strip()]

with open(genes_file) as f:
    keep_genes = [line.strip() for line in f if line.strip()]

filtered_df = df.loc[df.index.intersection(keep_ids), df.columns.intersection(keep_genes)]

filtered_df.to_csv(out_file)

