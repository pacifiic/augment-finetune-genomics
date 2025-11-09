import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr
import numpy as np

pred_path = "/path/to/2.merge_prediction_out.py/out"
true_path = "/path/to/data/13.geuvadis_peer_normalized.csv"
test_samples_path = "/path/to/data/4.test_samples_file_84.txt"
output_path = os.path.expanduser("/path/to/out_directory")

# 0️⃣ Read test sample list
with open(test_samples_path, 'r') as f:
    test_samples = [line.strip() for line in f if line.strip()]

print(f"📋 Number of test samples: {len(test_samples)}")
print(f"📋 First 5 samples: {test_samples[:5]}")

# 1️⃣ Read files
pred = pd.read_csv(pred_path, sep=r"[\t,]", engine="python")
true = pd.read_csv(true_path, sep=r"[\t,]", engine="python")

# 2️⃣ Check column names (for debugging)
print("🔍 pred columns:", list(pred.columns)[:5])
print("🔍 true columns:", list(true.columns)[:5])

# 3️⃣ Standardize ID column names
if "IID" not in pred.columns:
    pred.rename(columns={pred.columns[0]: "IID"}, inplace=True)
if "IID" not in true.columns:
    true.rename(columns={true.columns[0]: "IID"}, inplace=True)

# 4️⃣ Filter for test samples only
true = true[true["IID"].isin(test_samples)]
print(f"✅ Number of test samples found in true file: {len(true)}")

# 5️⃣ Find and align common IIDs (within test samples)
common_iids = sorted(set(pred["IID"]).intersection(set(true["IID"])))
print(f"✅ Number of common test samples in pred and true: {len(common_iids)}")

pred = pred[pred["IID"].isin(common_iids)].set_index("IID").sort_index()
true = true[true["IID"].isin(common_iids)].set_index("IID").sort_index()

# 6️⃣ Find common gene columns
common_genes = sorted(set(pred.columns).intersection(set(true.columns)))
print(f"✅ Number of common genes: {len(common_genes)}")

# 7️⃣ Compute correlations
results = []
for gene in common_genes:
    y_true = true[gene].values
    y_pred = pred[gene].values

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        continue

    pearson_corr, _ = pearsonr(y_true[mask], y_pred[mask])
    spearman_corr, _ = spearmanr(y_true[mask], y_pred[mask])

    results.append({"gene_id": gene, "pearson": pearson_corr, "spearman": spearman_corr})

corr_df = pd.DataFrame(results)

# 8️⃣ Compute mean correlations
summary = pd.DataFrame([{
    "gene_id": "MEAN",
    "pearson": corr_df["pearson"].mean(),
    "spearman": corr_df["spearman"].mean()
}])

corr_df = pd.concat([corr_df, summary], ignore_index=True)

# 9️⃣ Save results
corr_df.to_csv(output_path, index=False)
print(f"🎉 Results saved → {output_path}")
print(f"📊 Mean Pearson: {corr_df[corr_df['gene_id'] == 'MEAN']['pearson'].values[0]:.4f}")
print(f"📊 Mean Spearman: {corr_df[corr_df['gene_id'] == 'MEAN']['spearman'].values[0]:.4f}")
