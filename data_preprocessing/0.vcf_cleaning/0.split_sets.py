import pandas as pd
import numpy as np

df = pd.read_csv("/data/5.individual_id_sorted.txt", sep="\t", header=None, names=["individual"])

n_total = len(df)
assert n_total == 421, f"Expected 421 individuals, but got {n_total}"

rng = np.random.RandomState(seed=42)

individuals = df["individual"].tolist()
rng.shuffle(individuals)

train_ids = individuals[:295]
valid_ids = individuals[295:295+42]
test_ids  = individuals[295+42:]

pd.DataFrame(train_ids).to_csv("/data/2.train_samples_file_295.txt", index=False, header=False)
pd.DataFrame(valid_ids).to_csv("/data/3.valid_samples_file_42.txt", index=False, header=False)
pd.DataFrame(test_ids).to_csv("/data/4.test_samples_file_84.txt", index=False, header=False)

print(f"Train: {len(train_ids)}, Validation: {len(valid_ids)}, Test: {len(test_ids)}")
