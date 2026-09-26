import pandas as pd
from collections import Counter

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# Empty / singleton Source 1 entities
empty = gt["matched_entity_ids"].isna() | (
    gt["matched_entity_ids"].fillna("").str.strip() == ""
)

print("Total Source 1:", len(gt))
print("Singletons:", empty.sum())
print("Non-singletons:", (~empty).sum())

# Number of matched records per S1
match_count = (
    gt["matched_entity_ids"]
    .fillna("")
    .apply(lambda x: 0 if not x.strip() else len(x.split(",")))
)

print("\nMatch count statistics:")
print(match_count.describe())

print("\nDistribution:")
print(match_count.value_counts().sort_index().head(30))

# Source 2 vs Source 3 matches
def count_sources(x):
    if not x.strip():
        return 0, 0

    ids = x.split(",")

    s2 = sum(i.startswith("S2-") for i in ids)
    s3 = sum(i.startswith("S3-") for i in ids)

    return s2, s3


counts = gt["matched_entity_ids"].fillna("").apply(count_sources)

s2_counts = counts.apply(lambda x: x[0])
s3_counts = counts.apply(lambda x: x[1])

print("\nSource 2 matches:")
print(s2_counts.describe())

print("\nSource 3 matches:")
print(s3_counts.describe())

print("\nS1 entities with S2 matches:", (s2_counts > 0).sum())
print("S1 entities with S3 matches:", (s3_counts > 0).sum())

print("\nBoth S2 and S3:")
print(((s2_counts > 0) & (s3_counts > 0)).sum())