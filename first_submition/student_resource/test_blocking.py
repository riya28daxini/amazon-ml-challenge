import pandas as pd
import re
import unicodedata


def normalize_name(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    # Unicode normalization
    x = unicodedata.normalize("NFKC", x)

    # Replace punctuation with spaces
    x = re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", x)

    # Normalize whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x


print("Loading data...")

s1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Loaded.")


# --------------------------------------------------
# Normalize names
# --------------------------------------------------

print("Normalizing names...")

for df in [s1, s2, s3]:
    df["name_norm"] = df["business_name"].map(normalize_name)


# --------------------------------------------------
# Build lookup indexes
# --------------------------------------------------

print("Building indexes...")

s2_index = {}

for idx, row in s2.iterrows():
    key = (row["country"], row["name_norm"])

    if row["name_norm"]:
        s2_index.setdefault(key, []).append(row["entity_id"])


s3_index = {}

for idx, row in s3.iterrows():
    key = (row["country"], row["name_norm"])

    if row["name_norm"]:
        s3_index.setdefault(key, []).append(row["entity_id"])


# --------------------------------------------------
# Evaluate ground truth
# --------------------------------------------------

print("Evaluating...")

total_true = 0
retrieved_true = 0

s1_lookup = dict(
    zip(s1["entity_id"], zip(s1["country"], s1["name_norm"]))
)

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matches = row["matched_entity_ids"]

    if pd.isna(matches) or not str(matches).strip():
        continue

    country, name = s1_lookup[s1_id]

    candidates = set()

    key = (country, name)

    candidates.update(s2_index.get(key, []))
    candidates.update(s3_index.get(key, []))

    true_ids = set(str(matches).split(","))

    total_true += len(true_ids)
    retrieved_true += len(true_ids & candidates)


print("\n========================================")
print("RESULT")
print("========================================")

print("Total true matches:", total_true)
print("True matches retrieved:", retrieved_true)

if total_true:
    print(
        "Recall:",
        retrieved_true / total_true
    )