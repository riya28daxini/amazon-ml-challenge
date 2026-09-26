import pandas as pd
import os

files = [
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv"
]

for file in files:
    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    df = pd.read_csv(file, sep="\t")

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nDuplicate rows:", df.duplicated().sum())