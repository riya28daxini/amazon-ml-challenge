
# Amazon ML Challenge 2026 - Chunked S1 Candidate Generation
# Run this script in Google Colab.
#
# Expected folder:
# /content/student_resource/dataset/train/
#   train_source1.tsv
#   train_source2.tsv
#   train_source3.tsv
#   train_ground_truth.tsv

!pip -q install pyarrow

import os
import re
import unicodedata
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE_DIR = Path("/content/student_resource")
TRAIN_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "chunk_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"

# Start small.
CHUNK_SIZE = 50_000
MAX_CHUNKS = 1

NAME_MAP = {
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
}

ADDRESS_MAP = {
    "st": "street", "rd": "road", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "hwy": "highway", "ctr": "center",
    "ct": "court", "pl": "place", "pkwy": "parkway",
    "ste": "suite",
}

def clean_text(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x).lower())
    x = re.sub(r"[^\w\s]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def canonical(x, mapping):
    tokens = clean_text(x).split()
    out = []
    for token in tokens:
        out.extend(mapping.get(token, token).split())
    return " ".join(out)

def name_norm(x):
    return clean_text(x)

def name_canon(x):
    return canonical(x, NAME_MAP)

def address_norm(x):
    return clean_text(x)

def address_canon(x):
    return canonical(x, ADDRESS_MAP)

def country_norm(x):
    return "" if pd.isna(x) else str(x).strip().lower()

def build_indexes():
    print("Building S2/S3 indexes once...")
    indexes = {
        "name": defaultdict(list),
        "canon_name": defaultdict(list),
        "address": defaultdict(list),
        "name_address": defaultdict(list),
    }

    def add_file(path, source):
        total = 0
        for chunk in pd.read_csv(
            path, sep="\t", dtype=str,
            keep_default_na=False, chunksize=100_000
        ):
            for row in chunk.itertuples(index=False):
                c = country_norm(row.country)
                n = name_norm(row.business_name)
                cn = name_canon(row.business_name)
                a = address_norm(row.business_address)
                ca = address_canon(row.business_address)
                eid = row.entity_id

                if c and n:
                    indexes["name"][(c, n)].append(eid)
                if c and cn:
                    indexes["canon_name"][(c, cn)].append(eid)
                if c and a:
                    indexes["address"][(c, a)].append(eid)
                if c and cn and ca:
                    indexes["name_address"][(c, cn, ca)].append(eid)

            total += len(chunk)
            print(f"{source}: {total:,}", end="\r")

        print(f"{source}: {total:,} loaded")

    add_file(S2_FILE, "S2")
    add_file(S3_FILE, "S3")
    return indexes

def process_chunk(chunk, indexes, number):
    pairs = []

    for row in chunk.itertuples(index=False):
        c = country_norm(row.country)
        n = name_norm(row.business_name)
        cn = name_canon(row.business_name)
        a = address_norm(row.business_address)
        ca = address_canon(row.business_address)

        found = set()

        if c and n:
            found.update(indexes["name"].get((c, n), []))
        if c and cn:
            found.update(indexes["canon_name"].get((c, cn), []))
        if c and a:
            found.update(indexes["address"].get((c, a), []))
        if c and cn and ca:
            found.update(indexes["name_address"].get((c, cn, ca), []))

        for eid in found:
            source = "S2" if str(eid).startswith("S2-") else "S3"
            pairs.append((row.entity_id, eid, source))

    result = pd.DataFrame(
        pairs,
        columns=["source1_entity_id", "candidate_entity_id", "candidate_source"]
    )

    path = OUTPUT_DIR / f"candidates_chunk_{number:03d}.parquet"
    result.to_parquet(path, index=False)

    print(f"\nChunk {number}")
    print(f"S1 records: {len(chunk):,}")
    print(f"Candidate pairs: {len(result):,}")
    print(f"Average candidates/S1: {len(result)/max(len(chunk),1):.2f}")
    print(f"Saved: {path}")

print("=" * 70)
print("AMAZON ML 2026 - CHUNKED SOURCE-1 PIPELINE")
print("=" * 70)

for f in [S1_FILE, S2_FILE, S3_FILE]:
    print(f"{f}: {'FOUND' if f.exists() else 'MISSING'}")

if not S1_FILE.exists():
    raise FileNotFoundError(
        f"{S1_FILE} not found. Mount/upload student_resource to /content."
    )

indexes = build_indexes()

chunk_no = 0
for chunk in pd.read_csv(
    S1_FILE, sep="\t", dtype=str,
    keep_default_na=False, chunksize=CHUNK_SIZE
):
    chunk_no += 1
    print("\n" + "=" * 70)
    print(f"PROCESSING S1 CHUNK {chunk_no}")
    print("=" * 70)

    process_chunk(chunk, indexes, chunk_no)

    if MAX_CHUNKS is not None and chunk_no >= MAX_CHUNKS:
        break

print("\nDONE")
print(f"Processed chunks: {chunk_no}")
print(f"Output: {OUTPUT_DIR}")
