import pandas as pd
import re
import unicodedata
from collections import defaultdict, Counter
from time import time

# ============================================================
# Amazon ML Challenge 2026 - Multi-pass blocking evaluation
#
# This script evaluates blocking recall on TRAINING data.
# It does NOT train the final ML model.
#
# Blocking strategies:
#   B1: country + normalized name
#   B2: country + canonical name
#   B3: country + normalized address
#   B4: country + canonical name + canonical address
#   B5: country + informative business-name tokens
#
# The final candidate set is the UNION of all strategies.
# ============================================================

TRAIN_DIR = "dataset/train"
S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"
GT_FILE = f"{TRAIN_DIR}/train_ground_truth.tsv"

# Token blocking controls. Lower values reduce candidate explosion.
MAX_TOKEN_FREQUENCY = 500
MIN_TOKEN_LENGTH = 3
MAX_TOKENS_PER_RECORD = 5

# Set True for a fast first experiment; False for the full dataset.
QUICK_TEST = False
QUICK_SAMPLE_SIZE = 50000
RANDOM_STATE = 42

NAME_ABBREVIATIONS = {
    "pvt": "private", "pvt.": "private",
    "ltd": "limited", "ltd.": "limited",
    "corp": "corporation", "corp.": "corporation",
    "inc": "incorporated", "inc.": "incorporated",
    "co": "company", "co.": "company",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "st.": "street",
    "rd": "road", "rd.": "road",
    "ave": "avenue", "ave.": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "ctr": "center", "ct": "court",
    "pl": "place", "pkwy": "parkway", "ste": "suite",
}


def basic_clean(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKC", str(x).lower())
    x = re.sub(r"[^\w\s]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def normalize_name(x):
    return basic_clean(x)


def canonical_name(x):
    tokens = basic_clean(x).split()
    out = []
    for token in tokens:
        out.extend(NAME_ABBREVIATIONS.get(token, token).split())
    return " ".join(out)


def normalize_address(x):
    return basic_clean(x)


def canonical_address(x):
    tokens = basic_clean(x).split()
    out = []
    for token in tokens:
        out.extend(ADDRESS_ABBREVIATIONS.get(token, token).split())
    return " ".join(out)


def country_clean(x):
    return "" if pd.isna(x) else str(x).strip().lower()


def preprocess(df, label):
    print(f"Preprocessing {label}...")
    df = df.copy()
    df["country_norm"] = df["country"].map(country_clean)
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["name_canon"] = df["business_name"].map(canonical_name)
    df["address_norm"] = df["business_address"].map(normalize_address)
    df["address_canon"] = df["business_address"].map(canonical_address)
    df["name_tokens"] = df["name_canon"].map(
        lambda x: {t for t in x.split() if len(t) >= MIN_TOKEN_LENGTH}
    )
    return df


def build_index(df, *columns):
    index = defaultdict(list)
    for row in df.itertuples(index=False):
        values = [getattr(row, c) for c in columns]
        if all(values):
            index[tuple([row.country_norm] + values)].append(row.entity_id)
    return index


def build_token_index(s2, s3):
    # Count token frequency across S2+S3 by country.
    freq = Counter()
    for df in (s2, s3):
        for row in df.itertuples(index=False):
            for token in row.name_tokens:
                freq[(row.country_norm, token)] += 1

    index = defaultdict(list)
    for df in (s2, s3):
        for row in df.itertuples(index=False):
            tokens = [
                t for t in row.name_tokens
                if freq[(row.country_norm, t)] <= MAX_TOKEN_FREQUENCY
            ]
            tokens.sort(key=lambda t: freq[(row.country_norm, t)])
            for token in tokens[:MAX_TOKENS_PER_RECORD]:
                index[(row.country_norm, token)].append(row.entity_id)
    return index, freq


def merge_indexes(*indexes):
    result = defaultdict(list)
    for index in indexes:
        for key, values in index.items():
            result[key].extend(values)
    return result


def build_ground_truth(gt):
    truth = {}
    total_true = 0
    singletons = 0
    for row in gt.itertuples(index=False):
        value = row.matched_entity_ids
        if pd.isna(value) or not str(value).strip():
            truth[row.source1_entity_id] = set()
            singletons += 1
        else:
            ids = {x.strip() for x in str(value).split(",") if x.strip()}
            truth[row.source1_entity_id] = ids
            total_true += len(ids)
    return truth, total_true, singletons


def get_candidates(row, indexes):
    country = row.country_norm
    result = {
        "B1_exact_name": set(),
        "B2_canonical_name": set(),
        "B3_address": set(),
        "B4_name_address": set(),
        "B5_name_token": set(),
    }

    if row.name_norm:
        result["B1_exact_name"].update(
            indexes["name_norm"].get((country, row.name_norm), [])
        )
    if row.name_canon:
        result["B2_canonical_name"].update(
            indexes["name_canon"].get((country, row.name_canon), [])
        )
    if row.address_norm:
        result["B3_address"].update(
            indexes["address_norm"].get((country, row.address_norm), [])
        )
    if row.name_canon and row.address_canon:
        result["B4_name_address"].update(
            indexes["name_address"].get(
                (country, row.name_canon, row.address_canon), []
            )
        )

    tokens = [
        t for t in row.name_tokens
        if indexes["token_frequency"].get((country, t), MAX_TOKEN_FREQUENCY + 1)
        <= MAX_TOKEN_FREQUENCY
    ]
    tokens.sort(
        key=lambda t: indexes["token_frequency"].get((country, t), 10**9)
    )
    for token in tokens[:MAX_TOKENS_PER_RECORD]:
        result["B5_name_token"].update(
            indexes["name_token"].get((country, token), [])
        )
    return result


def evaluate(s1, truth, indexes, sample_ids=None):
    strategies = [
        "B1_exact_name", "B2_canonical_name", "B3_address",
        "B4_name_address", "B5_name_token"
    ]
    retrieved = {s: 0 for s in strategies}
    candidate_total = {s: 0 for s in strategies}
    union_retrieved = 0
    union_candidate_total = 0
    union_max = 0

    s1_rows = {r.entity_id: r for r in s1.itertuples(index=False)}
    ids = sample_ids if sample_ids is not None else s1["entity_id"].tolist()
    total_true = 0
    start = time()

    print("\nEvaluating blocking strategies...")
    for n, s1_id in enumerate(ids, 1):
        row = s1_rows[s1_id]
        true_ids = truth.get(s1_id, set())
        blocks = get_candidates(row, indexes)
        union = set()

        for strategy in strategies:
            c = blocks[strategy]
            candidate_total[strategy] += len(c)
            if true_ids:
                retrieved[strategy] += len(true_ids & c)
            union.update(c)

        total_true += len(true_ids)
        union_retrieved += len(true_ids & union)
        union_candidate_total += len(union)
        union_max = max(union_max, len(union))

        if n % 100000 == 0:
            print(f"Processed {n:,}/{len(ids):,} S1 entities; elapsed {time()-start:.1f}s")

    print("\n" + "=" * 72)
    print("BLOCKING EVALUATION RESULT")
    print("=" * 72)
    print(f"Entities evaluated       : {len(ids):,}")
    print(f"True matches evaluated   : {total_true:,}")

    rows = []
    for strategy in strategies:
        recall = retrieved[strategy] / total_true if total_true else 0
        avg_cand = candidate_total[strategy] / len(ids) if ids else 0
        print(f"\n{strategy}")
        print(f"  True matches retrieved : {retrieved[strategy]:,}")
        print(f"  Recall                 : {recall:.6f} ({recall*100:.2f}%)")
        print(f"  Avg candidates / S1    : {avg_cand:.2f}")
        rows.append({
            "strategy": strategy,
            "retrieved_true_matches": retrieved[strategy],
            "recall": recall,
            "avg_candidates_per_s1": avg_cand,
        })

    union_recall = union_retrieved / total_true if total_true else 0
    union_avg = union_candidate_total / len(ids) if ids else 0

    print("\n" + "-" * 72)
    print("MULTI-PASS UNION")
    print("-" * 72)
    print(f"True matches retrieved : {union_retrieved:,}")
    print(f"Union recall           : {union_recall:.6f} ({union_recall*100:.2f}%)")
    print(f"Avg candidates / S1    : {union_avg:.2f}")
    print(f"Maximum candidates     : {union_max:,}")

    rows.append({
        "strategy": "UNION_ALL",
        "retrieved_true_matches": union_retrieved,
        "recall": union_recall,
        "avg_candidates_per_s1": union_avg,
    })
    pd.DataFrame(rows).to_csv("blocking_results.csv", index=False)
    print("\nSaved comparison to blocking_results.csv")


def main():
    start = time()
    print("Loading data...")
    s1 = pd.read_csv(S1_FILE, sep="\t")
    s2 = pd.read_csv(S2_FILE, sep="\t")
    s3 = pd.read_csv(S3_FILE, sep="\t")
    gt = pd.read_csv(GT_FILE, sep="\t")
    print("Loaded.")
    print(f"Source 1: {len(s1):,}")
    print(f"Source 2: {len(s2):,}")
    print(f"Source 3: {len(s3):,}")

    s1 = preprocess(s1, "Source 1")
    s2 = preprocess(s2, "Source 2")
    s3 = preprocess(s3, "Source 3")

    truth, total_true, singletons = build_ground_truth(gt)
    print(f"\nTotal true matches: {total_true:,}")
    print(f"Singleton Source 1 entities: {singletons:,}")

    print("\nBuilding exact/canonical/address indexes...")
    name_norm = merge_indexes(
        build_index(s2, "name_norm"),
        build_index(s3, "name_norm")
    )
    name_canon = merge_indexes(
        build_index(s2, "name_canon"),
        build_index(s3, "name_canon")
    )
    address_norm = merge_indexes(
        build_index(s2, "address_norm"),
        build_index(s3, "address_norm")
    )
    name_address = merge_indexes(
        build_index(s2, "name_canon", "address_canon"),
        build_index(s3, "name_canon", "address_canon")
    )

    print("Building informative token index...")
    token_index, token_frequency = build_token_index(s2, s3)

    indexes = {
        "name_norm": name_norm,
        "name_canon": name_canon,
        "address_norm": address_norm,
        "name_address": name_address,
        "name_token": token_index,
        "token_frequency": token_frequency,
    }

    sample_ids = None
    if QUICK_TEST:
        sample_ids = s1["entity_id"].sample(
            n=min(QUICK_SAMPLE_SIZE, len(s1)),
            random_state=RANDOM_STATE
        ).tolist()
        print(f"\nQUICK_TEST=True: evaluating {len(sample_ids):,} random S1 entities.")

    evaluate(s1, truth, indexes, sample_ids)
    print(f"\nTotal runtime: {time()-start:.1f} seconds")


if __name__ == "__main__":
    main()
