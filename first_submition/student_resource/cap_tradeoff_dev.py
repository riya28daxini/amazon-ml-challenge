#!/usr/bin/env python3
"""
Amazon ML Challenge 2026
cap_tradeoff_dev.py

This replacement keeps the SAME filename as your current script.

What it does:
1. Loads all Source 2 and Source 3 records into SQLite.
2. Takes the first 10,000 Source 1 records (development run).
3. Uses the same 10 blocking keys from the previous cap experiment.
4. Builds a candidate pool up to CAP=2,000 for every S1.
5. Saves the actual S1 -> S2/S3 candidate relationships.
6. Calculates similarity features for candidates.
7. Creates a BASELINE final prediction using a conservative similarity rule.
8. Creates:
      candidate_pairs.tsv
      predictions.tsv
      matching_results.tsv
      f05_entity_details.tsv
      f05_summary.txt
      candidate_cap_results.csv

IMPORTANT:
- This is a development baseline on the first 10,000 S1 rows.
- It does NOT use ground truth to make predictions.
- Ground truth is used ONLY to evaluate F0.5.
- Source 1 is matched against BOTH Source 2 and Source 3.
- No external API/database is used.
- The country value is treated as a normal string and is not hard-coded.

The baseline prediction uses name/address similarity. It is NOT yet the
final LightGBM/XGBoost model. It gives you a real prediction file and
a real Macro F0.5 baseline from the current blocking strategy.
"""

import csv
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Same development size as your previous successful experiment.
S1_SAMPLE_SIZE = 10_000

# Use the best tested cap from your previous experiment.
CAP = 2_000

# Candidate output. This is the actual S1 -> S2/S3 candidate file.
CANDIDATE_FILE = os.path.join(BASE_DIR, "candidate_pairs.tsv")

# Final baseline prediction.
PREDICTION_FILE = os.path.join(BASE_DIR, "predictions.tsv")

# Cleaned/standardized final-result-format file.
RESULT_FILE = os.path.join(BASE_DIR, "matching_results.tsv")

DETAIL_FILE = os.path.join(BASE_DIR, "f05_entity_details.tsv")
SUMMARY_FILE = os.path.join(BASE_DIR, "f05_summary.txt")
CAP_RESULT_FILE = os.path.join(BASE_DIR, "candidate_cap_results.csv")

DB_FILE = os.path.join(BASE_DIR, "cap_tradeoff.sqlite")

# Conservative baseline threshold.
# We will keep a candidate if either:
#   name similarity is strong AND address has some evidence
# OR
#   name similarity is extremely strong
NAME_THRESHOLD = 88
VERY_STRONG_NAME = 95
ADDRESS_THRESHOLD = 70

# ============================================================
# NORMALIZATION
# ============================================================

NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
}

ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "ctr": "center",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "ste": "suite",
}

STOPWORDS = {
    "the", "and", "of", "for", "in", "at", "on",
    "private", "limited", "incorporated", "corporation",
    "company", "llc", "llp"
}


def clean_text(value):
    if value is None:
        return ""

    value = str(value).lower()
    value = unicodedata.normalize("NFKC", value)

    # Keep unicode letters/numbers and whitespace.
    value = re.sub(r"[^\w\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def canonical_name(value):
    tokens = clean_text(value).split()
    out = []

    for token in tokens:
        out.extend(NAME_ABBREVIATIONS.get(token, token).split())

    return " ".join(out)


def canonical_address(value):
    tokens = clean_text(value).split()
    out = []

    for token in tokens:
        out.extend(ADDRESS_ABBREVIATIONS.get(token, token).split())

    return " ".join(out)


def country_norm(value):
    return clean_text(value)


def prefix(value, n):
    value = value.replace(" ", "")
    return value[:n] if value else ""


def suffix(value, n):
    value = value.replace(" ", "")
    return value[-n:] if value else ""


def informative_tokens(value):
    tokens = []

    for token in value.split():
        if len(token) >= 3 and token not in STOPWORDS:
            tokens.append(token)

    return tokens


def first4(value):
    tokens = informative_tokens(value)
    return tokens[0][:4] if tokens else ""


def last4(value):
    tokens = informative_tokens(value)
    return tokens[-1][:4] if tokens else ""


def postal_code(value):
    """
    Extract a likely postal/PIN/ZIP code without assuming a country.
    Supports common 4-10 digit forms.
    """
    value = str(value)

    matches = re.findall(r"\b\d{4,10}\b", value)

    if not matches:
        return ""

    # Prefer the last numeric postal-like token.
    return matches[-1]


# ============================================================
# SIMPLE STRING SIMILARITY
# ============================================================

def levenshtein_ratio(a, b):
    """
    Standard normalized Levenshtein similarity.
    Returns 0..100.

    Implemented without external packages so the script works with
    normal Python installation.
    """
    if a == b:
        return 100.0

    if not a or not b:
        return 0.0

    # Keep the shorter string on the columns to reduce memory.
    if len(a) < len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a

    previous = list(range(len(short) + 1))

    for i, c2 in enumerate(long_, start=1):
        current = [i]

        for j, c1 in enumerate(short, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (c1 != c2)

            current.append(
                min(insert_cost, delete_cost, replace_cost)
            )

        previous = current

    distance = previous[-1]
    maximum = max(len(a), len(b))

    return 100.0 * (1.0 - distance / maximum)


def token_set_ratio(a, b):
    """
    Approximate token-set similarity.
    """
    ta = set(informative_tokens(a))
    tb = set(informative_tokens(b))

    if not ta or not tb:
        return 0.0

    intersection = len(ta & tb)
    union = len(ta | tb)

    if union == 0:
        return 0.0

    return 100.0 * intersection / union


def char_jaccard(a, b, n=3):
    def grams(s):
        s = s.replace(" ", "")
        if len(s) < n:
            return {s} if s else set()

        return {
            s[i:i+n]
            for i in range(len(s) - n + 1)
        }

    ga = grams(a)
    gb = grams(b)

    if not ga or not gb:
        return 0.0

    return 100.0 * len(ga & gb) / len(ga | gb)


def similarity_features(s1, candidate):
    """
    Candidate dictionaries contain normalized/canonical fields.
    """
    name = s1["name_canon"]
    cname = candidate["name_canon"]

    address = s1["address_canon"]
    caddress = candidate["address_canon"]

    name_lev = levenshtein_ratio(name, cname)
    name_tok = token_set_ratio(name, cname)
    name_char = char_jaccard(name, cname)

    addr_lev = levenshtein_ratio(address, caddress)
    addr_tok = token_set_ratio(address, caddress)
    addr_char = char_jaccard(address, caddress)

    postal_match = (
        bool(s1["postal"])
        and bool(candidate["postal"])
        and s1["postal"] == candidate["postal"]
    )

    country_match = (
        bool(s1["country"])
        and bool(candidate["country"])
        and s1["country"] == candidate["country"]
    )

    name_score = max(name_lev, name_tok, name_char)

    address_score = max(addr_lev, addr_tok, addr_char)

    # Weighted score. Name gets more weight because business-name
    # variation is generally more informative than raw address overlap.
    score = (
        0.60 * name_score
        + 0.30 * address_score
        + 8.0 * int(postal_match)
        + 2.0 * int(country_match)
    )

    return {
        "name_lev": name_lev,
        "name_tok": name_tok,
        "name_char": name_char,
        "address_lev": addr_lev,
        "address_tok": addr_tok,
        "address_char": addr_char,
        "postal_match": int(postal_match),
        "country_match": int(country_match),
        "score": score,
    }


def is_prediction(features):
    """
    Conservative baseline decision.

    We deliberately prefer precision because the challenge uses F0.5,
    which weights precision more heavily than recall.
    """
    name_score = max(
        features["name_lev"],
        features["name_tok"],
        features["name_char"],
    )

    address_score = max(
        features["address_lev"],
        features["address_tok"],
        features["address_char"],
    )

    # Very strong name match.
    if name_score >= VERY_STRONG_NAME:
        return True

    # Strong name + supporting address.
    if (
        name_score >= NAME_THRESHOLD
        and address_score >= ADDRESS_THRESHOLD
    ):
        return True

    # Very strong token/name evidence plus exact postal.
    if (
        name_score >= 85
        and features["postal_match"]
    ):
        return True

    return False


# ============================================================
# SQLITE
# ============================================================

def connect_db():
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except PermissionError:
            print(
                f"WARNING: Could not delete {DB_FILE}. "
                "Close programs using it and run again."
            )
            raise

    conn = sqlite3.connect(DB_FILE)

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA cache_size = -200000")

    conn.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT,
            business_name TEXT,
            business_address TEXT,
            name_norm TEXT,
            name_canon TEXT,
            address_norm TEXT,
            address_canon TEXT,
            postal TEXT,
            prefix6 TEXT,
            prefix4 TEXT,
            suffix4 TEXT,
            first4 TEXT,
            last4 TEXT
        )
    """)

    return conn


def insert_source(conn, path, source):
    print(f"\nLoading {source}: {os.path.basename(path)}")

    insert_sql = """
        INSERT INTO records (
            entity_id, source, country,
            business_name, business_address,
            name_norm, name_canon,
            address_norm, address_canon,
            postal,
            prefix6, prefix4, suffix4, first4, last4
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    total = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        fields = reader.fieldnames or []

        required = {
            "entity_id",
            "business_name",
            "business_address",
            "country",
        }

        missing = required - set(fields)

        if missing:
            raise ValueError(
                f"{path} missing columns: {sorted(missing)}"
            )

        batch = []

        for row in reader:
            eid = str(row["entity_id"]).strip()
            name = str(row["business_name"] or "")
            address = str(row["business_address"] or "")
            country = country_norm(row["country"])

            nn = clean_text(name)
            nc = canonical_name(name)
            an = clean_text(address)
            ac = canonical_address(address)

            item = (
                eid,
                source,
                country,
                name,
                address,
                nn,
                nc,
                an,
                ac,
                postal_code(address),
                prefix(nc, 6),
                prefix(nc, 4),
                suffix(nc, 4),
                first4(nc),
                last4(nc),
            )

            batch.append(item)

            if len(batch) >= 10_000:
                conn.executemany(insert_sql, batch)
                batch.clear()

            total += 1

            if total % 1_000_000 == 0:
                conn.commit()
                print(f"  inserted ~{total:,}")

        if batch:
            conn.executemany(insert_sql, batch)

    conn.commit()

    print(f"Finished {source}: {total:,}")


def create_indexes(conn):
    print("\nCreating indexes...")

    columns = [
        "name_norm",
        "name_canon",
        "address_norm",
        "address_canon",
        "prefix6",
        "prefix4",
        "suffix4",
        "first4",
        "last4",
        "postal",
    ]

    for i, col in enumerate(columns, 1):
        print(f"  {i}/10: {col}")

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{col} "
            f"ON records(source, {col})"
        )

    conn.commit()


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_sample():
    rows = []

    with open(S1_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            name = str(row["business_name"] or "")
            address = str(row["business_address"] or "")

            nc = canonical_name(name)
            ac = canonical_address(address)

            rows.append({
                "entity_id": str(row["entity_id"]).strip(),
                "country": country_norm(row["country"]),
                "business_name": name,
                "business_address": address,
                "name_norm": clean_text(name),
                "name_canon": nc,
                "address_norm": clean_text(address),
                "address_canon": ac,
                "postal": postal_code(address),
                "prefix6": prefix(nc, 6),
                "prefix4": prefix(nc, 4),
                "suffix4": suffix(nc, 4),
                "first4": first4(nc),
                "last4": last4(nc),
            })

            if len(rows) >= S1_SAMPLE_SIZE:
                break

    return rows


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():
    gt = {}

    with open(GT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = str(row["source1_entity_id"]).strip()

            value = str(
                row.get("matched_entity_ids", "") or ""
            ).strip()

            if value:
                ids = {
                    x.strip()
                    for x in value.split(",")
                    if x.strip()
                }
            else:
                ids = set()

            gt[s1_id] = ids

    return gt


# ============================================================
# CANDIDATE COLLECTION
# ============================================================

BLOCKS = [
    ("name_norm", "name_norm"),
    ("name_canon", "name_canon"),
    ("address_norm", "address_norm"),
    ("name_addr", None),
    ("prefix6", "prefix6"),
    ("prefix4", "prefix4"),
    ("suffix4", "suffix4"),
    ("first4", "first4"),
    ("last4", "last4"),
    ("postal", "postal"),
]


def collect_candidates(conn, s1):
    """
    Return candidate entity IDs from BOTH S2 and S3.

    Candidates are collected in deterministic order based on the
    blocking-pass order and then capped at 2,000.
    """
    candidates = []

    seen = set()

    def add_rows(sql, params):
        cur = conn.execute(sql, params)

        for row in cur:
            eid = row[0]

            if eid not in seen:
                seen.add(eid)
                candidates.append(eid)

                if len(candidates) >= CAP:
                    return True

        return False

    # 1. Exact normalized name
    if s1["name_norm"]:
        if add_rows(
            """
            SELECT entity_id
            FROM records
            WHERE source IN ('S2','S3')
              AND name_norm = ?
            """,
            (s1["name_norm"],),
        ):
            return candidates

    # 2. Canonical name
    if s1["name_canon"]:
        if add_rows(
            """
            SELECT entity_id
            FROM records
            WHERE source IN ('S2','S3')
              AND name_canon = ?
            """,
            (s1["name_canon"],),
        ):
            return candidates

    # 3. Exact normalized address
    if s1["address_norm"]:
        if add_rows(
            """
            SELECT entity_id
            FROM records
            WHERE source IN ('S2','S3')
              AND address_norm = ?
            """,
            (s1["address_norm"],),
        ):
            return candidates

    # 4. Canonical name + canonical address
    if s1["name_canon"] and s1["address_canon"]:
        if add_rows(
            """
            SELECT entity_id
            FROM records
            WHERE source IN ('S2','S3')
              AND name_canon = ?
              AND address_canon = ?
            """,
            (s1["name_canon"], s1["address_canon"]),
        ):
            return candidates

    # 5-9. Name signatures
    for col, value in [
        ("prefix6", s1["prefix6"]),
        ("prefix4", s1["prefix4"]),
        ("suffix4", s1["suffix4"]),
        ("first4", s1["first4"]),
        ("last4", s1["last4"]),
    ]:
        if value:
            if add_rows(
                f"""
                SELECT entity_id
                FROM records
                WHERE source IN ('S2','S3')
                  AND {col} = ?
                """,
                (value,),
            ):
                return candidates

    # 10. Postal
    if s1["postal"]:
        if add_rows(
            """
            SELECT entity_id
            FROM records
            WHERE source IN ('S2','S3')
              AND postal = ?
            """,
            (s1["postal"],),
        ):
            return candidates

    return candidates


# ============================================================
# FETCH CANDIDATE RECORDS
# ============================================================

def fetch_candidates(conn, ids):
    if not ids:
        return {}

    result = {}

    # SQLite parameter limit is normally ~999, so use chunks.
    for start in range(0, len(ids), 500):
        part = ids[start:start + 500]

        placeholders = ",".join(["?"] * len(part))

        sql = f"""
            SELECT
                entity_id,
                source,
                country,
                business_name,
                business_address,
                name_norm,
                name_canon,
                address_norm,
                address_canon,
                postal
            FROM records
            WHERE entity_id IN ({placeholders})
        """

        for row in conn.execute(sql, part):
            result[row[0]] = {
                "entity_id": row[0],
                "source": row[1],
                "country": row[2],
                "business_name": row[3],
                "business_address": row[4],
                "name_norm": row[5],
                "name_canon": row[6],
                "address_norm": row[7],
                "address_canon": row[8],
                "postal": row[9],
            }

    return result


# ============================================================
# F0.5
# ============================================================

def f05(true_set, pred_set):
    true_set = set(true_set)
    pred_set = set(pred_set)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta = 0.5
    b2 = beta * beta

    return (
        (1 + b2) * precision * recall
        / (b2 * precision + recall)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("CANDIDATE GENERATION + BASELINE MATCHING + MACRO F0.5")
    print("=" * 70)

    for path in [S1_FILE, S2_FILE, S3_FILE, GT_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    conn = connect_db()

    try:
        insert_source(conn, S2_FILE, "S2")
        insert_source(conn, S3_FILE, "S3")
        create_indexes(conn)

        s1_rows = load_s1_sample()
        gt = load_ground_truth()

        print(f"\nS1 sample: {len(s1_rows):,}")
        print(f"Candidate cap: {CAP:,}")

        # ----------------------------------------------------
        # Collect candidates
        # ----------------------------------------------------

        all_candidates = {}
        total_candidates = 0
        total_true = 0
        retrieved_true = 0

        print("\nCollecting candidate pools...")

        for i, s1 in enumerate(s1_rows, 1):
            ids = collect_candidates(conn, s1)

            all_candidates[s1["entity_id"]] = ids
            total_candidates += len(ids)

            true_ids = gt.get(s1["entity_id"], set())

            total_true += len(true_ids)
            retrieved_true += len(true_ids & set(ids))

            if i % 1000 == 0 or i == len(s1_rows):
                elapsed = (time.time() - start_time) / 60
                avg = total_candidates / i

                print(
                    f"  {i:,}/{len(s1_rows):,} "
                    f"| avg={avg:.1f} "
                    f"| time={elapsed:.1f} min"
                )

        blocking_recall = (
            retrieved_true / total_true
            if total_true
            else 0.0
        )

        avg_candidates = (
            total_candidates / len(s1_rows)
            if s1_rows
            else 0.0
        )

        # ----------------------------------------------------
        # Save actual candidate_pairs.tsv
        # ----------------------------------------------------

        print("\nSaving candidate_pairs.tsv...")

        with open(
            CANDIDATE_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n"
            )

            writer.writerow([
                "source1_entity_id",
                "candidate_entity_id",
            ])

            for s1_id, ids in all_candidates.items():
                for candidate_id in ids:
                    writer.writerow([
                        s1_id,
                        candidate_id,
                    ])

        # ----------------------------------------------------
        # Build predictions
        # ----------------------------------------------------

        print("\nCalculating candidate similarities...")
        print("Creating FINAL baseline predictions:")
        print("S1 -> S2 and S3")

        predictions = {}
        detail_rows = []

        candidate_pair_count = 0
        predicted_pair_count = 0

        for i, s1 in enumerate(s1_rows, 1):
            ids = all_candidates.get(s1["entity_id"], [])

            candidates = fetch_candidates(conn, ids)

            predicted = set()

            for candidate_id in ids:
                candidate = candidates.get(candidate_id)

                if candidate is None:
                    continue

                features = similarity_features(s1, candidate)

                if is_prediction(features):
                    predicted.add(candidate_id)

                candidate_pair_count += 1

            predictions[s1["entity_id"]] = predicted
            predicted_pair_count += len(predicted)

            if i % 1000 == 0 or i == len(s1_rows):
                print(
                    f"  scored {i:,}/{len(s1_rows):,} "
                    f"| predicted pairs={predicted_pair_count:,}"
                )

        # ----------------------------------------------------
        # Save predictions.tsv
        # ----------------------------------------------------

        print("\nSaving predictions.tsv...")

        with open(
            PREDICTION_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n"
            )

            writer.writerow([
                "source1_entity_id",
                "matched_entity_ids",
            ])

            for s1 in s1_rows:
                s1_id = s1["entity_id"]
                pred = sorted(predictions.get(s1_id, set()))

                writer.writerow([
                    s1_id,
                    ",".join(pred),
                ])

        # ----------------------------------------------------
        # Evaluate Macro F0.5
        # ----------------------------------------------------

        macro_sum = 0.0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        exact_matches = 0

        for s1 in s1_rows:
            s1_id = s1["entity_id"]

            true_set = set(gt.get(s1_id, set()))
            pred_set = set(predictions.get(s1_id, set()))

            # Only legal S2/S3 IDs should be in predictions.
            tp_set = true_set & pred_set
            fp_set = pred_set - true_set
            fn_set = true_set - pred_set

            tp = len(tp_set)
            fp = len(fp_set)
            fn = len(fn_set)

            score = f05(true_set, pred_set)

            macro_sum += score

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if true_set == pred_set:
                exact_matches += 1

            detail_rows.append([
                s1_id,
                ",".join(sorted(true_set)),
                ",".join(sorted(pred_set)),
                tp,
                fp,
                fn,
                f"{score:.8f}",
            ])

        n = len(s1_rows)

        macro_f05 = macro_sum / n if n else 0.0

        global_precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp
            else 1.0
        )

        global_recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn
            else 1.0
        )

        beta = 0.5
        b2 = beta * beta

        global_f05 = (
            (1 + b2)
            * global_precision
            * global_recall
            / (b2 * global_precision + global_recall)
            if global_precision + global_recall
            else 0.0
        )

        # ----------------------------------------------------
        # Save matching_results.tsv
        # ----------------------------------------------------

        print("\nSaving matching_results.tsv...")

        with open(
            RESULT_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n"
            )

            writer.writerow([
                "source1_entity_id",
                "matched_entity_ids",
            ])

            for s1 in s1_rows:
                s1_id = s1["entity_id"]

                pred = sorted(
                    predictions.get(s1_id, set())
                )

                writer.writerow([
                    s1_id,
                    ",".join(pred),
                ])

        # ----------------------------------------------------
        # Save detailed F0.5
        # ----------------------------------------------------

        with open(
            DETAIL_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(
                f,
                delimiter="\t",
                lineterminator="\n"
            )

            writer.writerow([
                "source1_entity_id",
                "true_matched_entity_ids",
                "predicted_matched_entity_ids",
                "TP",
                "FP",
                "FN",
                "F0.5",
            ])

            writer.writerows(detail_rows)

        # ----------------------------------------------------
        # Save cap result
        # ----------------------------------------------------

        with open(
            CAP_RESULT_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "cap",
                "blocking_recall",
                "avg_candidates_per_s1",
                "macro_f05_baseline",
            ])

            writer.writerow([
                CAP,
                f"{blocking_recall:.8f}",
                f"{avg_candidates:.4f}",
                f"{macro_f05:.8f}",
            ])

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        elapsed = (time.time() - start_time) / 60

        summary = f"""
AMAZON ML CHALLENGE 2026
CANDIDATE + BASELINE MATCHING RESULT
=====================================

Development S1 rows:
{len(s1_rows):,}

Source 2 rows:
5,034,616

Source 3 rows:
5,285,603

Candidate cap:
{CAP:,}

Total candidate pairs:
{candidate_pair_count:,}

Average candidates per S1:
{avg_candidates:.4f}

True match pairs:
{total_true:,}

True matches retrieved by blocking:
{retrieved_true:,}

Blocking recall:
{blocking_recall:.8f}

Baseline predicted pairs:
{predicted_pair_count:,}

TP:
{total_tp:,}

FP:
{total_fp:,}

FN:
{total_fn:,}

Global precision:
{global_precision:.8f}

Global recall:
{global_recall:.8f}

Global F0.5 (diagnostic):
{global_f05:.8f}

Exact S1 set matches:
{exact_matches:,}

=====================================
MACRO F0.5:
{macro_f05:.8f}
=====================================

IMPORTANT:
This Macro F0.5 is a DEVELOPMENT score on the first
{len(s1_rows):,} Source 1 records.

The baseline prediction was created from candidate similarity rules.
It is NOT yet the final LightGBM/XGBoost model.

Files:
{CANDIDATE_FILE}
{PREDICTION_FILE}
{RESULT_FILE}
{DETAIL_FILE}
{SUMMARY_FILE}
{CAP_RESULT_FILE}
"""

        with open(
            SUMMARY_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(summary.strip() + "\n")

        # ----------------------------------------------------
        # FINAL TERMINAL OUTPUT
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)

        print(f"S1 sample                  : {len(s1_rows):,}")
        print(f"True match pairs            : {total_true:,}")
        print(f"Retrieved true matches      : {retrieved_true:,}")
        print(f"Blocking recall             : {blocking_recall:.2%}")
        print(f"Average candidates / S1     : {avg_candidates:.1f}")

        print("\nBASELINE MATCHING")
        print(f"Predicted pairs             : {predicted_pair_count:,}")
        print(f"TP                          : {total_tp:,}")
        print(f"FP                          : {total_fp:,}")
        print(f"FN                          : {total_fn:,}")
        print(f"Precision                   : {global_precision:.6f}")
        print(f"Recall                      : {global_recall:.6f}")
        print(f"Global F0.5                 : {global_f05:.6f}")

        print("\n" + "*" * 70)
        print(f"MACRO F0.5                  : {macro_f05:.6f}")
        print("*" * 70)

        print("\nOUTPUT FILES")
        print(f"Candidate pairs : {CANDIDATE_FILE}")
        print(f"Predictions     : {PREDICTION_FILE}")
        print(f"Final results   : {RESULT_FILE}")
        print(f"F0.5 details    : {DETAIL_FILE}")
        print(f"Summary         : {SUMMARY_FILE}")

        print(f"\nTotal runtime: {elapsed:.2f} minutes")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
