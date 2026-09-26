#!/usr/bin/env python3
"""
Amazon ML Challenge 2026
Improved blocking development evaluator

Purpose
-------
This script is the NEXT STEP after the 31.42% baseline blocking result.

It:
1. Reads the first S1_SAMPLE_SIZE S1 rows (default 50,000).
2. Keeps S2/S3 on disk in SQLite, so S2/S3 are NOT loaded into RAM.
3. Creates several blocking keys:
   - exact normalized name
   - canonical name
   - exact normalized address
   - canonical name + address
   - name prefix
   - name suffix
   - first/last informative token
   - postal code (when available)
4. Produces the UNION of candidates from all blocking passes.
5. Evaluates blocking recall against train ground truth for the S1 sample.
6. Saves candidate_pairs_dev.tsv and blocking_dev_results.csv.

IMPORTANT
---------
This is a DEVELOPMENT evaluator. It intentionally uses only the first
50,000 S1 rows so that you can iterate without waiting ~2 hours each time.

It does NOT train LightGBM yet. We first want high-recall candidates.
"""

import csv
import os
import re
import sqlite3
import time
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change this if your script is stored elsewhere.
DATA_DIR = os.path.join(BASE_DIR, "dataset", "train")

S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

S1_SAMPLE_SIZE = 50_000

# Maximum candidates retained for each S1 from each broad blocking pass.
# Exact blocks are kept without this cap unless they become extremely large.
MAX_CANDIDATES_PER_PASS = 300

# Final maximum candidates per S1. Increase later if recall is too low.
MAX_FINAL_CANDIDATES = 500

DB_FILE = os.path.join(BASE_DIR, "blocking_dev.sqlite")

OUTPUT_CANDIDATES = os.path.join(BASE_DIR, "candidate_pairs_dev.tsv")
OUTPUT_RESULTS = os.path.join(BASE_DIR, "blocking_dev_results.csv")

# ============================================================
# NORMALIZATION
# ============================================================

ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    "llc": "llc",
    "llp": "llp",
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

COMMON_STOPWORDS = {
    "the", "and", "of", "for", "in", "at", "on",
    "private", "limited", "incorporated", "corporation",
    "company", "llc", "llp"
}


def clean_text(value):
    if value is None:
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value):
    text = clean_text(value)
    if not text:
        return ""

    tokens = []
    for token in text.split():
        tokens.append(ABBREVIATIONS.get(token, token))

    return " ".join(tokens)


def canonical_name(value):
    text = normalize_text(value)
    if not text:
        return ""

    # Remove legal suffixes and very common words.
    tokens = [
        t for t in text.split()
        if t not in {
            "private", "limited", "incorporated", "corporation",
            "company", "llc", "llp"
        }
    ]
    return " ".join(tokens)


def alnum_compact(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_postal(value):
    """
    Extract a postal/ZIP-like number.
    Works for Indian PINs and common international postal patterns.
    """
    if not value:
        return ""
    text = str(value)

    # Prefer 5- or 6-digit numeric postal codes.
    matches = re.findall(r"\b\d{5,6}\b", text)
    if matches:
        return matches[-1]

    return ""


def informative_tokens(name):
    text = normalize_text(name)
    tokens = []

    for token in text.split():
        if len(token) >= 3 and token not in COMMON_STOPWORDS:
            tokens.append(token)

    return tokens


def make_keys(name, address):
    n = normalize_text(name)
    c = canonical_name(name)
    a = normalize_text(address)

    compact = alnum_compact(c or n)

    toks = informative_tokens(name)
    first_tok = toks[0] if toks else ""
    last_tok = toks[-1] if toks else ""

    return {
        "name_norm": n,
        "name_canon": c,
        "address_norm": a,
        "name_addr": (c + " | " + a) if c and a else "",
        "name_prefix4": compact[:4] if len(compact) >= 4 else compact,
        "name_prefix6": compact[:6] if len(compact) >= 6 else compact,
        "name_suffix4": compact[-4:] if len(compact) >= 4 else compact,
        "first_token4": first_tok[:4],
        "last_token4": last_tok[:4],
        "postal": get_postal(address),
    }


# ============================================================
# TSV HELPERS
# ============================================================

def detect_column(fieldnames, aliases, required=True):
    lower = {f.lower().strip(): f for f in fieldnames}

    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]

    # Partial match fallback.
    for f in fieldnames:
        fl = f.lower().strip()
        for alias in aliases:
            if alias.lower() in fl:
                return f

    if required:
        raise ValueError(
            f"Could not find a column matching {aliases}. "
            f"Available columns: {fieldnames}"
        )
    return None


def inspect_header(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
    return header


# ============================================================
# SQLITE DATABASE
# ============================================================

def create_database():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-200000")  # ~200 MB cache target

    conn.execute("""
        CREATE TABLE source_records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            name TEXT,
            address TEXT,
            country TEXT,
            name_norm TEXT,
            name_canon TEXT,
            address_norm TEXT,
            name_addr TEXT,
            name_prefix4 TEXT,
            name_prefix6 TEXT,
            name_suffix4 TEXT,
            first_token4 TEXT,
            last_token4 TEXT,
            postal TEXT
        )
    """)

    return conn


def load_source(conn, path, source_name):
    print(f"\nLoading {source_name}: {path}")

    header = inspect_header(path)

    id_col = detect_column(
        header,
        ["entity_id", "id", "business_id", "record_id"]
    )
    name_col = detect_column(
        header,
        ["name", "business_name", "business name", "company_name"]
    )
    address_col = detect_column(
        header,
        ["address", "business_address", "business address"],
        required=False
    )
    country_col = detect_column(
        header,
        ["country", "country_code", "country code"],
        required=False
    )

    print("  ID column:", id_col)
    print("  Name column:", name_col)
    print("  Address column:", address_col)
    print("  Country column:", country_col)

    sql = """
        INSERT OR REPLACE INTO source_records (
            entity_id, source, name, address, country,
            name_norm, name_canon, address_norm, name_addr,
            name_prefix4, name_prefix6, name_suffix4,
            first_token4, last_token4, postal
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    count = 0

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            entity_id = str(row.get(id_col, "")).strip()
            name = str(row.get(name_col, "") or "")
            address = str(row.get(address_col, "") or "") if address_col else ""
            country = str(row.get(country_col, "") or "").strip().lower() if country_col else ""

            if not entity_id:
                continue

            keys = make_keys(name, address)

            batch.append((
                entity_id,
                source_name,
                name,
                address,
                country,
                keys["name_norm"],
                keys["name_canon"],
                keys["address_norm"],
                keys["name_addr"],
                keys["name_prefix4"],
                keys["name_prefix6"],
                keys["name_suffix4"],
                keys["first_token4"],
                keys["last_token4"],
                keys["postal"],
            ))

            if len(batch) >= 10_000:
                conn.executemany(sql, batch)
                batch.clear()
                count += 10_000

                if count % 500_000 == 0:
                    print(f"  inserted approximately {count:,} rows")

    if batch:
        conn.executemany(sql, batch)
        count += len(batch)

    conn.commit()
    print(f"Finished {source_name}: {count:,} rows")


def create_indexes(conn):
    print("\nCreating SQLite indexes...")

    indexes = [
        ("idx_name_norm", "name_norm"),
        ("idx_name_canon", "name_canon"),
        ("idx_address_norm", "address_norm"),
        ("idx_name_addr", "name_addr"),
        ("idx_prefix4", "name_prefix4"),
        ("idx_prefix6", "name_prefix6"),
        ("idx_suffix4", "name_suffix4"),
        ("idx_first_token4", "first_token4"),
        ("idx_last_token4", "last_token4"),
        ("idx_postal", "postal"),
    ]

    for i, (idx_name, column) in enumerate(indexes, 1):
        print(f"  {i}/{len(indexes)}: {column}")
        conn.execute(
            f"CREATE INDEX {idx_name} ON source_records({column})"
        )

    # Country is mainly useful as an additional filter.
    conn.execute(
        "CREATE INDEX idx_country ON source_records(country)"
    )

    conn.commit()
    print("Indexes created.")


# ============================================================
# LOAD S1 SAMPLE + GROUND TRUTH
# ============================================================

def load_s1_sample(path, limit):
    print(f"\nLoading first {limit:,} S1 records...")

    header = inspect_header(path)

    id_col = detect_column(
        header,
        ["entity_id", "id", "business_id", "record_id"]
    )
    name_col = detect_column(
        header,
        ["name", "business_name", "business name", "company_name"]
    )
    address_col = detect_column(
        header,
        ["address", "business_address", "business address"],
        required=False
    )
    country_col = detect_column(
        header,
        ["country", "country_code", "country code"],
        required=False
    )

    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if i >= limit:
                break

            entity_id = str(row.get(id_col, "")).strip()
            name = str(row.get(name_col, "") or "")
            address = str(row.get(address_col, "") or "") if address_col else ""
            country = str(row.get(country_col, "") or "").strip().lower() if country_col else ""

            if not entity_id:
                continue

            rows.append({
                "entity_id": entity_id,
                "name": name,
                "address": address,
                "country": country,
                "keys": make_keys(name, address),
            })

    print(f"S1 sample loaded: {len(rows):,}")
    return rows


def load_ground_truth_sample(path, sample_ids):
    """
    Returns:
        {s1_id: set(matched_external_ids)}
    """
    print("\nLoading ground truth for S1 sample...")

    gt = {sid: set() for sid in sample_ids}

    header = inspect_header(path)

    id_col = detect_column(
        header,
        ["s1_entity_id", "s1_id", "source1_entity_id", "entity_id", "id"]
    )

    match_col = detect_column(
        header,
        [
            "matched_entity_ids",
            "matched_ids",
            "matches",
            "matched_entity_id",
            "source2_source3_entity_ids",
        ]
    )

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            sid = str(row.get(id_col, "")).strip()

            if sid not in gt:
                continue

            value = str(row.get(match_col, "") or "").strip()

            if not value:
                continue

            # Ground truth is comma-separated according to the challenge.
            # Also accept semicolon-separated defensively.
            parts = re.split(r"[,;]", value)

            for p in parts:
                p = p.strip()
                if p:
                    gt[sid].add(p)

    total_true = sum(len(v) for v in gt.values())
    singletons = sum(1 for v in gt.values() if len(v) == 0)

    print(f"Ground-truth S1 sample rows: {len(gt):,}")
    print(f"S1 no-match/singleton rows: {singletons:,}")
    print(f"True match pairs in sample: {total_true:,}")

    return gt


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def query_ids(conn, column, value, country=None, max_rows=300):
    if not value:
        return []

    if country:
        sql = f"""
            SELECT entity_id
            FROM source_records
            WHERE {column} = ?
              AND (country = ? OR country = '' OR ? = '')
            LIMIT ?
        """
        rows = conn.execute(
            sql, (value, country, country, max_rows)
        ).fetchall()
    else:
        sql = f"""
            SELECT entity_id
            FROM source_records
            WHERE {column} = ?
            LIMIT ?
        """
        rows = conn.execute(sql, (value, max_rows)).fetchall()

    return [r[0] for r in rows]


def generate_candidates_for_s1(conn, row):
    keys = row["keys"]
    country = row["country"]

    candidates = set()
    pass_counts = defaultdict(int)

    passes = [
        ("B1_exact_name", "name_norm", keys["name_norm"]),
        ("B2_canonical_name", "name_canon", keys["name_canon"]),
        ("B3_exact_address", "address_norm", keys["address_norm"]),
        ("B4_name_plus_address", "name_addr", keys["name_addr"]),
        ("B5_name_prefix4", "name_prefix4", keys["name_prefix4"]),
        ("B6_name_prefix6", "name_prefix6", keys["name_prefix6"]),
        ("B7_name_suffix4", "name_suffix4", keys["name_suffix4"]),
        ("B8_first_token4", "first_token4", keys["first_token4"]),
        ("B9_last_token4", "last_token4", keys["last_token4"]),
        ("B10_postal", "postal", keys["postal"]),
    ]

    for pass_name, column, value in passes:
        if not value:
            continue

        ids = query_ids(
            conn,
            column,
            value,
            country=country,
            max_rows=MAX_CANDIDATES_PER_PASS
        )

        before = len(candidates)
        candidates.update(ids)
        pass_counts[pass_name] = len(candidates) - before

    # Never allow one broad key to make the final candidate set enormous.
    # Keep the union if it is small; otherwise prioritize exact/canonical
    # candidates and then the deterministic prefix/token candidates.
    if len(candidates) > MAX_FINAL_CANDIDATES:
        priority = []

        for pass_name, column, value in passes:
            if not value:
                continue

            ids = query_ids(
                conn,
                column,
                value,
                country=country,
                max_rows=MAX_CANDIDATES_PER_PASS
            )

            for eid in ids:
                priority.append(eid)

        # Ordered de-duplication.
        selected = []
        seen = set()

        for eid in priority:
            if eid not in seen:
                seen.add(eid)
                selected.append(eid)

            if len(selected) >= MAX_FINAL_CANDIDATES:
                break

        candidates = set(selected)

    return candidates, pass_counts


# ============================================================
# EVALUATION
# ============================================================

def evaluate(conn, s1_rows, ground_truth):
    print("\n" + "=" * 70)
    print("IMPROVED MULTI-PASS BLOCKING DEVELOPMENT EVALUATION")
    print("=" * 70)

    all_candidates = {}
    pass_retrieved = defaultdict(int)
    pass_candidate_count = defaultdict(int)

    total_true = 0
    retrieved_true = 0

    start = time.time()

    for i, row in enumerate(s1_rows, 1):
        sid = row["entity_id"]

        candidates, pass_counts = generate_candidates_for_s1(conn, row)

        all_candidates[sid] = candidates

        true_ids = ground_truth.get(sid, set())
        total_true += len(true_ids)

        retrieved_true += len(true_ids.intersection(candidates))

        for p, count in pass_counts.items():
            pass_candidate_count[p] += count

        if i % 5_000 == 0:
            elapsed = time.time() - start
            avg = sum(len(v) for v in all_candidates.values()) / len(all_candidates)
            print(
                f"Processed {i:,}/{len(s1_rows):,} | "
                f"avg candidates/S1={avg:.1f} | "
                f"time={elapsed/60:.1f} min"
            )

    # Overall union recall.
    recall = (
        retrieved_true / total_true
        if total_true
        else 0.0
    )

    avg_candidates = (
        sum(len(v) for v in all_candidates.values()) / len(all_candidates)
        if all_candidates
        else 0.0
    )

    max_candidates = (
        max((len(v) for v in all_candidates.values()), default=0)
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)
    print(f"S1 sample:                 {len(s1_rows):,}")
    print(f"True match pairs:          {total_true:,}")
    print(f"Retrieved true matches:    {retrieved_true:,}")
    print(f"UNION blocking recall:     {recall:.2%}")
    print(f"Average candidates / S1:   {avg_candidates:.2f}")
    print(f"Maximum candidates / S1:   {max_candidates:,}")

    # Candidate-count distribution.
    counts = [len(v) for v in all_candidates.values()]
    counts.sort()

    if counts:
        p50 = counts[int(0.50 * (len(counts) - 1))]
        p90 = counts[int(0.90 * (len(counts) - 1))]
        p95 = counts[int(0.95 * (len(counts) - 1))]
        p99 = counts[int(0.99 * (len(counts) - 1))]

        print(f"Candidate count P50:       {p50:,}")
        print(f"Candidate count P90:       {p90:,}")
        print(f"Candidate count P95:       {p95:,}")
        print(f"Candidate count P99:       {p99:,}")

    return all_candidates, {
        "s1_sample": len(s1_rows),
        "true_pairs": total_true,
        "retrieved_true": retrieved_true,
        "blocking_recall": recall,
        "avg_candidates_per_s1": avg_candidates,
        "max_candidates_per_s1": max_candidates,
    }


def save_candidates(s1_rows, all_candidates):
    print(f"\nSaving candidate pairs: {OUTPUT_CANDIDATES}")

    with open(
        OUTPUT_CANDIDATES,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["s1_entity_id", "candidate_entity_id"])

        for row in s1_rows:
            sid = row["entity_id"]

            for eid in sorted(all_candidates.get(sid, set())):
                writer.writerow([sid, eid])

    print("Candidate pairs saved.")


def save_results(metrics):
    fields = list(metrics.keys())

    with open(
        OUTPUT_RESULTS,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Metrics saved: {OUTPUT_RESULTS}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("IMPROVED BLOCKING - 50K DEVELOPMENT RUN")
    print("=" * 70)

    print("\nData directory:")
    print(DATA_DIR)

    for path in [S1_FILE, S2_FILE, S3_FILE, GT_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nFile not found:\n{path}\n\n"
                f"Expected files:\n"
                f"  {S1_FILE}\n"
                f"  {S2_FILE}\n"
                f"  {S3_FILE}\n"
                f"  {GT_FILE}\n"
            )

    total_start = time.time()

    # 1. Create disk-backed DB.
    conn = create_database()

    # 2. Load only S2/S3 into SQLite.
    load_source(conn, S2_FILE, "S2")
    load_source(conn, S3_FILE, "S3")

    # 3. Build indexes.
    create_indexes(conn)

    # 4. Load first 50k S1.
    s1_rows = load_s1_sample(S1_FILE, S1_SAMPLE_SIZE)

    # 5. Ground truth for those S1 rows only.
    sample_ids = {r["entity_id"] for r in s1_rows}
    ground_truth = load_ground_truth_sample(GT_FILE, sample_ids)

    # 6. Evaluate blocking.
    all_candidates, metrics = evaluate(
        conn,
        s1_rows,
        ground_truth
    )

    # 7. Save candidate pairs.
    save_candidates(s1_rows, all_candidates)

    # 8. Save metrics.
    save_results(metrics)

    conn.close()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Runtime: {(time.time() - total_start)/60:.2f} minutes")
    print(f"Candidate file: {OUTPUT_CANDIDATES}")
    print(f"Metrics file:   {OUTPUT_RESULTS}")
    print("\nNEXT STEP:")
    print("If recall is substantially better than 31.42%,")
    print("we will use this blocking design for the full dataset.")
    print("Do NOT train LightGBM yet.")
    print("=" * 70)


if __name__ == "__main__":
    main()
