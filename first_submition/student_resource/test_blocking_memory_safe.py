import os
import re
import sqlite3
import unicodedata
from time import time

import pandas as pd


# ============================================================
# Amazon ML Challenge 2026
# MEMORY-SAFE MULTI-PASS BLOCKING EVALUATION
#
# The previous dictionary-based version caused MemoryError because
# it attempted to keep millions of records in several Python
# dictionaries/lists.
#
# This version uses SQLite indexes on disk. SQLite is included
# with Python, so no extra package is required.
#
# It evaluates:
#   B1 = country + normalized name
#   B2 = country + canonical name
#   B3 = country + normalized address
#   B4 = country + canonical name + canonical address
#
# It evaluates TRUE-MATCH RECALL without constructing the enormous
# candidate sets in Python RAM.
#
# This is a blocking-evaluation script only. It does not train
# LightGBM/XGBoost and does not create the final submission.
# ============================================================


TRAIN_DIR = "dataset/train"

S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_FILE = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

CHUNK_SIZE = 100_000

# Temporary SQLite database. It is deleted after the run.
DB_FILE = "amazon_blocking_eval.sqlite"


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


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = unicodedata.normalize("NFKC", value)

    # Keep letters/numbers/spaces. This is deliberately conservative.
    value = re.sub(r"[^\w\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    return clean_text(value)


def canonical_name(value):
    value = clean_text(value)

    if not value:
        return ""

    output = []

    for token in value.split():
        replacement = NAME_ABBREVIATIONS.get(token, token)
        output.extend(replacement.split())

    return " ".join(output)


def normalize_address(value):
    return clean_text(value)


def canonical_address(value):
    value = clean_text(value)

    if not value:
        return ""

    output = []

    for token in value.split():
        replacement = ADDRESS_ABBREVIATIONS.get(token, token)
        output.extend(replacement.split())

    return " ".join(output)


def connect_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)

    # Performance settings for a temporary rebuildable database.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=FILE;")
    conn.execute("PRAGMA cache_size=-200000;")

    return conn


def create_tables(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS records;
        DROP TABLE IF EXISTS truth_pairs;

        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT,
            name_norm TEXT,
            name_canon TEXT,
            address_norm TEXT,
            address_canon TEXT
        );

        CREATE TABLE truth_pairs (
            source1_entity_id TEXT NOT NULL,
            matched_entity_id TEXT NOT NULL
        );
        """
    )
    conn.commit()


def insert_source_file(conn, path, source):
    print(f"\nLoading {source}: {path}")

    total = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        rows = []

        for row in chunk.itertuples(index=False):
            rows.append(
                (
                    row.entity_id,
                    source,
                    clean_text(row.country),
                    normalize_name(row.business_name),
                    canonical_name(row.business_name),
                    normalize_address(row.business_address),
                    canonical_address(row.business_address),
                )
            )

        conn.executemany(
            """
            INSERT INTO records (
                entity_id,
                source,
                country,
                name_norm,
                name_canon,
                address_norm,
                address_canon
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        total += len(rows)

        if total % 500_000 < CHUNK_SIZE:
            print(f"  loaded {total:,} rows")

    conn.commit()
    print(f"Finished {source}: {total:,} rows")


def insert_ground_truth(conn):
    print("\nLoading ground truth...")

    total_s1 = 0
    total_true = 0
    singletons = 0

    for chunk in pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        rows = []

        for row in chunk.itertuples(index=False):
            total_s1 += 1

            value = row.matched_entity_ids

            if not value or not str(value).strip():
                singletons += 1
                continue

            for matched_id in str(value).split(","):
                matched_id = matched_id.strip()

                if matched_id:
                    rows.append(
                        (
                            row.source1_entity_id,
                            matched_id,
                        )
                    )
                    total_true += 1

        if rows:
            conn.executemany(
                """
                INSERT INTO truth_pairs (
                    source1_entity_id,
                    matched_entity_id
                )
                VALUES (?, ?)
                """,
                rows,
            )

        if total_s1 % 500_000 < CHUNK_SIZE:
            print(
                f"  processed {total_s1:,} ground-truth S1 rows"
            )

    conn.commit()

    print(f"Ground-truth S1 rows: {total_s1:,}")
    print(f"Singletons: {singletons:,}")
    print(f"True match pairs: {total_true:,}")


def create_indexes(conn):
    print("\nCreating SQLite indexes on disk...")
    print("This avoids the previous Python MemoryError.")

    statements = [
        """
        CREATE INDEX idx_records_name_norm
        ON records(source, country, name_norm)
        """,
        """
        CREATE INDEX idx_records_name_canon
        ON records(source, country, name_canon)
        """,
        """
        CREATE INDEX idx_records_address_norm
        ON records(source, country, address_norm)
        """,
        """
        CREATE INDEX idx_records_name_address
        ON records(
            source,
            country,
            name_canon,
            address_canon
        )
        """,
        """
        CREATE INDEX idx_truth_s1
        ON truth_pairs(source1_entity_id)
        """,
        """
        CREATE INDEX idx_truth_match
        ON truth_pairs(matched_entity_id)
        """,
    ]

    for i, sql in enumerate(statements, start=1):
        print(f"  Creating index {i}/{len(statements)}...")
        conn.execute(sql)
        conn.commit()

    print("All indexes created.")


def evaluate_strategy(conn, sql):
    start = time()
    count = conn.execute(sql).fetchone()[0]
    elapsed = time() - start
    return count, elapsed


def evaluate_blocking(conn):
    total_true = conn.execute(
        "SELECT COUNT(*) FROM truth_pairs"
    ).fetchone()[0]

    print("\n" + "=" * 75)
    print("BLOCKING RECALL EVALUATION")
    print("=" * 75)
    print(f"Total true matches: {total_true:,}\n")

    strategies = {
        "B1_exact_name": """
            SELECT COUNT(*)
            FROM truth_pairs t
            JOIN records a
              ON a.entity_id = t.source1_entity_id
             AND a.source = 'S1'
            JOIN records b
              ON b.entity_id = t.matched_entity_id
             AND b.source IN ('S2', 'S3')
            WHERE a.country <> ''
              AND a.name_norm <> ''
              AND a.country = b.country
              AND a.name_norm = b.name_norm
        """,

        "B2_canonical_name": """
            SELECT COUNT(*)
            FROM truth_pairs t
            JOIN records a
              ON a.entity_id = t.source1_entity_id
             AND a.source = 'S1'
            JOIN records b
              ON b.entity_id = t.matched_entity_id
             AND b.source IN ('S2', 'S3')
            WHERE a.country <> ''
              AND a.name_canon <> ''
              AND a.country = b.country
              AND a.name_canon = b.name_canon
        """,

        "B3_exact_address": """
            SELECT COUNT(*)
            FROM truth_pairs t
            JOIN records a
              ON a.entity_id = t.source1_entity_id
             AND a.source = 'S1'
            JOIN records b
              ON b.entity_id = t.matched_entity_id
             AND b.source IN ('S2', 'S3')
            WHERE a.country <> ''
              AND a.address_norm <> ''
              AND a.country = b.country
              AND a.address_norm = b.address_norm
        """,

        "B4_name_plus_address": """
            SELECT COUNT(*)
            FROM truth_pairs t
            JOIN records a
              ON a.entity_id = t.source1_entity_id
             AND a.source = 'S1'
            JOIN records b
              ON b.entity_id = t.matched_entity_id
             AND b.source IN ('S2', 'S3')
            WHERE a.country <> ''
              AND a.name_canon <> ''
              AND a.address_canon <> ''
              AND a.country = b.country
              AND a.name_canon = b.name_canon
              AND a.address_canon = b.address_canon
        """,
    }

    results = []

    for name, sql in strategies.items():
        count, elapsed = evaluate_strategy(conn, sql)

        recall = (
            count / total_true
            if total_true
            else 0.0
        )

        results.append(
            {
                "strategy": name,
                "retrieved_true_matches": count,
                "recall_percent": recall * 100,
            }
        )

        print(name)
        print(f"  Retrieved true matches: {count:,}")
        print(f"  Recall: {recall * 100:.2f}%")
        print(f"  Query time: {elapsed:.1f} seconds")
        print()

    # --------------------------------------------------------
    # UNION of B1+B2+B3+B4
    # --------------------------------------------------------
    print("Calculating UNION recall...")
    print("This may take longer than the individual joins.")

    union_sql = """
        SELECT COUNT(*)
        FROM truth_pairs t
        JOIN records a
          ON a.entity_id = t.source1_entity_id
         AND a.source = 'S1'
        JOIN records b
          ON b.entity_id = t.matched_entity_id
         AND b.source IN ('S2', 'S3')
        WHERE
            (
                a.country <> ''
                AND a.name_norm <> ''
                AND a.country = b.country
                AND a.name_norm = b.name_norm
            )
            OR
            (
                a.country <> ''
                AND a.name_canon <> ''
                AND a.country = b.country
                AND a.name_canon = b.name_canon
            )
            OR
            (
                a.country <> ''
                AND a.address_norm <> ''
                AND a.country = b.country
                AND a.address_norm = b.address_norm
            )
            OR
            (
                a.country <> ''
                AND a.name_canon <> ''
                AND a.address_canon <> ''
                AND a.country = b.country
                AND a.name_canon = b.name_canon
                AND a.address_canon = b.address_canon
            )
    """

    union_count, union_time = evaluate_strategy(
        conn,
        union_sql,
    )

    union_recall = (
        union_count / total_true
        if total_true
        else 0.0
    )

    print("\n" + "-" * 75)
    print("MULTI-PASS UNION")
    print("-" * 75)
    print(f"True matches retrieved: {union_count:,}")
    print(f"UNION blocking recall: {union_recall * 100:.2f}%")
    print(f"Query time: {union_time:.1f} seconds")

    results.append(
        {
            "strategy": "UNION_B1_B2_B3_B4",
            "retrieved_true_matches": union_count,
            "recall_percent": union_recall * 100,
        }
    )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        "blocking_results.csv",
        index=False,
    )

    print("\nSaved: blocking_results.csv")

    return results


def main():
    start = time()

    print("=" * 75)
    print("AMAZON ML CHALLENGE 2026")
    print("MEMORY-SAFE BLOCKING EVALUATION")
    print("=" * 75)

    conn = connect_db()

    try:
        create_tables(conn)

        # Files are processed in chunks, so we never create giant
        # Python DataFrames for all 12+ million source records.
        insert_source_file(conn, S1_FILE, "S1")
        insert_source_file(conn, S2_FILE, "S2")
        insert_source_file(conn, S3_FILE, "S3")

        insert_ground_truth(conn)

        create_indexes(conn)

        evaluate_blocking(conn)

        print("\n" + "=" * 75)
        print("NEXT STEP")
        print("=" * 75)
        print(
            "Use the blocking recall numbers to choose the best "
            "candidate-generation strategy."
        )
        print(
            "After that, build candidate pairs and move to "
            "RapidFuzz/TF-IDF features + LightGBM/XGBoost."
        )

        print(
            f"\nTotal runtime: {time() - start:.1f} seconds"
        )

    finally:
        conn.close()

        # Delete temporary database after the evaluation.
        for suffix in ("", "-wal", "-shm"):
            path = DB_FILE + suffix

            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass


if __name__ == "__main__":
    main()
