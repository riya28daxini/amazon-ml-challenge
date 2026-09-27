#!/usr/bin/env python3
"""
train_lightgbm_matcher.py  (final)

Trains a LightGBM match/no-match classifier on top of the existing
blocking + similarity-feature pipeline (reuses connect_db, insert_source,
create_indexes, collect_candidates, fetch_candidates, similarity_features,
and f05 from cap_tradeoff_dev.py) and produces a test submission whose
decision threshold is tuned directly against macro F_0.5.

DROP THIS FILE IN THE SAME FOLDER AS cap_tradeoff_dev.py (student_resource/).

Setup (once):
    pip install lightgbm pandas numpy rapidfuzz

Run:
    python train_lightgbm_matcher.py

What makes this the "final" version vs earlier iterations:
  - Training-pair collection AND test-set scoring both run in parallel
    across CPU cores (previously only test scoring was parallelized).
  - rapidfuzz is used for string similarity when available (falls back
    to cap_tradeoff_dev.py's pure-Python versions otherwise).
  - Training uses a 150k-row sample of train S1 (not all 2.2M) -- already
    yields several million labeled pairs, which is plenty for LightGBM.
  - The trained model + tuned threshold are pickled to disk right after
    training. If you ever have to kill the process during test scoring,
    RE-RUNNING THIS SCRIPT SKIPS TRAINING ENTIRELY and resumes at test
    scoring. Delete matcher_model_state.pkl only if you want to retrain
    from scratch (e.g. after changing NEG_CAP_PER_S1 or the sample size).
  - Ground truth passed to worker processes is pre-filtered to just the
    sampled S1 entities, to keep per-worker memory down.

Expected order-of-magnitude timing on a mid-range laptop (6 usable
cores): training-pair collection ~10-20 min, training ~1-2 min,
threshold tuning ~1 min, test scoring ~1-3 hours (test set is much
bigger than the 150k train sample and scores ALL candidates, not a
capped subsample). Your exact numbers depend on core count and disk
speed -- there is nothing left to tune for correctness, only for speed
if you want to trade it against blocking recall (see CANDIDATE_CAP_TEST
below).
"""

import csv
import multiprocessing as mp
import os
import pickle
import random
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("LightGBM is not installed. Run: pip install lightgbm")

# Reuse the already-working blocking + feature pipeline instead of
# reimplementing it.
import cap_tradeoff_dev as ctd

# ------------------------------------------------------------------
# Speed patch: cap_tradeoff_dev.py's levenshtein_ratio / token_set_ratio
# are pure-Python DP loops. If rapidfuzz is installed, swap in its
# C-accelerated equivalents. similarity_features() calls these as bare
# module-level names, so reassigning the attributes on the ctd module is
# enough. Because Windows multiprocessing uses "spawn", each worker
# re-imports this module fresh, so this patch re-applies automatically
# in every worker too -- nothing extra needed there.
# ------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz.distance import Levenshtein as _rf_lev

    def _fast_levenshtein_ratio(a, b):
        a, b = a or "", b or ""
        if not a and not b:
            return 1.0
        return _rf_lev.normalized_similarity(a, b)

    def _fast_token_set_ratio(a, b):
        a, b = a or "", b or ""
        if not a and not b:
            return 1.0
        return _rf_fuzz.token_set_ratio(a, b) / 100.0

    ctd.levenshtein_ratio = _fast_levenshtein_ratio
    ctd.token_set_ratio = _fast_token_set_ratio
    print("rapidfuzz detected -- using fast C-accelerated string similarity")
except ImportError:
    print("rapidfuzz NOT installed -- similarity will be much slower. "
          "Run: pip install rapidfuzz")

# ------------------------------------------------------------------
# Accent-folding patch: cap_tradeoff_dev.py's clean_text() uses Unicode
# NFKC normalization, which does NOT strip diacritics ("cafe" and "café"
# stay different strings). Since some entities (e.g. France) appear only
# in test, never in train, accented names/addresses currently compare as
# dissimilar to their unaccented equivalents -- for both blocking keys
# AND similarity features. This wraps clean_text() to additionally fold
# accents to their base ASCII letters. Applied before any DB is built, so
# both blocking indexes and stored canonical text pick it up consistently.
# ------------------------------------------------------------------
import unicodedata as _ud

_original_clean_text = ctd.clean_text


def _strip_accents(text):
    decomposed = _ud.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not _ud.combining(ch))


def _accent_folding_clean_text(value):
    return _strip_accents(_original_clean_text(value))


ctd.clean_text = _accent_folding_clean_text
print("accent-folding enabled on top of clean_text()")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

TRAIN_DB_FILE = os.path.join(BASE_DIR, "matcher_train.sqlite")
TEST_DB_FILE = os.path.join(BASE_DIR, "matcher_test.sqlite")

# Trained model + tuned threshold get saved here. If this file exists,
# main() skips straight to test-set scoring.
MODEL_STATE_FILE = os.path.join(BASE_DIR, "matcher_model_state.pkl")

# How many candidates collect_candidates() may return per S1 entity.
CANDIDATE_CAP_TRAIN = 500
CANDIDATE_CAP_TEST = 500   # lower this (e.g. 200) to trade recall for speed

# Per S1 entity, how many non-matching candidates to keep for training.
# collect_candidates() returns candidates in blocking-priority order, so
# the first non-matches are the closest near-misses -- exactly what the
# classifier needs to learn precision from.
NEG_CAP_PER_S1 = 25

# You don't need every train S1 row -- at ~25-30 labeled pairs per row,
# 150k rows already yields several million training pairs.
TRAIN_S1_SAMPLE_SIZE = 150_000

VAL_FRACTION = 0.15   # fraction of sampled train S1 entities held out
RANDOM_SEED = 42

THRESHOLD_GRID = np.round(np.arange(0.05, 0.99, 0.02), 2)

FEATURE_COLUMNS = [
    "name_lev", "name_tok", "name_char",
    "address_lev", "address_tok", "address_char",
    "postal_match", "country_match", "score",
    "name_len_diff", "address_len_diff", "exact_name",
]

# Parallelism. Leave one core free for the OS/foreground apps.
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
TRAIN_CHUNK_SIZE = 300   # S1 rows per work unit during training collection
TEST_CHUNK_SIZE = 50     # S1 rows per work unit during test scoring

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# SHARED HELPERS (used by both main process and workers)
# ============================================================

def load_s1_full(path):
    """Full S1 loader, same row shape as ctd.load_s1_sample() so it works
    with ctd.collect_candidates() / ctd.similarity_features() unchanged."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = str(row["business_name"] or "")
            address = str(row["business_address"] or "")
            nc = ctd.canonical_name(name)
            ac = ctd.canonical_address(address)
            rows.append({
                "entity_id": str(row["entity_id"]).strip(),
                "country": ctd.country_norm(row["country"]),
                "business_name": name,
                "business_address": address,
                "name_norm": ctd.clean_text(name),
                "name_canon": nc,
                "address_norm": ctd.clean_text(address),
                "address_canon": ac,
                "postal": ctd.postal_code(address),
                "prefix6": ctd.prefix(nc, 6),
                "prefix4": ctd.prefix(nc, 4),
                "suffix4": ctd.suffix(nc, 4),
                "first4": ctd.first4(nc),
                "last4": ctd.last4(nc),
            })
    return rows


def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = str(row["source1_entity_id"]).strip()
            value = str(row.get("matched_entity_ids", "") or "").strip()
            gt[s1_id] = (
                {x.strip() for x in value.split(",") if x.strip()}
                if value else set()
            )
    return gt


def extra_features(s1, candidate):
    name_len_diff = abs(len(s1["name_canon"]) - len(candidate["name_canon"]))
    address_len_diff = abs(
        len(s1["address_canon"]) - len(candidate["address_canon"])
    )
    exact_name = int(
        bool(s1["name_norm"])
        and s1.get("name_norm") == candidate.get("name_norm")
    )
    return name_len_diff, address_len_diff, exact_name


def build_token_index(conn):
    """Extra blocking layer on top of cap_tradeoff_dev.py's 10 exact-match
    passes: an inverted index from informative name tokens to entity_id.
    informative_tokens() (stopword/short-token filtered) already exists in
    cap_tradeoff_dev.py and is used for scoring similarity, but was never
    used for candidate generation -- this catches true matches that share
    a meaningful word but differ in word order, prefix, or suffix (which
    the existing exact-match blocking keys cannot)."""
    print("  building token index (extra blocking layer)...")
    conn.execute("DROP TABLE IF EXISTS token_index")
    conn.execute("CREATE TABLE token_index (token TEXT, entity_id TEXT)")

    cur = conn.execute(
        "SELECT entity_id, name_canon FROM records WHERE source IN ('S2','S3')"
    )
    batch = []
    BATCH_SIZE = 50_000
    for entity_id, name_canon in cur:
        for tok in ctd.informative_tokens(name_canon or ""):
            batch.append((tok, entity_id))
            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO token_index VALUES (?, ?)", batch
                )
                batch.clear()
    if batch:
        conn.executemany("INSERT INTO token_index VALUES (?, ?)", batch)

    conn.execute("CREATE INDEX idx_token_index_token ON token_index(token)")
    conn.commit()


def collect_candidates_extended(conn, s1):
    """cap_tradeoff_dev.py's collect_candidates() (10 exact-match passes,
    already capped at ctd.CAP) PLUS a token-based pass that only adds
    candidates if the cap wasn't already reached -- pure recall gain,
    never removes anything the original function found."""
    ids = ctd.collect_candidates(conn, s1)
    if len(ids) >= ctd.CAP:
        return ids

    seen = set(ids)
    for tok in ctd.informative_tokens(s1.get("name_canon", "")):
        cur = conn.execute(
            "SELECT entity_id FROM token_index WHERE token = ?", (tok,)
        )
        for (eid,) in cur:
            if eid not in seen:
                seen.add(eid)
                ids.append(eid)
                if len(ids) >= ctd.CAP:
                    return ids
    return ids


def build_db(db_file, s2_path, s3_path):
    ctd.DB_FILE = db_file
    conn = ctd.connect_db()
    ctd.insert_source(conn, s2_path, "S2")
    ctd.insert_source(conn, s3_path, "S3")
    ctd.create_indexes(conn)
    build_token_index(conn)
    return conn


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ============================================================
# STEP 1-4 (parallel): BUILD LABELED TRAINING TABLE
# ============================================================

_train_gt = None  # set per worker by _init_train_worker


def _init_train_worker(db_file, cap, gt_subset):
    global _train_gt
    ctd.DB_FILE = db_file
    ctd.CAP = cap
    _train_gt = gt_subset


def _build_pairs_for_chunk(s1_chunk):
    conn = sqlite3.connect(f"file:{ctd.DB_FILE}?mode=ro", uri=True)
    records = []
    try:
        for s1 in s1_chunk:
            ids = collect_candidates_extended(conn, s1)
            if not ids:
                continue

            true_ids = _train_gt.get(s1["entity_id"], set())
            pos_ids = [c for c in ids if c in true_ids]
            neg_ids = [c for c in ids if c not in true_ids][:NEG_CAP_PER_S1]
            kept_ids = pos_ids + neg_ids
            if not kept_ids:
                continue

            candidates = ctd.fetch_candidates(conn, kept_ids)
            for cid in kept_ids:
                cand = candidates.get(cid)
                if cand is None:
                    continue
                feats = ctd.similarity_features(s1, cand)
                nld, ald, exact = extra_features(s1, cand)
                feats["name_len_diff"] = nld
                feats["address_len_diff"] = ald
                feats["exact_name"] = exact
                feats["label"] = int(cid in true_ids)
                feats["s1_entity_id"] = s1["entity_id"]
                feats["candidate_entity_id"] = cid
                records.append(feats)
    finally:
        conn.close()

    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


def build_training_table():
    print("=" * 70)
    print("STEP 1: building TRAIN blocking DB (full S2 + S3)")
    print("=" * 70)

    conn = build_db(TRAIN_DB_FILE, TRAIN_S2, TRAIN_S3)
    conn.close()  # workers open their own read-only connections

    print("\nSTEP 2: loading train S1 + ground truth")
    s1_rows = load_s1_full(TRAIN_S1)
    gt_full = load_ground_truth(TRAIN_GT)
    print(f"  train S1 entities available: {len(s1_rows):,}")

    if TRAIN_S1_SAMPLE_SIZE and len(s1_rows) > TRAIN_S1_SAMPLE_SIZE:
        random.Random(RANDOM_SEED).shuffle(s1_rows)
        s1_rows = s1_rows[:TRAIN_S1_SAMPLE_SIZE]
        print(f"  sampled down to: {len(s1_rows):,} (TRAIN_S1_SAMPLE_SIZE)")

    # Keep only the ground-truth entries the sampled rows actually need --
    # this is what gets copied into every worker process, so trimming it
    # keeps per-worker memory down.
    gt_subset = {s1["entity_id"]: gt_full.get(s1["entity_id"], set())
                 for s1 in s1_rows}

    print(f"\nSTEP 3-4: collecting candidates, labeling, computing features "
          f"({NUM_WORKERS} worker processes)")
    chunks = list(chunked(s1_rows, TRAIN_CHUNK_SIZE))

    dfs = []
    processed_rows = 0
    start = time.time()

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_init_train_worker,
        initargs=(TRAIN_DB_FILE, CANDIDATE_CAP_TRAIN, gt_subset),
    ) as pool:
        for i, chunk_df in enumerate(
            pool.imap_unordered(_build_pairs_for_chunk, chunks), 1
        ):
            if not chunk_df.empty:
                dfs.append(chunk_df)
            processed_rows += TRAIN_CHUNK_SIZE

            if i % 20 == 0 or i == len(chunks):
                elapsed = (time.time() - start) / 60
                pairs_so_far = sum(len(d) for d in dfs)
                shown = min(processed_rows, len(s1_rows))
                print(f"  ~{shown:,}/{len(s1_rows):,} S1 processed | "
                      f"labeled pairs so far: {pairs_so_far:,} | "
                      f"time: {elapsed:.1f} min")

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"\nTraining table built: {len(df):,} pairs "
          f"({int(df['label'].sum()):,} positive, "
          f"{int((df['label'] == 0).sum()):,} negative)")
    return df


# ============================================================
# STEP 5-7: SPLIT, TRAIN, TUNE THRESHOLD
# ============================================================

def split_by_entity(df, val_fraction, seed):
    entity_ids = df["s1_entity_id"].unique().tolist()
    random.Random(seed).shuffle(entity_ids)
    n_val = int(len(entity_ids) * val_fraction)
    val_ids = set(entity_ids[:n_val])
    val_mask = df["s1_entity_id"].isin(val_ids)
    return df[~val_mask].copy(), df[val_mask].copy()


def train_model(train_df):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["label"]

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


def tune_threshold(model, val_df):
    val_df = val_df.copy()
    val_df["proba"] = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]

    truth = (
        val_df[val_df["label"] == 1]
        .groupby("s1_entity_id")["candidate_entity_id"]
        .apply(set)
    )
    all_s1 = val_df["s1_entity_id"].unique()

    best_threshold, best_score = None, -1.0

    for threshold in THRESHOLD_GRID:
        pred = (
            val_df[val_df["proba"] >= threshold]
            .groupby("s1_entity_id")["candidate_entity_id"]
            .apply(set)
        )

        scores = []
        for s1_id in all_s1:
            t = truth.get(s1_id, set())
            p = pred.get(s1_id, set())
            scores.append(ctd.f05(t, p))

        macro_f05 = sum(scores) / len(scores)
        print(f"  threshold={threshold:.2f}  macro F0.5={macro_f05:.4f}")

        if macro_f05 > best_score:
            best_score, best_threshold = macro_f05, threshold

    print(f"\nBest threshold: {best_threshold:.2f} "
          f"(validation macro F0.5 = {best_score:.4f})")
    return best_threshold, best_score


# ============================================================
# STEP 8 (parallel): SCORE TEST SET AND WRITE SUBMISSION
# ============================================================

_test_model = None
_test_threshold = None


def _init_test_worker(db_file, cap, model, threshold):
    global _test_model, _test_threshold
    ctd.DB_FILE = db_file
    ctd.CAP = cap
    _test_model = model
    _test_threshold = threshold


def _score_chunk(s1_chunk):
    conn = sqlite3.connect(f"file:{ctd.DB_FILE}?mode=ro", uri=True)
    results = []
    try:
        for s1 in s1_chunk:
            ids = collect_candidates_extended(conn, s1)
            if not ids:
                results.append((s1["entity_id"], "", ""))
                continue

            candidates = ctd.fetch_candidates(conn, ids)
            rows = []
            for cid in ids:
                cand = candidates.get(cid)
                if cand is None:
                    continue
                feats = ctd.similarity_features(s1, cand)
                nld, ald, exact = extra_features(s1, cand)
                feats["name_len_diff"] = nld
                feats["address_len_diff"] = ald
                feats["exact_name"] = exact
                feats["candidate_entity_id"] = cid
                rows.append(feats)

            if not rows:
                results.append((s1["entity_id"], ",".join(sorted(ids)), ""))
                continue

            feat_df = pd.DataFrame.from_records(rows)
            proba = _test_model.predict_proba(feat_df[FEATURE_COLUMNS])[:, 1]
            matched = feat_df.loc[
                proba >= _test_threshold, "candidate_entity_id"
            ]
            results.append((
                s1["entity_id"],
                ",".join(sorted(ids)),
                ",".join(sorted(matched)),
            ))
    finally:
        conn.close()
    return results


def generate_test_submission(model, threshold):
    print("\n" + "=" * 70)
    print(f"STEP 8: building TEST blocking DB and scoring with the model "
          f"({NUM_WORKERS} worker processes)")
    print("=" * 70)

    ctd.CAP = CANDIDATE_CAP_TEST
    conn = build_db(TEST_DB_FILE, TEST_S2, TEST_S3)
    conn.close()

    s1_rows = load_s1_full(TEST_S1)
    print(f"  test S1 entities: {len(s1_rows):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunks = list(chunked(s1_rows, TEST_CHUNK_SIZE))
    start = time.time()
    scored = 0

    with open(MATCHING_FILE, "w", encoding="utf-8", newline="") as mf, \
         open(CANDIDATE_FILE, "w", encoding="utf-8", newline="") as cf, \
         mp.Pool(
             processes=NUM_WORKERS,
             initializer=_init_test_worker,
             initargs=(TEST_DB_FILE, CANDIDATE_CAP_TEST, model, threshold),
         ) as pool:

        match_writer = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cand_writer = csv.writer(cf, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk_results in pool.imap_unordered(_score_chunk, chunks):
            for entity_id, cand_ids, matched_ids in chunk_results:
                cand_writer.writerow([entity_id, cand_ids])
                match_writer.writerow([entity_id, matched_ids])
            scored += len(chunk_results)

            if scored % 10000 < TEST_CHUNK_SIZE or scored >= len(s1_rows):
                elapsed = (time.time() - start) / 60
                print(f"  scored ~{min(scored, len(s1_rows)):,}/"
                      f"{len(s1_rows):,} | time: {elapsed:.1f} min")

    print(f"\nWrote {MATCHING_FILE}")
    print(f"Wrote {CANDIDATE_FILE}")
    print("\nNEXT STEP: run utils/validate_submission.py before uploading.")


# ============================================================
# MAIN
# ============================================================

def main():
    for path in [TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
                 TEST_S1, TEST_S2, TEST_S3]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found:\n{path}")

    if os.path.exists(MODEL_STATE_FILE):
        print(f"Found saved model at {MODEL_STATE_FILE} -- "
              f"skipping STEP 1-7 and resuming at STEP 8.")
        print("(Delete this file if you want to retrain from scratch.)")
        with open(MODEL_STATE_FILE, "rb") as f:
            state = pickle.load(f)
        model, threshold, val_score = (
            state["model"], state["threshold"], state["val_score"]
        )
    else:
        df = build_training_table()
        train_df, val_df = split_by_entity(df, VAL_FRACTION, RANDOM_SEED)
        print(f"\nTrain pairs: {len(train_df):,} | Val pairs: {len(val_df):,}")

        print("\n" + "=" * 70)
        print("STEP 6: training LightGBM classifier")
        print("=" * 70)
        model = train_model(train_df)

        print("\n" + "=" * 70)
        print("STEP 7: tuning decision threshold against macro F0.5")
        print("=" * 70)
        threshold, val_score = tune_threshold(model, val_df)

        with open(MODEL_STATE_FILE, "wb") as f:
            pickle.dump(
                {"model": model, "threshold": threshold,
                 "val_score": val_score},
                f,
            )
        print(f"\nSaved model + threshold to {MODEL_STATE_FILE}")

    generate_test_submission(model, threshold)

    print("\n" + "=" * 70)
    print(f"DONE. Validation macro F0.5 was {val_score:.4f} at "
          f"threshold {threshold:.2f}.")
    print("Compare that number to your 0.345 leaderboard baseline after "
          "you upload matching_results.tsv.")
    print("=" * 70)


if __name__ == "__main__":
    main()