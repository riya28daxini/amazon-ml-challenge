import csv
import re
import os
import time
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_FILE = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    General normalization for business names and addresses.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Replace common symbols with spaces
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def canonical_name(text):
    """
    Slightly stronger normalization for business names.
    Removes common legal suffixes.
    """

    text = normalize_text(text)

    suffixes = [
        "private limited",
        "pvt ltd",
        "pvt limited",
        "private ltd",
        "limited",
        "ltd",
        "llp",
        "incorporated",
        "inc",
        "corporation",
        "corp",
        "company",
        "co"
    ]

    changed = True

    while changed:
        changed = False

        for suffix in suffixes:
            if text.endswith(" " + suffix):
                text = text[:-len(suffix)].strip()
                changed = True
                break

    return text


def normalize_address(text):
    """
    Address normalization.
    """

    text = normalize_text(text)

    replacements = {
        " road ": " rd ",
        " street ": " st ",
        " avenue ": " ave ",
        " apartment ": " apt ",
        " building ": " bldg ",
        " highway ": " hwy ",
        " lane ": " ln ",
        " sector ": " sec ",
    }

    text = " " + text + " "

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def make_name_address_key(name, address):
    n = canonical_name(name)
    a = normalize_address(address)

    if not n or not a:
        return ""

    return n + "||" + a


# ============================================================
# FILE READER
# ============================================================

def read_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            yield row


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(path, source_name):

    print()
    print("=" * 70)
    print(f"Building indexes for {source_name}")
    print("=" * 70)

    name_index = defaultdict(list)
    canonical_index = defaultdict(list)
    name_address_index = defaultdict(list)

    count = 0

    start = time.time()

    for row in read_tsv(path):

        entity_id = row["entity_id"]
        name = row.get("business_name", "")
        address = row.get("business_address", "")

        n1 = normalize_text(name)
        n2 = canonical_name(name)
        na = make_name_address_key(name, address)

        if n1:
            name_index[n1].append(entity_id)

        if n2:
            canonical_index[n2].append(entity_id)

        if na:
            name_address_index[na].append(entity_id)

        count += 1

        if count % 500000 == 0:
            elapsed = time.time() - start
            print(
                f"{source_name}: {count:,} records processed "
                f"({elapsed / 60:.1f} min)"
            )

    print(f"{source_name}: {count:,} records indexed")
    print(f"Time: {(time.time() - start) / 60:.2f} minutes")

    return (
        name_index,
        canonical_index,
        name_address_index
    )


# ============================================================
# CREATE MATCHES
# ============================================================

def generate_submission():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("AMAZON ML CHALLENGE")
    print("TEST-SET BASELINE SUBMISSION")
    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [S1_FILE, S2_FILE, S3_FILE]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n"
                "Please check your dataset/test folder."
            )

    # --------------------------------------------------------
    # Build S2 indexes
    # --------------------------------------------------------

    s2_indexes = build_indexes(
        S2_FILE,
        "S2"
    )

    # --------------------------------------------------------
    # Build S3 indexes
    # --------------------------------------------------------

    s3_indexes = build_indexes(
        S3_FILE,
        "S3"
    )

    (
        s2_name,
        s2_canonical,
        s2_name_address
    ) = s2_indexes

    (
        s3_name,
        s3_canonical,
        s3_name_address
    ) = s3_indexes

    # --------------------------------------------------------
    # Process S1
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Generating test predictions")
    print("=" * 70)

    start = time.time()

    total_s1 = 0
    total_candidates = 0
    total_matches = 0

    with open(
        MATCHING_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as match_out, open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as candidate_out:

        match_writer = csv.writer(
            match_out,
            delimiter="\t",
            lineterminator="\n"
        )

        candidate_writer = csv.writer(
            candidate_out,
            delimiter="\t",
            lineterminator="\n"
        )

        # Headers required by challenge
        match_writer.writerow([
            "source1_entity_id",
            "matched_entity_ids"
        ])

        candidate_writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])

        for row in read_tsv(S1_FILE):

            s1_id = row["entity_id"]

            name = row.get("business_name", "")
            address = row.get("business_address", "")

            n1 = normalize_text(name)
            n2 = canonical_name(name)
            na = make_name_address_key(name, address)

            candidates = set()
            matches = set()

            # ==================================================
            # BLOCK 1: EXACT NORMALIZED NAME
            # ==================================================

            if n1:

                for entity_id in s2_name.get(n1, []):
                    candidates.add(entity_id)
                    matches.add(entity_id)

                for entity_id in s3_name.get(n1, []):
                    candidates.add(entity_id)
                    matches.add(entity_id)

            # ==================================================
            # BLOCK 2: CANONICAL NAME
            # ==================================================

            if n2:

                for entity_id in s2_canonical.get(n2, []):
                    candidates.add(entity_id)

                for entity_id in s3_canonical.get(n2, []):
                    candidates.add(entity_id)

            # ==================================================
            # BLOCK 3: NAME + ADDRESS
            # ==================================================

            if na:

                for entity_id in s2_name_address.get(na, []):
                    candidates.add(entity_id)
                    matches.add(entity_id)

                for entity_id in s3_name_address.get(na, []):
                    candidates.add(entity_id)
                    matches.add(entity_id)

            # --------------------------------------------------
            # IMPORTANT:
            # Only exact-name / exact name+address matches
            # are accepted for this baseline.
            #
            # Canonical-name candidates are retained in
            # candidate_pairs.tsv but not automatically predicted.
            # --------------------------------------------------

            candidate_list = sorted(candidates)

            match_list = sorted(matches)

            total_candidates += len(candidate_list)
            total_matches += len(match_list)

            # Candidate file
            candidate_writer.writerow([
                s1_id,
                ",".join(candidate_list)
            ])

            # Final prediction file
            match_writer.writerow([
                s1_id,
                ",".join(match_list)
            ])

            total_s1 += 1

            if total_s1 % 100000 == 0:

                elapsed = time.time() - start

                print(
                    f"S1 processed: {total_s1:,} | "
                    f"avg candidates: "
                    f"{total_candidates / total_s1:.2f} | "
                    f"predicted matches: {total_matches:,} | "
                    f"time: {elapsed / 60:.1f} min"
                )

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("SUBMISSION FILES CREATED")
    print("=" * 70)

    print(f"S1 entities processed : {total_s1:,}")
    print(f"Total candidates      : {total_candidates:,}")
    print(f"Total predictions     : {total_matches:,}")
    print(
        f"Average candidates/S1: "
        f"{total_candidates / max(total_s1, 1):.2f}"
    )

    print()
    print("matching_results.tsv:")
    print(MATCHING_FILE)

    print()
    print("candidate_pairs.tsv:")
    print(CANDIDATE_FILE)

    print()
    print(f"Total generation time: {elapsed / 60:.2f} minutes")

    print()
    print("NEXT STEP:")
    print("Run the official submission validator before uploading.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        generate_submission()

    except KeyboardInterrupt:

        print("\nProcess interrupted by user.")

    except Exception as e:

        print()
        print("ERROR:")
        print(e)
        raise