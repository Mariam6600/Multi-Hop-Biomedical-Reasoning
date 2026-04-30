"""
src/load_dataset.py
====================
Biomedical Multi-Hop QA Project — Baseline 1

Loads and normalizes the MedHop dataset (dev.json).

What this module does:
  1. Reads dev.json from the real path on disk
  2. Extracts: id, query, candidates, answer per record
  3. Resolves Drug IDs to real names using DrugBank vocabulary
  4. Saves cleaned data to data/medhop.json
  5. Prints statistics about the loaded dataset

Output format (each record in medhop.json):
{
    "id":             "MH_dev_0",
    "query":          "interacts_with DB01171?",
    "query_drug_id":  "DB01171",
    "query_drug_name":"Moclobemide",
    "candidates":     ["DB00786", "DB00863", ...],
    "candidate_names":["Marimastat", "Ranitidine", ...],
    "answer":         "DB04844",
    "answer_name":    "Tetrabenazine"
}

Usage:
    py -3.10 src/load_dataset.py
"""

import json
import csv
import os
import sys
import re

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    ACTIVE_DATASET,
    DRUGBANK_VOCAB,
    MEDHOP_FILE,
    MAX_QUESTIONS,
    DATA_DIR,
)


# ─────────────────────────────────────────────
# STEP 1 — LOAD DRUGBANK VOCABULARY
# ─────────────────────────────────────────────

def load_drugbank_vocab(vocab_path: str) -> dict:
    """
    Load DrugBank vocabulary CSV.
    Returns dict: { "DB01171": "Moclobemide", ... }
    """
    vocab = {}

    if not os.path.exists(vocab_path):
        print(f"  [WARN] DrugBank vocab not found: {vocab_path}")
        print("  [WARN] Drug IDs will not be resolved to names.")
        return vocab

    try:
        with open(vocab_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # DrugBank CSV columns: DrugBank ID, Common name, ...
                drug_id   = row.get("DrugBank ID", "").strip()
                drug_name = row.get("Common name", "").strip()
                if drug_id and drug_name:
                    vocab[drug_id] = drug_name

        print(f"  [OK]   DrugBank vocab loaded — {len(vocab):,} drugs")
    except Exception as e:
        print(f"  [FAIL] Error loading DrugBank vocab: {e}")

    return vocab


# ─────────────────────────────────────────────
# STEP 2 — EXTRACT DRUG ID FROM QUERY
# ─────────────────────────────────────────────

def extract_drug_id_from_query(query: str) -> str:
    """
    Extract the drug ID from the query string.
    Example: "interacts_with DB01171?" -> "DB01171"
    """
    match = re.search(r"(DB\d{5})", query)
    if match:
        return match.group(1)
    return ""


# ─────────────────────────────────────────────
# STEP 3 — LOAD AND NORMALIZE MEDHOP DATASET
# ─────────────────────────────────────────────

def load_medhop(dataset_path: str, vocab: dict, max_questions: int = None) -> list:
    """
    Load MedHop JSON file and normalize records.

    Args:
        dataset_path:  Path to dev.json or train.json
        vocab:         DrugBank vocabulary dict
        max_questions: Limit number of records (None = all)

    Returns:
        List of normalized question dicts
    """
    if not os.path.exists(dataset_path):
        print(f"  [FAIL] Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"  [OK]   Raw records loaded: {len(raw_data):,}")

    # Apply limit if set
    if max_questions is not None:
        raw_data = raw_data[:max_questions]
        print(f"  [INFO] Limited to first {max_questions} questions (MAX_QUESTIONS setting)")

    normalized = []
    missing_vocab = 0

    for record in raw_data:
        question_id = record.get("id", "")
        query       = record.get("query", "").strip()
        candidates  = record.get("candidates", [])
        answer      = record.get("answer", "").strip()

        # Extract drug ID from query
        query_drug_id = extract_drug_id_from_query(query)

        # Resolve drug IDs to names
        query_drug_name  = vocab.get(query_drug_id, query_drug_id)
        answer_name      = vocab.get(answer, answer)
        candidate_names  = [vocab.get(c, c) for c in candidates]

        # Count unresolved IDs
        if query_drug_id and query_drug_name == query_drug_id:
            missing_vocab += 1

        # Extract supports (needed for Baseline 2 BM25 retrieval)
        supports = record.get("supports", [])

        normalized.append({
            "id":              question_id,
            "query":           query,
            "query_drug_id":   query_drug_id,
            "query_drug_name": query_drug_name,
            "candidates":      candidates,
            "candidate_names": candidate_names,
            "answer":          answer,
            "answer_name":     answer_name,
            "supports":        supports,
            "supports_count":  len(supports),
        })

    if missing_vocab > 0:
        print(f"  [WARN] {missing_vocab} query drug IDs not found in DrugBank vocab")

    return normalized


# ─────────────────────────────────────────────
# STEP 4 — SAVE TO OUTPUT FILE
# ─────────────────────────────────────────────

def save_dataset(data: list, output_path: str):
    """Save normalized dataset to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [OK]   Saved {len(data):,} records → {output_path}")
    print(f"  [INFO] File size: {size_kb:.1f} KB")


# ─────────────────────────────────────────────
# STEP 5 — PRINT DATASET STATISTICS
# ─────────────────────────────────────────────

def print_statistics(data: list):
    """Print useful statistics about the loaded dataset."""
    if not data:
        print("  [WARN] No data to show statistics for.")
        return

    total          = len(data)
    candidate_counts = [len(r["candidates"]) for r in data]
    avg_candidates = sum(candidate_counts) / total
    min_candidates = min(candidate_counts)
    max_candidates = max(candidate_counts)

    # Count unique answer drugs
    unique_answers = set(r["answer"] for r in data)

    # Supports stats
    supports_counts = [r.get("supports_count", 0) for r in data]
    avg_supports = sum(supports_counts) / total if total > 0 else 0

    print(f"  [INFO] Total questions     : {total:,}")
    print(f"  [INFO] Unique answers      : {len(unique_answers):,}")
    print(f"  [INFO] Avg candidates/q    : {avg_candidates:.1f}")
    print(f"  [INFO] Min candidates/q    : {min_candidates}")
    print(f"  [INFO] Max candidates/q    : {max_candidates}")
    print(f"  [INFO] Avg supports/q      : {avg_supports:.1f}")
    print(f"  [INFO] Supports loaded     : {'YES' if avg_supports > 0 else 'NO - check dataset'}")

    # Show first 3 examples
    print()
    print("  Sample records:")
    print("  " + "-" * 55)
    for record in data[:3]:
        print(f"  ID       : {record['id']}")
        print(f"  Query    : {record['query']}")
        print(f"  Drug     : {record['query_drug_id']} → {record['query_drug_name']}")
        print(f"  Answer   : {record['answer']} → {record['answer_name']}")
        print(f"  Candidates ({len(record['candidates'])}): {', '.join(record['candidates'][:4])}...")
        print("  " + "-" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Load Dataset — MedHop")
    print("=" * 60)
    print(f"  Source : {ACTIVE_DATASET}")
    print(f"  Output : {MEDHOP_FILE}")
    print(f"  Vocab  : {DRUGBANK_VOCAB}")
    print()

    # Step 1 — Load DrugBank vocabulary
    print("--- Loading DrugBank Vocabulary ---")
    vocab = load_drugbank_vocab(DRUGBANK_VOCAB)

    # Step 2 — Load and normalize MedHop dataset
    print("\n--- Loading MedHop Dataset ---")
    data = load_medhop(ACTIVE_DATASET, vocab, MAX_QUESTIONS)

    # Step 3 — Print statistics
    print("\n--- Dataset Statistics ---")
    print_statistics(data)

    # Step 4 — Save to medhop.json
    print("\n--- Saving Normalized Dataset ---")
    save_dataset(data, MEDHOP_FILE)

    print("\n" + "=" * 60)
    print("  ✅  Dataset loaded and saved successfully.")
    print("  Next step: src/prompt_builder.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
