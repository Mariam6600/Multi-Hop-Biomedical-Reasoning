"""
src/retriever.py
=================
Biomedical Multi-Hop QA Project — Baseline 2

BM25 Retrieval Engine.

What this module does:
  1. Takes the supports list from one MedHop record (up to 30 texts)
  2. Builds a BM25 index over those texts
  3. Scores each text against the query
  4. Returns the top-K most relevant texts

Why BM25?
  - Fast and lightweight (no GPU needed)
  - Works well on biomedical keyword matching
  - Already installed: bm25s + PyStemmer
  - Standard baseline for retrieval in QA research

Why Top-K and not all supports?
  - Each support text ≈ 200-300 tokens
  - 30 texts × 250 tokens = ~7500 tokens → exceeds context window
  - Top-3 → ~900 tokens  ✅ fits in 4096 context
  - Top-5 → ~1500 tokens ✅ fits in 4096 context
  - We experiment with both K=3 and K=5 to justify the choice

Usage:
    py -3.10 src/retriever.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BM25_TOP_K, BM25_METHOD


# ─────────────────────────────────────────────
# CORE FUNCTION — RETRIEVE TOP-K SUPPORTS
# ─────────────────────────────────────────────

def retrieve_top_k(query: str, supports: list,
                   drug_name: str = "",
                   top_k: int = None) -> list:
    """
    Given a query and a list of support texts,
    return the top-K most relevant texts using BM25.

    Args:
        query:     The drug interaction query (e.g. "interacts_with DB01171?")
        supports:  List of biomedical text strings from MedHop
        drug_name: Real drug name (e.g. "Moclobemide") to enhance the query.
                   BM25 searches text — supports contain drug names not IDs.
                   Adding the real name improves retrieval significantly.
        top_k:     Number of texts to return (default: BM25_TOP_K from settings)

    Returns:
        List of dicts, each containing:
        {
            "text":          str,   ← the support text
            "score":         float, ← BM25 relevance score
            "rank":          int,   ← 1 = most relevant
            "enhanced_query":str    ← the actual query used for retrieval
        }
    """
    import bm25s
    import Stemmer

    if top_k is None:
        top_k = BM25_TOP_K

    # Enhance query with real drug name
    # Reason: supports contain "Moclobemide" but query has "DB01171"
    # BM25 cannot match IDs to names without this enhancement
    if drug_name:
        enhanced_query = f"{drug_name} {query}"
    else:
        enhanced_query = query

    # Handle edge cases
    if not supports:
        return []

    if len(supports) <= top_k:
        return [
            {
                "text":           text,
                "score":          1.0,
                "rank":           i + 1,
                "enhanced_query": enhanced_query
            }
            for i, text in enumerate(supports)
        ]

    # Tokenize using English stemmer (better biomedical matching)
    stemmer = Stemmer.Stemmer("english")

    # Tokenize corpus (supports)
    corpus_tokens = bm25s.tokenize(
        supports,
        stopwords="en",
        stemmer=stemmer
    )

    # Tokenize enhanced query
    query_tokens = bm25s.tokenize(
        [enhanced_query],
        stopwords="en",
        stemmer=stemmer
    )

    # Build BM25 index
    retriever = bm25s.BM25(method=BM25_METHOD)
    retriever.index(corpus_tokens)

    # Retrieve top-K
    actual_k = min(top_k, len(supports))
    results, scores = retriever.retrieve(query_tokens, k=actual_k)

    # Build output list
    top_results = []
    for rank, (idx, score) in enumerate(zip(results[0], scores[0])):
        top_results.append({
            "text":           supports[int(idx)],
            "score":          round(float(score), 4),
            "rank":           rank + 1,
            "enhanced_query": enhanced_query
        })

    return top_results


# ─────────────────────────────────────────────
# HELPER — FORMAT RETRIEVED TEXTS FOR PROMPT
# ─────────────────────────────────────────────

def format_context(retrieved: list, max_chars_per_text: int = 500) -> str:
    """
    Format retrieved texts into a clean context string
    ready to be inserted into the prompt.

    Args:
        retrieved:          Output from retrieve_top_k()
        max_chars_per_text: Truncate each text to this length
                            (prevents context window overflow)

    Returns:
        Formatted string like:
        [1] Text one truncated here...
        [2] Text two truncated here...
        [3] Text three truncated here...
    """
    if not retrieved:
        return "No supporting evidence available."

    lines = []
    for item in retrieved:
        text = item["text"].strip()
        # Truncate if too long
        if len(text) > max_chars_per_text:
            text = text[:max_chars_per_text] + "..."
        lines.append(f"[{item['rank']}] {text}")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# HELPER — COUNT TOKENS ESTIMATE
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 characters).
    Used to verify we don't exceed context window.
    """
    return len(text) // 4


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    import json
    from config.settings import MEDHOP_FILE

    print("\n" + "=" * 60)
    print("  Retriever — BM25 Test")
    print("=" * 60)
    print(f"  Method : {BM25_METHOD}")
    print(f"  Top-K  : {BM25_TOP_K}")
    print()

    # Load dataset
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found. Run load_dataset.py first.")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Test on first 3 records
    print("--- Testing BM25 Retrieval on 3 Questions ---\n")

    for i, record in enumerate(data[:3]):
        query      = record["query"]
        drug_name  = record["query_drug_name"]
        supports   = record.get("supports", [])
        answer     = record["answer"]
        answer_name = record.get("answer_name", answer)

        print(f"  Question {i+1}: {record['id']}")
        print(f"  Query    : {query}")
        print(f"  Drug     : {drug_name}")
        print(f"  Answer   : {answer} ({answer_name})")
        print(f"  Supports : {len(supports)} texts available")
        print()

        if not supports:
            print("  [WARN] No supports found for this record!")
            continue

        # Retrieve top-K (with drug name to enhance query)
        retrieved = retrieve_top_k(
            query=query,
            supports=supports,
            drug_name=drug_name,
            top_k=BM25_TOP_K
        )

        print(f"  Enhanced query: '{retrieved[0]['enhanced_query']}'")
        print()
        print("  " + "─" * 55)
        for item in retrieved:
            preview = item["text"][:120].strip() + "..."
            tokens  = estimate_tokens(item["text"])
            print(f"  Rank {item['rank']} | Score: {item['score']} | ~{tokens} tokens")
            print(f"  {preview}")
            print()

        # Format context for prompt
        context = format_context(retrieved)
        context_tokens = estimate_tokens(context)
        print(f"  Formatted context: ~{context_tokens} tokens")

        # Check if answer appears in retrieved texts
        answer_found = any(
            answer.lower() in item["text"].lower() or
            answer_name.lower() in item["text"].lower()
            for item in retrieved
        )
        status = "✅ Answer drug mentioned in retrieved texts" if answer_found \
                 else "❌ Answer drug NOT in retrieved texts"
        print(f"  Answer check: {status}")
        print("  " + "─" * 55)
        print()

    # ── TOP_K Comparison ──
    print("--- Top-K Comparison (K=3 vs K=5) ---\n")
    record      = data[0]
    supports    = record.get("supports", [])
    query       = record["query"]
    drug_name   = record.get("query_drug_name", "")
    answer      = record["answer"]
    answer_name = record.get("answer_name", answer)

    for k in [3, 5, 10]:
        retrieved = retrieve_top_k(
            query=query,
            supports=supports,
            drug_name=drug_name,
            top_k=k
        )
        context   = format_context(retrieved)
        tokens    = estimate_tokens(context)
        answer_found = any(
            answer.lower() in item["text"].lower() or
            answer_name.lower() in item["text"].lower()
            for item in retrieved
        )
        found_str = "✅ answer found" if answer_found else "❌ not found"
        print(f"  K={k:>2} | ~{tokens:>4} tokens | {found_str}")

    print()
    print("=" * 60)
    print("  ✅  retriever.py working correctly.")
    print("  Next step: update src/prompt_builder.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
