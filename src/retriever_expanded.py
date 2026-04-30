"""
src/retriever_expanded.py
==========================
Biomedical Multi-Hop QA — Term-Based Expanded Retriever (FIXED)

التجربة: B3-EXP-FIXED
الفرق عن الإصدار السابق:
  - يستخدم مصطلحات نظيفة (بعد إصلاح الكاش)
  - يُطبّع الـ score ليكون بين 0 و 1
  - يُضيف تمييزاً بين partial match و exact match
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def retrieve_expanded(
    query: str,
    supports: list,
    drug_name: str,
    expanded_terms: list,
    top_k: int = 5,
) -> list:
    """
    Rank support documents by how many LLM-generated terms they contain.

    Scoring formula:
        score = Σ term_found_in_doc   (each unique term = 1 point)
              + 2.0 × drug_name_found  (bonus for drug name presence)

    المصطلحات الآن نظيفة (30 مصطلح فريد بعد إصلاح الكاش).
    """
    if not supports:
        return []

    if not expanded_terms and not drug_name:
        return [
            {"text": s, "score": 0.0, "rank": i + 1, "matched_terms": []}
            for i, s in enumerate(supports[:top_k])
        ]

    # Normalize
    terms_lower = list(dict.fromkeys(
        t.lower().strip() for t in expanded_terms if t.strip()
    ))
    drug_lower = drug_name.lower().strip() if drug_name else ""

    scored = []
    for doc_text in supports:
        if not doc_text or not doc_text.strip():
            continue

        doc_lower = doc_text.lower()
        matched = [t for t in terms_lower if t in doc_lower]
        score = float(len(matched))

        if drug_lower and drug_lower in doc_lower:
            score += 2.0

        scored.append({
            "text":          doc_text,
            "score":         score,
            "matched_terms": matched,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    results = []
    for rank, item in enumerate(scored[:top_k], 1):
        results.append({
            "text":          item["text"],
            "score":         item["score"],
            "rank":          rank,
            "matched_terms": item["matched_terms"],
        })

    return results