"""
src/retriever_weighted_struct.py
=================================
Biomedical Multi-Hop QA — Structured Weighted Retriever
التجربة: B3-EXP-STRUCT

الفرق عن retriever_expanded.py:
  بدل كل مصطلح = 1 نقطة، كل مصطلح يأخذ وزناً حسب فئته:
    - Mechanism terms  → 3.0
    - Protein terms    → 2.5
    - Pathway terms    → 2.0
    - Drug class terms → 1.5
    - Disease terms    → 1.0

  مثال:
    وجود "CYP2D6 inhibitor" (mechanism) في نص = +3.0 نقطة
    وجود "depression" (disease) في نص = +1.0 نقطة فقط

  يُستخدم مع query_expander_structured.py

الاستخدام في inference_pipeline5.py:
    EXP_RETRIEVER = "weighted_struct"
    → if retriever_type == "weighted_struct":
           retrieved = retrieve_weighted_struct(...)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def retrieve_weighted_struct(
    query: str,
    supports: list,
    drug_name: str,
    weighted_terms: list,
    top_k: int = 5,
) -> list:
    """
    Rank documents using weighted term matching.

    Args:
        weighted_terms: List of dicts from query_expander_structured:
                        [{"term": str, "weight": float, "category": str}, ...]

    Returns: Same format as retrieve_expanded()
    """
    if not supports:
        return []

    if not weighted_terms and not drug_name:
        return [
            {"text": s, "score": 0.0, "rank": i + 1, "matched_terms": []}
            for i, s in enumerate(supports[:top_k])
        ]

    drug_lower = drug_name.lower().strip() if drug_name else ""

    # Normalize weighted terms
    norm_terms = []
    seen = set()
    for wt in weighted_terms:
        t_lower = wt["term"].lower().strip()
        if t_lower and t_lower not in seen:
            seen.add(t_lower)
            norm_terms.append({
                "term":     t_lower,
                "weight":   wt["weight"],
                "category": wt.get("category", "unknown"),
                "original": wt["term"],
            })

    scored = []
    for doc_text in supports:
        if not doc_text or not doc_text.strip():
            continue

        doc_lower = doc_text.lower()
        score = 0.0
        matched = []

        for wt in norm_terms:
            if wt["term"] in doc_lower:
                score += wt["weight"]
                matched.append(f"{wt['original']}(w={wt['weight']})")

        # Drug name bonus (same as before)
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
