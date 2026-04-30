"""
src/retriever_hybrid_scored.py
================================
Biomedical Multi-Hop QA — Hybrid Scored Retriever

  "score = 0.5 * MedCPT_similarity + 0.5 * term_count"
  + أوزان حسب نوع المصطلح (Mechanism > Protein > Pathway > Drug_class > Disease)

لماذا هذا أفضل من term_count فقط؟
  - term_count يعامل كل المصطلحات بنفس الوزن
  - MedCPT يفهم السياق الدلالي لكن يضمحل مع النصوص الطويلة
  - الجمع بينهما: MedCPT يمسك السياق العام + term_count يمسك الكلمات المفتاحية المهمة

الأوزان:
  Mechanism terms  → 3.0  (آلية عمل الدواء — الأهم للتفاعلات)
  Protein terms    → 2.5  (البروتينات والإنزيمات المستهدفة)
  Pathway terms    → 2.0  (المسارات البيولوجية)
  Drug class terms → 1.5  (تصنيف الدواء)
  Disease terms    → 1.0  (الأمراض المرتبطة)
  Drug name bonus  → 2.0  (وجود اسم الدواء في النص)
  Flat terms (no weight) → 1.0 (من الـ cache العادي)

يقبل:
  - weighted_terms: قائمة من {term, weight, category}  ← من query_expander_structured
  - flat_terms: قائمة من strings                        ← من query_expander العادي
  إذا توفّرت weighted_terms تُستخدم، وإلا تُستخدم flat_terms بوزن 1.0

Usage: EXP_RETRIEVER = "hybrid_scored" في inference_pipeline5.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    "mechanism":  3.0,
    "protein":    2.5,
    "pathway":    2.0,
    "drug_class": 1.5,
    "disease":    1.0,
    "unknown":    1.0,
}
DRUG_NAME_BONUS = 2.0
ALPHA = 0.5  # weight for MedCPT component (1-ALPHA for term component)


def retrieve_hybrid_scored(
    query: str,
    supports: list,
    drug_name: str,
    flat_terms: list = None,
    weighted_terms: list = None,
    top_k: int = 5,
) -> list:
    """
    Hybrid retrieval: combines MedCPT semantic similarity + weighted term count.

    score = ALPHA * medcpt_sim_norm + (1-ALPHA) * term_score_norm

    Args:
        query:         original query string
        supports:      list of support texts
        drug_name:     drug name for bonus scoring
        flat_terms:    list of strings (from regular cache) — each gets weight 1.0
        weighted_terms: list of {term, weight, category} dicts (from structured cache)
        top_k:         number of docs to return

    Returns: [{text, score, rank, medcpt_score, term_score, matched_terms}, ...]
    """
    if not supports:
        return []

    valid = [s for s in supports if s and isinstance(s, str) and s.strip()]
    if not valid:
        return []

    # ── Build weighted term list ──
    term_weights = []
    if weighted_terms:
        for wt in weighted_terms:
            t = wt.get("term", "").strip()
            if t:
                term_weights.append({
                    "term":     t.lower(),
                    "weight":   wt.get("weight", 1.0),
                    "category": wt.get("category", "unknown"),
                })
    elif flat_terms:
        for t in flat_terms:
            if t and t.strip():
                term_weights.append({
                    "term":     t.lower().strip(),
                    "weight":   1.0,
                    "category": "unknown",
                })

    drug_lower = drug_name.lower().strip() if drug_name else ""

    # ── Component 1: Term-count scores ──
    term_scores = []
    matched_per_doc = []
    for doc in valid:
        doc_lower = doc.lower()
        score = 0.0
        matched = []
        for tw in term_weights:
            if tw["term"] in doc_lower:
                score += tw["weight"]
                matched.append(f"{tw['term']}(w={tw['weight']})")
        if drug_lower and drug_lower in doc_lower:
            score += DRUG_NAME_BONUS
        term_scores.append(score)
        matched_per_doc.append(matched)

    term_arr = np.array(term_scores, dtype=float)

    # ── Component 2: MedCPT semantic similarity ──
    medcpt_arr = _get_medcpt_scores(query, drug_name, valid)

    # ── Normalize both to [0, 1] ──
    term_norm   = _normalize(term_arr)
    medcpt_norm = _normalize(medcpt_arr)

    # ── Combined score ──
    final = ALPHA * medcpt_norm + (1 - ALPHA) * term_norm

    # ── Rank and return top-K ──
    indices = np.argsort(final)[::-1][:top_k]

    results = []
    for rank_idx, doc_idx in enumerate(indices, 1):
        results.append({
            "text":          valid[doc_idx],
            "score":         float(final[doc_idx]),
            "rank":          rank_idx,
            "medcpt_score":  float(medcpt_arr[doc_idx]),
            "term_score":    float(term_arr[doc_idx]),
            "matched_terms": matched_per_doc[doc_idx],
        })

    return results


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _get_medcpt_scores(query: str, drug_name: str, supports: list) -> np.ndarray:
    """Get MedCPT cosine similarity scores for all supports."""
    try:
        from src.retriever_semantic import get_model
        import numpy as np

        model = get_model()
        enhanced_query = f"{drug_name} {query}".strip() if drug_name else query
        q_emb = np.array(model.encode([enhanced_query], show_progress_bar=False))
        d_embs = np.array(model.encode(supports, show_progress_bar=False))

        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        d_norm = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-9)
        sims = np.dot(q_norm, d_norm.T).flatten()
        return sims
    except Exception as e:
        print(f"  [WARN] MedCPT failed in hybrid_scored: {e} — using zeros")
        return np.zeros(len(supports))