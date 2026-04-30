"""
src/retriever_adaptive_k.py
============================
Biomedical Multi-Hop QA — Adaptive Top-K Retriever

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [FIXED v2] المنطق المعتمد: SCORE GAP (نسختان فقط: k=2 أو k=3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  التغييرات عن النسخة السابقة:
  ─────────────────────────────
  1. MAX_K = 3 (كان 5): حذف k=5 نهائياً بعد إثبات أنه يُضرّ الأداء
     من التحليل: k=5 أعطى 24.2% بينما k=3 أعطى 29% (مع نفس الأسئلة)
  2. decide_k_from_gap: فقط k=2 أو k=3 (لا EXTRA_K=5 بعد الآن)
     قرار بسيط: gap >= GAP_HIGH → k=2 | كل الباقي → k=3

  لماذا Gap وليس top1_score مباشرة؟
  ──────────────────────────────────
  من score_diagnostics.py على 342 سؤال (بيانات حقيقية):
    72% من الأسئلة لها top1_score >= 0.90
    42% من الأسئلة لها top1_score == 1.0
  → الأسكور مضغوط قرب 1.0 بسبب min-max normalization
  → Absolute threshold على top1_score لا يميّز بشكل فعّال

  الـ Score Gap (فرق rank1 - rank2) يعكس مدى وضوح الوثيقة الأفضل:
    Gap كبير  = rank1 متميّز جداً عن باقي الوثائق  → k=2 كافٍ
    Gap صغير  = الوثائق متقاربة في الجودة          → نحتاج k=3

  الثريشولدات (مشتقّة من البيانات الفعلية):
  ─────────────────────────────────────────
  GAP_HIGH = 0.309  (p75) → ~25% من الأسئلة → k = 2  (rank1 واضح)
  أي gap آخر                → ~75% من الأسئلة → k = 3  (الحالة الطبيعية)

  الحدود:
  ─────────
  MIN_K = 2:  multi-hop يحتاج على الأقل وثيقتين للمسار الكامل
  MAX_K = 3:  من التجارب:
              k=3 = 33.33% (B4-EXP4 baseline)
              k=5 = 24.2%  (EXP10 تحليل k=5 tier)
              k=5 يرفع retrieval recall بـ 4% لكن يخفض EM بـ 9%!
              → MAX_K = 3 هو التوازن الأمثل المُثبَت بالبيانات

  التوزيع المتوقع على 342 سؤال:
  ────────────────────────────────
    k=2: ~25%  (~85 سؤال)   — gap >= 0.309 (rank1 واضح وقوي)
    k=3: ~75%  (~257 سؤال)  — gap < 0.309  (الحالة الطبيعية)
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────
# THRESHOLDS — مشتقّة من score_diagnostics.py (342 سؤال كامل)
# ─────────────────────────────────────────────────────
#
#  Gap-based thresholds (GAP = score[rank1] - score[rank2])
#  من البيانات الفعلية 342 سؤال:
GAP_HIGH = 0.309   # p75  → k = MIN_K  (2)   | ~25% من الأسئلة، rank1 واضح قوي
GAP_MID  = 0.130   # p50  → k = DEFAULT_K (3) | مرجعي فقط — لم يعد يُستخدم في القرار
GAP_LOW  = 0.058   # p25  → محذوف من القرار (كان يؤدي إلى k=5 الضار)

# K values — [FIXED] MAX_K = 3 (was 5)
MIN_K     = 2   # أقل ما يُرسل للـ LLM (multi-hop requirement)
DEFAULT_K = 3   # يتوافق مع B4-EXP4 الأفضل (33.33%)
MAX_K     = 3   # [FIXED] الحد الأقصى = 3 (كان 5, أثبتت البيانات أن k>3 يضرّ)

# EXTRA_K محذوف — كان يؤدي إلى k=5 الضار
# EXTRA_K = 5  ← محذوف في v2


def decide_k_from_gap(
    score_rank1: float,
    score_rank2: float,
    gap_high: float = GAP_HIGH,
    gap_mid:  float = GAP_MID,   # مُبقى للتوافق مع الكود القديم لكن لا يؤثر في القرار
    gap_low:  float = GAP_LOW,   # مُبقى للتوافق مع الكود القديم لكن لا يؤثر في القرار
) -> tuple:
    """
    [FIXED v2] يحدد k المناسب: k=2 أو k=3 فقط.

    القرار المبسّط:
      gap >= gap_high → k=2  (rank1 متميّز جداً → k=2 كافٍ)
      أي شيء آخر    → k=3  (الحالة الطبيعية)

    تم حذف k=5 نهائياً بعد إثبات أنه يُضرّ الأداء في كل الحالات.

    Returns: (k, tier_label, gap_value)
    """
    gap = score_rank1 - score_rank2
    if gap >= gap_high:
        return MIN_K, "HIGH", gap
    else:
        return DEFAULT_K, "MID", gap


def retrieve_adaptive_k(
    query: str,
    supports: list,
    drug_name: str,
    flat_terms: list = None,
    weighted_terms: list = None,
    gap_high: float = GAP_HIGH,
    gap_mid:  float = GAP_MID,
    gap_low:  float = GAP_LOW,
) -> tuple:
    """
    [FIXED v2] Adaptive Top-K Retrieval — Score-Gap Based.

    الخوارزمية:
      1. استرجع MAX_K=3 وثائق (كان 5)
      2. احسب gap بين rank1 وrank2
      3. حدّد k: gap >= GAP_HIGH → k=2 | أي شيء آخر → k=3
      4. فرض MIN_K=2 و MAX_K=3

    Returns:
        (retrieved_docs, adaptive_k, gap_value, decision_info)
    """
    from src.retriever_hybrid_scored import retrieve_hybrid_scored

    if not supports:
        return [], MIN_K, 0.0, {"reason": "no_supports", "k": MIN_K}

    # Step 1: Retrieve MAX_K=3 docs (FIXED: was MAX_K=5)
    all_retrieved = retrieve_hybrid_scored(
        query=query,
        supports=supports,
        drug_name=drug_name,
        flat_terms=flat_terms or [],
        weighted_terms=weighted_terms or [],
        top_k=MAX_K,  # MAX_K=3
    )

    if not all_retrieved:
        return [], MIN_K, 0.0, {"reason": "no_results", "k": MIN_K}

    scores = [r["score"] for r in all_retrieved]

    # Step 2: Calculate gap
    gap = scores[0] - scores[1] if len(scores) >= 2 else scores[0]

    # Step 3: Decide k (FIXED: only k=2 or k=3)
    s2 = scores[1] if len(scores) >= 2 else 0.0
    adaptive_k, tier, _ = decide_k_from_gap(scores[0], s2, gap_high, gap_mid, gap_low)

    # Step 4: Enforce bounds (MIN_K=2, MAX_K=3)
    adaptive_k = max(MIN_K, min(MAX_K, adaptive_k))
    adaptive_k = min(adaptive_k, len(all_retrieved))
    adaptive_k = max(adaptive_k, min(MIN_K, len(all_retrieved)))

    selected = all_retrieved[:adaptive_k]

    decision_info = {
        "method":      "score_gap_v2",
        "adaptive_k":  adaptive_k,
        "gap":         round(gap, 4),
        "tier":        tier,
        "top1_score":  round(scores[0], 4),
        "all_scores":  [round(s, 4) for s in scores],
        "n_available": len(all_retrieved),
    }

    return selected, adaptive_k, gap, decision_info


def simulate_k_distribution(diagnostics_file: str) -> None:
    """
    يحاكي توزيع k على بيانات score_diagnostics للتحقق من الثريشولدات.

    Usage:
        py -3.10 src/retriever_adaptive_k.py
    """
    import json
    with open(diagnostics_file, encoding="utf-8") as f:
        data = json.load(f)

    recs   = data.get("per_question_scores", [])
    total  = len(recs)
    counts = {2: 0, 3: 0}

    for r in recs:
        s = r.get("all_scores", [])
        s1 = s[0] if s else 0.0
        s2 = s[1] if len(s) >= 2 else 0.0
        k, _, _ = decide_k_from_gap(s1, s2)
        counts[k] = counts.get(k, 0) + 1

    print("\n  ── Simulated K Distribution [FIXED v2] (from diagnostics data) ──")
    for k in sorted(counts.keys()):
        pct = counts[k] / total * 100 if total else 0
        bar = "█" * int(pct / 3)
        print(f"  k={k}: {counts[k]:>4} ({pct:>5.1f}%) {bar}")
    print(f"\n  Note: k=5 tier REMOVED (was {total - counts.get(2,0) - counts.get(3,0)} questions, avg 24.2% EM)")
    print()


if __name__ == "__main__":
    from config.settings import OUTPUTS_DIR
    diag_file = os.path.join(OUTPUTS_DIR, "score_diagnostics.json")
    if os.path.exists(diag_file):
        print(f"Simulating from: {diag_file}")
        simulate_k_distribution(diag_file)
    else:
        print(f"Run score_diagnostics.py first: {diag_file}")
