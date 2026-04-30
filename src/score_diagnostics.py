"""
src/score_diagnostics.py
=========================
أداة التشخيص — تحليل توزيع درجات التشابه لتحديد الـ threshold المثالي

الهدف: الإجابة على سؤال المشرفة:
  "عندك شي مخزن متل ارقام فيها ال retrieved data يلي جاوب منها؟
   بتفيدنا لنحدد قيمة لل threshold"

ما يفعله هذا الملف:
  1. يشغّل الـ retriever (بدون LLM) على عيّنة من الأسئلة
  2. يسجّل درجات التشابه لكل وثيقة مُسترجعة
  3. يحسب: هل الوثيقة الصحيحة كانت موجودة؟ وما كان score-ها؟
  4. يقترح threshold مناسب بناءً على توزيع الـ scores

Usage:
    py -3.10 src/score_diagnostics.py                  ← تحليل كامل (342 سؤال)
    py -3.10 src/score_diagnostics.py --sample 50      ← عيّنة 50 سؤال
    py -3.10 src/score_diagnostics.py --plot            ← مع رسم بياني (يحتاج matplotlib)
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MEDHOP_FILE, OUTPUTS_DIR

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(sample=None):
    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if sample:
        data = data[:sample]
    return data


def load_bridge_cache():
    cache_path = os.path.join(OUTPUTS_DIR, "bridge_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def get_bridge_entity(record, cache):
    """استخراج الـ bridge entity من الـ cache — tries both key formats."""
    drug_name = record.get("query_drug_name", "")
    # FIX: Try both cache key formats — mechanism:: first (B4-EXP4 baseline 33.33%)
    key_v1 = f"mechanism::{drug_name}"
    key_v2 = f"2hop_mechanism::{drug_name}"
    raw = cache.get(key_v1, "") or cache.get(key_v2, "")
    if not raw:
        return ""
    # تنظيف: استخرج أول جملة كـ bridge
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE)
    return raw[:100].strip()


def doc_contains_answer(doc_text, answer_id, answer_name):
    """هل الوثيقة تذكر الإجابة الصحيحة؟"""
    text_lower = doc_text.lower()
    if answer_id and answer_id.lower() in text_lower:
        return True
    if answer_name and len(answer_name) > 3 and answer_name.lower() in text_lower:
        return True
    return False


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────

def run_diagnostics(sample=None, max_k=7, output_file=None):
    """
    يشغّل الـ retriever على كل الأسئلة ويحلّل درجات التشابه.

    Returns: dict مع إحصائيات كاملة
    """
    print("\n" + "="*60)
    print("  Score Diagnostics — Similarity Distribution Analysis")
    print("="*60)

    # Load dependencies
    from src.query_expander import expand_query
    from src.query_expander_structured import get_weighted_terms
    from src.retriever_hybrid_scored import retrieve_hybrid_scored

    data = load_data(sample)
    bridge_cache = load_bridge_cache()

    print(f"  Dataset   : {len(data)} questions")
    print(f"  Max K     : {max_k}")
    print(f"  Bridge cache: {len(bridge_cache)} entries")
    print()

    # ── Containers for analysis ──
    records_with_answer = []     # قضايا وُجد الـ answer في المُسترجَع
    records_without_answer = []  # قضايا لم يُوجَد الـ answer

    all_top1_scores = []
    all_top3_scores = []   # متوسط top-3
    found_rank = []        # الترتيب الذي ظهرت فيه الإجابة الصحيحة

    all_score_records = []  # كل النتائج مفصّلة

    for i, record in enumerate(data):
        drug_name  = record.get("query_drug_name", "")
        query      = record.get("query", "")
        supports   = record.get("supports", [])
        answer_id  = record.get("answer", "")
        answer_name= record.get("answer_name", "")

        if not supports:
            continue

        # ── Expand query ──
        exp_result     = expand_query(drug_name)
        flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
        weighted_terms = get_weighted_terms(drug_name)

        # ── Bridge entity (كـ مصطلحات منفصلة بوزن 5.0 — مطابق لـ B4-EXP4) ──
        bridge = get_bridge_entity(record, bridge_cache)
        if bridge:
            # FIX: Split bridge phrase into individual terms (same as B4-EXP4)
            # Instead of one term with the full phrase, each word gets weight=5.0
            clean = (bridge
                     .replace("inhibits", "").replace("blocks", "")
                     .replace("acts", "").replace("via", "")
                     .replace("through", ""))
            bridge_words = [w.strip() for w in clean.split() if len(w.strip()) > 2]
            for bw in bridge_words:
                weighted_terms.append({
                    "term":     bw,
                    "weight":   5.0,
                    "category": "mechanism",
                })

        # ── Run retriever (بدون حد أدنى لـ k) ──
        try:
            retrieved = retrieve_hybrid_scored(
                query=query,
                supports=supports,
                drug_name=drug_name,
                flat_terms=flat_terms,
                weighted_terms=weighted_terms,
                top_k=max_k,
            )
        except Exception as e:
            print(f"  [WARN] Failed on {record.get('id','?')}: {e}")
            continue

        # ── Analyze scores ──
        scores = [r["score"] for r in retrieved]
        medcpt = [r["medcpt_score"] for r in retrieved]
        terms  = [r["term_score"]   for r in retrieved]

        top1 = scores[0] if scores else 0.0
        top3_avg = np.mean(scores[:3]) if len(scores) >= 3 else np.mean(scores) if scores else 0.0

        all_top1_scores.append(top1)
        all_top3_scores.append(top3_avg)

        # ── Check if answer is in retrieved ──
        answer_found = False
        answer_rank  = -1
        answer_score = 0.0

        for rank_idx, r in enumerate(retrieved, 1):
            if doc_contains_answer(r["text"], answer_id, answer_name):
                answer_found = True
                answer_rank  = rank_idx
                answer_score = r["score"]
                found_rank.append(rank_idx)
                break

        # ── Save record ──
        record_info = {
            "question_id":  record.get("id", ""),
            "drug_name":    drug_name,
            "answer_id":    answer_id,
            "bridge":       bridge,
            "n_supports":   len(supports),
            "n_retrieved":  len(retrieved),
            "top1_score":   round(top1, 4),
            "top3_avg":     round(top3_avg, 4),
            "top1_medcpt":  round(medcpt[0], 4) if medcpt else 0.0,
            "top1_terms":   round(terms[0],  4) if terms  else 0.0,
            "answer_found": answer_found,
            "answer_rank":  answer_rank,
            "answer_score": round(answer_score, 4),
            "all_scores":   [round(s, 4) for s in scores],
        }

        all_score_records.append(record_info)

        if answer_found:
            records_with_answer.append(record_info)
        else:
            records_without_answer.append(record_info)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>3}/{len(data)}] processed...")

    # ── Statistics ──
    print()
    print("="*60)
    print("  RESULTS")
    print("="*60)

    total_q = len(all_score_records)
    n_found = len(records_with_answer)
    n_miss  = len(records_without_answer)
    recall  = n_found / total_q * 100 if total_q else 0

    print(f"  Total questions analyzed : {total_q}")
    print(f"  Answer found in top-{max_k}   : {n_found} ({recall:.1f}%)")
    print(f"  Answer NOT found         : {n_miss}  ({100-recall:.1f}%)")
    print()

    # Score distributions
    found_top1  = [r["top1_score"] for r in records_with_answer]
    missed_top1 = [r["top1_score"] for r in records_without_answer]

    print("  ── Top-1 Score Distribution ──")
    if found_top1:
        print(f"  When answer FOUND  : mean={np.mean(found_top1):.3f}  "
              f"median={np.median(found_top1):.3f}  "
              f"min={np.min(found_top1):.3f}  "
              f"max={np.max(found_top1):.3f}")
    if missed_top1:
        print(f"  When answer MISSED : mean={np.mean(missed_top1):.3f}  "
              f"median={np.median(missed_top1):.3f}  "
              f"min={np.min(missed_top1):.3f}  "
              f"max={np.max(missed_top1):.3f}")
    print()

    # Answer score distribution (when found)
    if records_with_answer:
        ans_scores = [r["answer_score"] for r in records_with_answer]
        print("  ── Answer Doc Score (when found) ──")
        print(f"  mean={np.mean(ans_scores):.3f}  "
              f"median={np.median(ans_scores):.3f}  "
              f"p25={np.percentile(ans_scores, 25):.3f}  "
              f"p75={np.percentile(ans_scores, 75):.3f}")
        print()

    # Rank distribution
    if found_rank:
        rank_dist = {}
        for r in found_rank:
            rank_dist[r] = rank_dist.get(r, 0) + 1
        print("  ── Answer found at rank ──")
        for rank in sorted(rank_dist.keys()):
            pct = rank_dist[rank] / n_found * 100
            bar = "█" * int(pct / 2)
            print(f"  Rank {rank}: {rank_dist[rank]:>3} ({pct:>5.1f}%) {bar}")
        print()

    # ── Threshold suggestions ──
    print("  ── Threshold Recommendations ──")
    all_s = [r["top1_score"] for r in all_score_records]
    pcts  = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
    print("  Percentile distribution of top-1 scores:")
    for p in pcts:
        print(f"  p{p:>2}: {np.percentile(all_s, p):.3f}")
    print()

    # ── CRITICAL: Check if scores are clustered near 1.0 ──
    pct_near_top = sum(1 for s in all_s if s >= 0.90) / len(all_s) * 100
    print(f"  ⚠️  Scores >= 0.90: {pct_near_top:.1f}% of questions")
    if pct_near_top > 50:
        print("  ⚠️  WARNING: Scores heavily normalized near 1.0!")
        print("     Absolute threshold-based adaptive k may not be meaningful.")
        print("     Recommended: Use SCORE GAP approach instead.")
        print()

    # المنطق المقترح للـ adaptive k
    p25 = np.percentile(all_s, 25)
    p50 = np.percentile(all_s, 50)
    p75 = np.percentile(all_s, 75)

    # ── Recall at each K ──
    print("  ── Recall@K (answer found in top-K) ──")
    for k in range(1, max_k + 1):
        found_k = sum(
            1 for r in all_score_records
            if r["answer_found"] and r["answer_rank"] <= k
        )
        pct = found_k / total_q * 100 if total_q else 0
        bar = "█" * int(pct / 2)
        print(f"  k={k}: {found_k:>3}/{total_q} ({pct:>5.1f}%) {bar}")
    print()

    # Optimal MAX_K: find the k where marginal gain < 2%
    recall_at = {}
    for k in range(1, max_k + 1):
        recall_at[k] = sum(
            1 for r in all_score_records
            if r["answer_found"] and r["answer_rank"] <= k
        ) / total_q * 100 if total_q else 0

    recommended_max_k = max_k
    for k in range(2, max_k + 1):
        gain = recall_at[k] - recall_at[k - 1]
        if gain < 2.0 and k >= 4:
            recommended_max_k = k
            break

    print(f"  ✓  Recommended MAX_K = {recommended_max_k} (marginal recall gain < 2% beyond this)")
    print(f"  ✓  Recommended MIN_K = 2  (multi-hop needs at least 2 docs)")
    print()

    # Score gap analysis
    gaps = []
    for r in all_score_records:
        s = r.get("all_scores", [])
        if len(s) >= 2:
            gaps.append(s[0] - s[1])

    if gaps:
        gap_p25 = np.percentile(gaps, 25)
        gap_p75 = np.percentile(gaps, 75)
        pct_clear = sum(1 for g in gaps if g > 0.20) / len(gaps) * 100
        print(f"  ── Score Gap Analysis ──")
        print(f"  Clear winner (gap>0.20): {pct_clear:.1f}% of questions")
        print(f"  Gap p25={gap_p25:.3f} | p75={gap_p75:.3f}")
        print(f"  ✓  Score-Gap filter (ratio=0.6) is more reliable than absolute thresholds")
        print()

    print("  ── Suggested Adaptive K Logic ──")
    if pct_near_top > 50:
        print("  [RECOMMENDED] Score-Gap Adaptive K:")
        print(f"    Step 1: Fetch MAX_K={recommended_max_k} docs always")
        print(f"    Step 2: Apply score_gap_filter(ratio=0.6) → adaptive selection")
        print(f"    Step 3: Enforce MIN_K=2 / MAX_K={recommended_max_k}")
        print()
        print("  [ALTERNATIVE] Absolute threshold (less reliable with normalized scores):")
    print(f"  HIGH_THRESHOLD   = {p75:.3f}  → k = 2  (score فوق p75)")
    print(f"  MID_THRESHOLD    = {p50:.3f}  → k = 3  (score بين p50-p75)")
    print(f"  LOW_THRESHOLD    = {p25:.3f}  → k = {min(5, recommended_max_k)}  (score بين p25-p50)")
    print(f"  VERY_LOW         = < {p25:.3f} → k = {recommended_max_k}  (score تحت p25)")
    print()

    # ── Save results ──
    results = {
        "summary": {
            "total_questions": total_q,
            "answer_found_count": n_found,
            "answer_missed_count": n_miss,
            "recall_at_k": round(recall, 2),
            "max_k_used": max_k,
        },
        "score_stats": {
            "top1_all":   {"mean": round(np.mean(all_top1_scores), 4), "median": round(np.median(all_top1_scores), 4)},
            "top1_found": {"mean": round(np.mean(found_top1), 4), "median": round(np.median(found_top1), 4)} if found_top1 else {},
            "top1_missed":{"mean": round(np.mean(missed_top1), 4), "median": round(np.median(missed_top1), 4)} if missed_top1 else {},
            "pct_scores_near_1": round(sum(1 for s in all_top1_scores if s >= 0.90) / len(all_top1_scores) * 100, 1),
        },
        "recall_at_k": {
            f"k={k}": round(sum(
                1 for r in all_score_records
                if r["answer_found"] and r["answer_rank"] <= k
            ) / total_q * 100, 2) if total_q else 0
            for k in range(1, max_k + 1)
        },
        "threshold_suggestions": {
            "recommended_approach": "score_gap_filter (ratio=0.6)" if pct_near_top > 50 else "absolute_threshold",
            "warning":             "Scores heavily normalized — use gap-based approach" if pct_near_top > 50 else None,
            "min_k":               2,
            "max_k":               recommended_max_k,
            "high_conf":           round(p75, 3),
            "mid_conf":            round(p50, 3),
            "low_conf":            round(p25, 3),
            "adaptive_k_logic": {
                f"score >= {p75:.3f}": "k = 2  (min enforced)",
                f"score >= {p50:.3f}": "k = 3",
                f"score >= {p25:.3f}": f"k = {min(5, recommended_max_k)}",
                f"score  < {p25:.3f}": f"k = {recommended_max_k}",
            },
        },
        "per_question_scores": all_score_records,
    }

    # Save
    if output_file is None:
        output_file = os.path.join(OUTPUTS_DIR, "score_diagnostics.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Results saved to: {output_file}")
    print()

    return results, p25, p50, p75


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Diagnostics — similarity distribution analysis")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to analyze (default: all)")
    parser.add_argument("--max-k",  type=int, default=7,    help="Max K to retrieve per question")
    parser.add_argument("--plot",   action="store_true",    help="Generate histogram plots (requires matplotlib)")
    args = parser.parse_args()

    results, p25, p50, p75 = run_diagnostics(sample=args.sample, max_k=args.max_k)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            scores = [r["top1_score"] for r in results["per_question_scores"]]
            found  = [r["top1_score"] for r in results["per_question_scores"] if r["answer_found"]]
            missed = [r["top1_score"] for r in results["per_question_scores"] if not r["answer_found"]]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].hist(found,  bins=20, alpha=0.7, color="green", label=f"Answer Found (n={len(found)})")
            axes[0].hist(missed, bins=20, alpha=0.7, color="red",   label=f"Answer Missed (n={len(missed)})")
            axes[0].axvline(p25, color="orange", linestyle="--", label=f"p25={p25:.3f}")
            axes[0].axvline(p50, color="blue",   linestyle="--", label=f"p50={p50:.3f}")
            axes[0].axvline(p75, color="purple", linestyle="--", label=f"p75={p75:.3f}")
            axes[0].set_title("Top-1 Score Distribution")
            axes[0].set_xlabel("Score")
            axes[0].set_ylabel("Frequency")
            axes[0].legend()

            rank_counts = {}
            for r in results["per_question_scores"]:
                if r["answer_found"]:
                    rank = r["answer_rank"]
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
            ranks = sorted(rank_counts.keys())
            axes[1].bar(ranks, [rank_counts[r] for r in ranks], color="steelblue")
            axes[1].set_title("Answer Found at Which Rank?")
            axes[1].set_xlabel("Rank")
            axes[1].set_ylabel("Count")

            plt.tight_layout()
            plot_path = os.path.join(OUTPUTS_DIR, "score_diagnostics_plot.png")
            plt.savefig(plot_path, dpi=150)
            print(f"  [OK] Plot saved to: {plot_path}")
            plt.show()
        except ImportError:
            print("  [WARN] matplotlib not installed — skipping plots")
