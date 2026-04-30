"""
src/utils.py
=============
Biomedical Multi-Hop QA Project

Utility functions for evaluation, reporting, and analysis.

What this module does:
  1. Calculates Exact Match (EM) score
  2. Calculates Recall@K (retrieval quality)
  3. Generates detailed evaluation report
  4. Analyzes error patterns
  5. Prints formatted summary tables

Usage:
    py -3.10 src/utils.py
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    PREDICTIONS_FILE,
    LOGS_FILE,
    OUTPUTS_DIR,
    PREDICTIONS_FILE_B2,
    PREDICTIONS_FILE_B3,
)


# ─────────────────────────────────────────────
# EXACT MATCH EVALUATION
# ─────────────────────────────────────────────

def exact_match(prediction: str, answer: str) -> int:
    """Calculate Exact Match for a single prediction."""
    if not prediction or not answer:
        return 0
    return int(prediction.strip().upper() == answer.strip().upper())


def calculate_em_score(predictions: list) -> dict:
    """Calculate EM score over full predictions list."""
    total   = len(predictions)
    correct = 0
    wrong   = 0
    failed  = 0
    skipped = 0

    for pred in predictions:
        if not pred.get("success", False):
            failed += 1
            continue

        prediction_val = pred.get("prediction", "").strip()
        answer_val     = pred.get("answer", "").strip()

        if not prediction_val:
            skipped += 1
            continue

        if exact_match(prediction_val, answer_val):
            correct += 1
        else:
            wrong += 1

    answered = correct + wrong
    em_score = round((correct / answered * 100), 2) if answered > 0 else 0.0
    random_baseline = round((1 / 8.6) * 100, 2)

    return {
        "total":            total,
        "answered":         answered,
        "correct":          correct,
        "wrong":            wrong,
        "failed":           failed,
        "skipped":          skipped,
        "em_score":         em_score,
        "random_baseline":  random_baseline,
        "above_random":     round(em_score - random_baseline, 2),
    }


# ─────────────────────────────────────────────
# RECALL@K EVALUATION
# ─────────────────────────────────────────────

def check_answer_in_supports(answer_name: str, supports: list) -> bool:
    """
    Check if answer drug name appears in supporting texts.
    
    Args:
        answer_name: Drug name to search for (e.g., "Tetrabenazine")
        supports: List of supporting text strings
    
    Returns:
        True if answer name found in any support text
    """
    if not answer_name or not supports:
        return False
    
    answer_lower = answer_name.lower()
    
    for support in supports:
        if not support:
            continue
        if isinstance(support, dict):
            text = support.get("text", "")
        else:
            text = str(support)
        
        if answer_lower in text.lower():
            return True
    
    return False


def calculate_recall_at_k(predictions: list, supports_key: str = "retrieved_supports") -> dict:
    """
    Calculate Recall@K - whether the answer appears in retrieved texts.
    
    Args:
        predictions: List of prediction dicts
        supports_key: Key for stored retrieved supports ("retrieved_supports")
    
    Returns:
        Dict with Recall@K statistics
    """
    total = 0
    found = 0
    not_found = 0
    no_supports = 0
    
    for pred in predictions:
        if not pred.get("success", False):
            continue
        
        total += 1
        answer_name = pred.get("answer_name", "")
        supports = pred.get(supports_key, [])
        
        if not supports:
            no_supports += 1
            continue
        
        if check_answer_in_supports(answer_name, supports):
            found += 1
        else:
            not_found += 1
    
    recall_score = round((found / total * 100), 2) if total > 0 else 0.0
    
    return {
        "total_questions": total,
        "answer_found":    found,
        "answer_not_found": not_found,
        "no_supports":     no_supports,
        "recall_at_k":     recall_score,
    }


def calculate_recall_from_original_data(predictions: list, original_data_path: str) -> dict:
    """
    Calculate Recall@K by loading original data and checking supports.
    
    This is used when retrieved supports weren't stored in predictions.
    """
    # Load original medhop.json
    with open(original_data_path, encoding="utf-8") as f:
        original_data = json.load(f)
    
    # Build lookup by question_id
    data_lookup = {r["id"]: r for r in original_data}
    
    total = 0
    found = 0
    not_found = 0
    
    for pred in predictions:
        if not pred.get("success", False):
            continue
        
        qid = pred.get("question_id", "")
        record = data_lookup.get(qid)
        
        if not record:
            continue
        
        total += 1
        answer_name = pred.get("answer_name", "")
        supports = record.get("supports", [])
        
        if check_answer_in_supports(answer_name, supports):
            found += 1
        else:
            not_found += 1
    
    recall_score = round((found / total * 100), 2) if total > 0 else 0.0
    
    return {
        "total_questions": total,
        "answer_found":    found,
        "answer_not_found": not_found,
        "recall_at_k":     recall_score,
    }


# ─────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────

def analyze_errors(predictions: list) -> dict:
    """Analyze patterns in wrong predictions."""
    wrong_preds    = []
    missed_answers = []
    valid_but_wrong = 0

    for pred in predictions:
        if not pred.get("success", False):
            continue

        prediction_val = pred.get("prediction", "")
        answer_val     = pred.get("answer", "")
        candidates     = pred.get("candidates", [])

        if exact_match(prediction_val, answer_val):
            continue

        wrong_preds.append(prediction_val)
        missed_answers.append(answer_val)

        if prediction_val.upper() in [c.upper() for c in candidates]:
            valid_but_wrong += 1

    top_wrong   = Counter(wrong_preds).most_common(10)
    top_missed  = Counter(missed_answers).most_common(10)

    return {
        "total_wrong":         len(wrong_preds),
        "valid_but_wrong":     valid_but_wrong,
        "valid_but_wrong_pct": round(valid_but_wrong / len(wrong_preds) * 100, 1) if wrong_preds else 0,
        "top_wrong_predictions": [
            {"drug_id": drug, "count": count}
            for drug, count in top_wrong
        ],
        "top_missed_answers": [
            {"drug_id": drug, "count": count}
            for drug, count in top_missed
        ],
    }


# ─────────────────────────────────────────────
# TIMING ANALYSIS
# ─────────────────────────────────────────────

def analyze_timing(predictions: list) -> dict:
    """Analyze inference timing statistics."""
    times = [
        p["inference_time"]
        for p in predictions
        if p.get("success") and p.get("inference_time", 0) > 0
    ]

    if not times:
        return {}

    times_sorted = sorted(times)
    n = len(times_sorted)

    return {
        "total_questions":  n,
        "total_time_sec":   round(sum(times), 2),
        "total_time_min":   round(sum(times) / 60, 2),
        "avg_time_sec":     round(sum(times) / n, 3),
        "min_time_sec":     round(min(times), 3),
        "max_time_sec":     round(max(times), 3),
        "median_time_sec":  round(times_sorted[n // 2], 3),
        "p95_time_sec":     round(times_sorted[int(n * 0.95)], 3),
    }


# ─────────────────────────────────────────────
# PRINT FORMATTED REPORT
# ─────────────────────────────────────────────

def print_evaluation_report(
    predictions: list, 
    stage: str = "Baseline 1",
    original_data_path: str = None
):
    """Print a full formatted evaluation report to console."""

    em      = calculate_em_score(predictions)
    errors  = analyze_errors(predictions)
    timing  = analyze_timing(predictions)

    width = 60
    print()
    print("=" * width)
    print(f"  EVALUATION REPORT — {stage}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * width)

    # ── EM Results ──
    print()
    print("  EXACT MATCH RESULTS")
    print("  " + "-" * (width - 2))
    print(f"  Total questions    : {em['total']}")
    print(f"  Answered           : {em['answered']}")
    print(f"  Correct  (EM = 1)  : {em['correct']}")
    print(f"  Wrong    (EM = 0)  : {em['wrong']}")
    print(f"  Failed             : {em['failed']}")
    print()
    print(f"  EM Score           : {em['em_score']}%")
    print(f"  Random Baseline    : {em['random_baseline']}%")
    print(f"  Above Random       : +{em['above_random']}%")

    # Visual bar
    bar_len  = 40
    filled   = int(em['em_score'] / 100 * bar_len)
    rand_pos = int(em['random_baseline'] / 100 * bar_len)
    bar      = ["-"] * bar_len
    for j in range(filled):
        bar[j] = "#"
    if rand_pos < bar_len:
        bar[rand_pos] = "|"
    print()
    print(f"  [{''.join(bar)}] {em['em_score']}%")
    print(f"   {'':>{rand_pos}}^ random ({em['random_baseline']}%)")

    # ── Recall@K (if retrieval was used) ──
    if original_data_path:
        recall = calculate_recall_from_original_data(predictions, original_data_path)
        print()
        print("  RECALL@K (Retrieval Quality)")
        print("  " + "-" * (width - 2))
        print(f"  Total questions    : {recall['total_questions']}")
        print(f"  Answer found       : {recall['answer_found']}")
        print(f"  Answer NOT found   : {recall['answer_not_found']}")
        print()
        print(f"  Recall@K Score     : {recall['recall_at_k']}%")
        
        # Interpretation
        if recall['recall_at_k'] > 80:
            quality = "GOOD"
        elif recall['recall_at_k'] > 50:
            quality = "MODERATE"
        else:
            quality = "POOR"
        print(f"  Retrieval Quality  : {quality}")
        print()
        print(f"  Note: Recall@K measures if the correct answer")
        print(f"        appears in the retrieved supporting texts.")

    # ── Error Analysis ──
    print()
    print("  ERROR ANALYSIS")
    print("  " + "-" * (width - 2))
    print(f"  Total wrong        : {errors['total_wrong']}")
    print(f"  Valid but wrong    : {errors['valid_but_wrong']} "
          f"({errors['valid_but_wrong_pct']}%)")
    print()

    print("  Top 5 wrong predictions:")
    for item in errors["top_wrong_predictions"][:5]:
        print(f"    {item['drug_id']:<12} x {item['count']} times")

    print()
    print("  Top 5 missed answers:")
    for item in errors["top_missed_answers"][:5]:
        print(f"    {item['drug_id']:<12} missed {item['count']} times")

    # ── Timing ──
    if timing:
        print()
        print("  TIMING")
        print("  " + "-" * (width - 2))
        print(f"  Total time         : {timing['total_time_min']} min")
        print(f"  Avg per question   : {timing['avg_time_sec']} sec")
        print(f"  Min / Max          : {timing['min_time_sec']}s / {timing['max_time_sec']}s")
        print(f"  Median             : {timing['median_time_sec']}s")

    print()
    print("=" * width)
    print()


# ─────────────────────────────────────────────
# SAVE EVALUATION REPORT
# ─────────────────────────────────────────────

def save_evaluation_report(
    predictions: list, 
    stage: str = "baseline1",
    original_data_path: str = None
):
    """Save full evaluation report as JSON."""
    report = {
        "stage":          stage,
        "generated_at":   datetime.now().isoformat(),
        "em_results":     calculate_em_score(predictions),
        "error_analysis": analyze_errors(predictions),
        "timing":         analyze_timing(predictions),
    }
    
    # Add Recall@K if retrieval was used
    if original_data_path:
        report["recall_at_k"] = calculate_recall_from_original_data(
            predictions, original_data_path
        )

    report_path = os.path.join(OUTPUTS_DIR, f"{stage}_evaluation_report.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  [OK]   Evaluation report saved -> {report_path}")
    return report_path


# ─────────────────────────────────────────────
# COMPARISON REPORT
# ─────────────────────────────────────────────

def print_comparison_report(baseline: str = "all"):
    """Print comparison between all baselines."""
    
    print()
    print("=" * 60)
    print("  COMPARISON REPORT — All Baselines")
    print("=" * 60)
    
    results = []
    
    # Baseline 1
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            b1_preds = json.load(f)
        b1_em = calculate_em_score(b1_preds)
        results.append({
            "stage": "Baseline 1 (Direct LLM)",
            "em": b1_em["em_score"],
            "recall": None,
            "correct": b1_em["correct"],
            "total": b1_em["answered"],
        })
    
    # Baseline 2
    if os.path.exists(PREDICTIONS_FILE_B2):
        with open(PREDICTIONS_FILE_B2, encoding="utf-8") as f:
            b2_preds = json.load(f)
        b2_em = calculate_em_score(b2_preds)
        results.append({
            "stage": "Baseline 2 (BM25 RAG)",
            "em": b2_em["em_score"],
            "recall": None,
            "correct": b2_em["correct"],
            "total": b2_em["answered"],
        })
    
    # Baseline 3
    if os.path.exists(PREDICTIONS_FILE_B3):
        with open(PREDICTIONS_FILE_B3, encoding="utf-8") as f:
            b3_preds = json.load(f)
        b3_em = calculate_em_score(b3_preds)
        results.append({
            "stage": "Baseline 3 (MedCPT RAG)",
            "em": b3_em["em_score"],
            "recall": None,
            "correct": b3_em["correct"],
            "total": b3_em["answered"],
        })
    
    if not results:
        print("  No predictions found. Run inference pipelines first.")
        return
    
    # Print table
    print()
    print("  " + "-" * 56)
    print(f"  {'Stage':<25} {'EM':>8} {'Correct':>10} {'Total':>8}")
    print("  " + "-" * 56)
    
    for r in results:
        print(f"  {r['stage']:<25} {r['em']:>7.2f}% {r['correct']:>10} {r['total']:>8}")
    
    print("  " + "-" * 56)
    print()
    
    # Show improvement
    if len(results) > 1:
        best = max(results, key=lambda x: x["em"])
        print(f"  Best performing: {best['stage']} ({best['em']:.2f}% EM)")


# ─────────────────────────────────────────────
# LOAD PREDICTIONS HELPER
# ─────────────────────────────────────────────

def load_predictions(path: str = None) -> list:
    """Load predictions from JSON file."""
    path = path or PREDICTIONS_FILE

    if not os.path.exists(path):
        print(f"  [FAIL] Predictions file not found: {path}")
        print("         Run inference_pipeline.py first.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  [OK]   Loaded {len(data)} predictions from {path}")
    return data


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Utils — Evaluation Report")
    print("=" * 60)

    # Load predictions
    print("\n--- Loading Predictions ---")
    predictions = load_predictions()

    # Print full report
    print_evaluation_report(predictions, stage="Baseline 1")

    # Save report
    print("--- Saving Evaluation Report ---")
    save_evaluation_report(predictions, stage="baseline1")

    print("\n  utils.py working correctly.\n")


if __name__ == "__main__":
    main()