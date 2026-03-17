"""
src/utils.py
=============
Biomedical Multi-Hop QA Project — Baseline 1

Utility functions for evaluation, reporting, and analysis.

What this module does:
  1. Calculates Exact Match (EM) score
  2. Generates detailed evaluation report
  3. Analyzes error patterns
  4. Prints formatted summary tables
  5. Exports results to readable formats

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
)


# ─────────────────────────────────────────────
# EXACT MATCH EVALUATION
# ─────────────────────────────────────────────

def exact_match(prediction: str, answer: str) -> int:
    """
    Calculate Exact Match for a single prediction.

    Returns:
        1 if prediction matches answer (case-insensitive, stripped)
        0 otherwise
    """
    if not prediction or not answer:
        return 0
    return int(prediction.strip().upper() == answer.strip().upper())


def calculate_em_score(predictions: list) -> dict:
    """
    Calculate EM score over full predictions list.

    Args:
        predictions: List of prediction dicts from inference_pipeline

    Returns:
        Dict with full EM breakdown
    """
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
    random_baseline = round((1 / 8.6) * 100, 2)  # avg 8.6 candidates per question

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
# ERROR ANALYSIS
# ─────────────────────────────────────────────

def analyze_errors(predictions: list) -> dict:
    """
    Analyze patterns in wrong predictions.

    Returns dict with:
    - Most commonly predicted wrong drugs
    - Most commonly missed correct answers
    - Questions where model was close (predicted a valid candidate)
    """
    wrong_preds    = []
    missed_answers = []
    valid_but_wrong = 0  # predicted a candidate but wrong one

    for pred in predictions:
        if not pred.get("success", False):
            continue

        prediction_val = pred.get("prediction", "")
        answer_val     = pred.get("answer", "")
        candidates     = pred.get("candidates", [])

        if exact_match(prediction_val, answer_val):
            continue  # correct, skip

        wrong_preds.append(prediction_val)
        missed_answers.append(answer_val)

        # Check if prediction was at least a valid candidate
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

def print_evaluation_report(predictions: list, stage: str = "Baseline 1"):
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
    print("  📊 EXACT MATCH RESULTS")
    print("  " + "─" * (width - 2))
    print(f"  Total questions    : {em['total']}")
    print(f"  Answered           : {em['answered']}")
    print(f"  Correct  (EM = 1)  : {em['correct']}")
    print(f"  Wrong    (EM = 0)  : {em['wrong']}")
    print(f"  Failed             : {em['failed']}")
    print()
    print(f"  EM Score           : {em['em_score']}%")
    print(f"  Random Baseline    : {em['random_baseline']}%  (1/avg_candidates)")
    print(f"  Above Random       : +{em['above_random']}%")

    # Visual bar
    bar_len  = 40
    filled   = int(em['em_score'] / 100 * bar_len)
    rand_pos = int(em['random_baseline'] / 100 * bar_len)
    bar      = ["─"] * bar_len
    for j in range(filled):
        bar[j] = "█"
    if rand_pos < bar_len:
        bar[rand_pos] = "│"
    print()
    print(f"  [{''.join(bar)}] {em['em_score']}%")
    print(f"   {'':>{rand_pos}}↑ random ({em['random_baseline']}%)")

    # ── Error Analysis ──
    print()
    print("  🔍 ERROR ANALYSIS")
    print("  " + "─" * (width - 2))
    print(f"  Total wrong        : {errors['total_wrong']}")
    print(f"  Valid but wrong    : {errors['valid_but_wrong']} "
          f"({errors['valid_but_wrong_pct']}%) "
          f"← predicted a candidate, but wrong one")
    print()

    print("  Top 5 wrong predictions (model keeps predicting these):")
    for item in errors["top_wrong_predictions"][:5]:
        print(f"    {item['drug_id']:<12} × {item['count']} times")

    print()
    print("  Top 5 missed answers (model never gets these right):")
    for item in errors["top_missed_answers"][:5]:
        print(f"    {item['drug_id']:<12} missed {item['count']} times")

    # ── Timing ──
    if timing:
        print()
        print("  ⏱️  TIMING")
        print("  " + "─" * (width - 2))
        print(f"  Total time         : {timing['total_time_min']} min")
        print(f"  Avg per question   : {timing['avg_time_sec']} sec")
        print(f"  Min / Max          : {timing['min_time_sec']}s / {timing['max_time_sec']}s")
        print(f"  Median             : {timing['median_time_sec']}s")
        print(f"  95th percentile    : {timing['p95_time_sec']}s")

    print()
    print("=" * width)
    print()


# ─────────────────────────────────────────────
# SAVE FULL EVALUATION REPORT TO FILE
# ─────────────────────────────────────────────

def save_evaluation_report(predictions: list, stage: str = "baseline1"):
    """Save full evaluation report as JSON."""
    report = {
        "stage":          stage,
        "generated_at":   datetime.now().isoformat(),
        "em_results":     calculate_em_score(predictions),
        "error_analysis": analyze_errors(predictions),
        "timing":         analyze_timing(predictions),
    }

    report_path = os.path.join(OUTPUTS_DIR, f"{stage}_evaluation_report.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  [OK]   Evaluation report saved → {report_path}")
    return report_path


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

    print("\n  ✅  utils.py working correctly.")
    print("  Next step: main_baseline1.py\n")


if __name__ == "__main__":
    main()
