"""
src/inference_pipeline.py
==========================
Biomedical Multi-Hop QA Project — Baseline 1

The main orchestrator that runs the full inference pipeline.

What this module does:
  1. Loads normalized dataset from data/medhop.json
  2. Builds prompts for all 342 questions
  3. Sends each prompt to BioMistral via llm_runner
  4. Extracts clean Drug ID from each response
  5. Saves predictions incrementally (safe against crashes)
  6. Calculates Exact Match (EM) score
  7. Saves final predictions + logs to outputs/

Output files:
  outputs/baseline1_predictions.json  ← predictions + ground truth
  outputs/baseline1_logs.json         ← timing + stats + EM score

Usage:
    py -3.10 src/inference_pipeline.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    PREDICTIONS_FILE,
    LOGS_FILE,
    OUTPUTS_DIR,
    MAX_QUESTIONS,
    LOG_EVERY_N,
    OLLAMA_MODEL,
    EXP_NAME,
)
from src.prompt_builder import (
    build_prompt,
    build_prompt_cot,
    build_prompt_fewshot,
    extract_drug_id,
)

# Select prompt builder based on EXP_NAME in settings.py
PROMPT_BUILDERS = {
    "original":        build_prompt,
    "cot":             build_prompt_cot,
    "fewshot":         build_prompt_fewshot,
    "temp0":           build_prompt,        # same prompt, temperature changes in settings
    "fewshot_temp0":   build_prompt_fewshot,
}
_build_prompt_fn = PROMPT_BUILDERS.get(EXP_NAME, build_prompt)
from src.llm_runner     import get_ollama_client, check_model_available, run_inference


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────

def load_data() -> list:
    """Load normalized MedHop dataset from medhop.json."""
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        print("         Run: py -3.10 src/load_dataset.py")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if MAX_QUESTIONS is not None:
        data = data[:MAX_QUESTIONS]
        print(f"  [INFO] Limited to {MAX_QUESTIONS} questions (MAX_QUESTIONS setting)")

    print(f"  [OK]   Loaded {len(data)} questions from medhop.json")
    return data


# ─────────────────────────────────────────────
# STEP 2 — RESUME SUPPORT
# Check if a previous run was interrupted
# ─────────────────────────────────────────────

def load_existing_predictions() -> dict:
    """
    Load predictions from a previous interrupted run.
    Returns dict: { question_id: prediction_record }
    Allows resuming without re-running completed questions.
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return {}

    try:
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        done = {r["question_id"]: r for r in existing}
        print(f"  [INFO] Found {len(done)} predictions from previous run — will resume")
        return done
    except Exception:
        return {}


# ─────────────────────────────────────────────
# STEP 3 — SAVE INCREMENTALLY
# ─────────────────────────────────────────────

def save_predictions(predictions: list):
    """Save predictions list to JSON (called after every question)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def save_logs(logs: dict):
    """Save run logs and statistics to JSON."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# STEP 4 — CALCULATE EM SCORE
# ─────────────────────────────────────────────

def calculate_em(predictions: list) -> dict:
    """
    Calculate Exact Match (EM) score.

    EM = 1 if prediction exactly matches answer (case-insensitive)
    EM = 0 otherwise

    Returns dict with EM score and breakdown.
    """
    total    = len(predictions)
    correct  = 0
    failed   = 0
    skipped  = 0

    for pred in predictions:
        if not pred.get("success", False):
            failed += 1
            continue

        prediction = pred.get("prediction", "").strip().upper()
        answer     = pred.get("answer", "").strip().upper()

        if not prediction:
            skipped += 1
            continue

        if prediction == answer:
            correct += 1

    answered = total - failed - skipped
    em_score = (correct / answered * 100) if answered > 0 else 0.0

    return {
        "total":    total,
        "correct":  correct,
        "wrong":    answered - correct,
        "failed":   failed,
        "skipped":  skipped,
        "answered": answered,
        "em_score": round(em_score, 2),
    }


# ─────────────────────────────────────────────
# STEP 5 — PRINT PROGRESS SUMMARY
# ─────────────────────────────────────────────

def print_progress(i: int, total: int, record: dict,
                   prediction: str, is_correct: bool,
                   inference_time: float, correct_count: int,
                   start_time: float):
    """Print a progress line every LOG_EVERY_N questions."""

    elapsed   = time.time() - start_time
    avg_time  = elapsed / (i + 1)
    remaining = avg_time * (total - i - 1)
    current_em = correct_count / (i + 1) * 100

    status = "✅" if is_correct else "❌"

    print(
        f"  [{i+1:>3}/{total}] {status} "
        f"{record['id']:<12} | "
        f"pred: {prediction:<10} | "
        f"ans: {record['answer']:<10} | "
        f"{inference_time:.1f}s | "
        f"EM so far: {current_em:.1f}% | "
        f"ETA: {remaining/60:.1f}min"
    )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    """
    Full Baseline 1 inference pipeline.
    Runs all questions through BioMistral and saves results.
    """

    print("\n" + "=" * 60)
    print("  Baseline 1 — Full Inference Pipeline")
    print("=" * 60)
    print(f"  Experiment: {EXP_NAME}")
    print(f"  Model   : {OLLAMA_MODEL}")
    print(f"  Dataset : {MEDHOP_FILE}")
    print(f"  Output  : {PREDICTIONS_FILE}")
    print()

    pipeline_start = time.time()

    # ── Step 1: Connect to Ollama ──
    print("--- Step 1: Connecting to Ollama ---")
    client = get_ollama_client()
    if not check_model_available(client):
        sys.exit(1)
    print(f"  [OK]   Connected — model '{OLLAMA_MODEL}' ready\n")

    # ── Step 2: Load dataset ──
    print("--- Step 2: Loading Dataset ---")
    data = load_data()
    total = len(data)
    print()

    # ── Step 3: Check for previous run ──
    print("--- Step 3: Checking Previous Progress ---")
    existing = load_existing_predictions()
    print()

    # ── Step 4: Run inference ──
    print("--- Step 4: Running Inference ---")
    print(f"  Total questions : {total}")
    print(f"  Already done    : {len(existing)}")
    print(f"  Remaining       : {total - len(existing)}")
    print()

    predictions  = list(existing.values())
    correct_count = sum(
        1 for p in predictions
        if p.get("prediction", "").upper() == p.get("answer", "").upper()
        and p.get("success", False)
    )
    failed_count = 0
    inference_times = []

    for i, record in enumerate(data):
        question_id = record["id"]

        # Skip if already done in previous run
        if question_id in existing:
            continue

        # Build prompt using selected experiment function
        prompt = _build_prompt_fn(record)

        # Run inference
        result = run_inference(client, prompt, question_id)

        # Extract Drug ID from response
        if result["success"]:
            prediction = extract_drug_id(
                result["raw_response"],
                record["candidates"]
            )
        else:
            prediction = ""
            failed_count += 1

        # Check correctness
        is_correct = (
            prediction.strip().upper() == record["answer"].strip().upper()
            and result["success"]
        )
        if is_correct:
            correct_count += 1

        if result["inference_time"] > 0:
            inference_times.append(result["inference_time"])

        # Build prediction record
        pred_record = {
            "question_id":    question_id,
            "query":          record["query"],
            "query_drug_id":  record["query_drug_id"],
            "query_drug_name":record["query_drug_name"],
            "candidates":     record["candidates"],
            "prediction":     prediction,
            "raw_response":   result["raw_response"],
            "answer":         record["answer"],
            "answer_name":    record.get("answer_name", record["answer"]),
            "is_correct":     is_correct,
            "inference_time": result["inference_time"],
            "success":        result["success"],
            "error":          result["error"],
        }

        predictions.append(pred_record)
        existing[question_id] = pred_record

        # Save incrementally after every question
        save_predictions(predictions)

        # Print progress every LOG_EVERY_N questions
        if (i + 1) % LOG_EVERY_N == 0 or (i + 1) == total:
            print_progress(
                i, total, record,
                prediction, is_correct,
                result["inference_time"],
                correct_count,
                pipeline_start
            )

    # ── Step 5: Calculate final EM ──
    print()
    print("--- Step 5: Calculating Results ---")
    em_results = calculate_em(predictions)

    total_time  = time.time() - pipeline_start
    avg_time    = (sum(inference_times) / len(inference_times)) if inference_times else 0

    # ── Step 6: Save logs ──
    logs = {
        "run_info": {
            "model":         OLLAMA_MODEL,
            "dataset":       MEDHOP_FILE,
            "total_time_min": round(total_time / 60, 2),
            "avg_time_sec":   round(avg_time, 3),
        },
        "em_results": em_results,
        "per_question_times": [
            {
                "question_id":    p["question_id"],
                "inference_time": p["inference_time"],
                "is_correct":     p["is_correct"],
                "success":        p["success"],
            }
            for p in predictions
        ],
    }
    save_logs(logs)

    # ── Step 7: Print final summary ──
    print()
    print("=" * 60)
    print("  BASELINE 1 — FINAL RESULTS")
    print("=" * 60)
    print(f"  Total questions : {em_results['total']}")
    print(f"  Answered        : {em_results['answered']}")
    print(f"  Correct (EM=1)  : {em_results['correct']}")
    print(f"  Wrong   (EM=0)  : {em_results['wrong']}")
    print(f"  Failed          : {em_results['failed']}")
    print()
    print(f"  ✅  Exact Match (EM) Score : {em_results['em_score']}%")
    print()
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Avg per question: {avg_time:.2f} seconds")
    print()
    print(f"  Predictions saved : {PREDICTIONS_FILE}")
    print(f"  Logs saved        : {LOGS_FILE}")
    print("=" * 60)
    print()

    return em_results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()