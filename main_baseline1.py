"""
main_baseline1.py
==================
Biomedical Multi-Hop QA Project — Baseline 1

Single entry point for the complete Baseline 1 pipeline.

What this script does (in order):
  1. Verifies environment (Ollama, model, dataset)
  2. Loads and validates MedHop dataset
  3. Runs full inference on all 342 questions
  4. Calculates Exact Match (EM) score
  5. Generates and saves evaluation report
  6. Prints final summary

This is the ONLY file you need to run for a complete Baseline 1 evaluation.

Usage:
    py -3.10 main_baseline1.py                  ← full run
    py -3.10 main_baseline1.py --test           ← test on 5 questions only
    py -3.10 main_baseline1.py --report-only    ← re-generate report from existing predictions
    py -3.10 main_baseline1.py --reset          ← delete previous predictions and start fresh
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    MEDHOP_FILE,
    PREDICTIONS_FILE,
    LOGS_FILE,
    OUTPUTS_DIR,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    ACTIVE_DATASET,
    DRUGBANK_VOCAB,
)


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline 1 — Direct LLM QA with BioMistral"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on first 5 questions only (quick test)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip inference, re-generate report from existing predictions"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete previous predictions and start fresh"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Run on first N questions (e.g. --questions 50)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────

def preflight_checks() -> bool:
    """
    Verify all prerequisites before running inference.
    Returns True if all checks pass.
    """
    print("--- Pre-flight Checks ---")
    all_ok = True

    # Check 1: Ollama running
    try:
        import urllib.request
        urllib.request.urlopen(OLLAMA_HOST, timeout=3)
        print(f"  [OK]   Ollama running at {OLLAMA_HOST}")
    except Exception:
        print(f"  [FAIL] Ollama not running.")
        print(f"         Fix: open a terminal and run: ollama serve")
        all_ok = False

    # Check 2: Model available
    try:
        import ollama as ollama_lib
        client = ollama_lib.Client(host=OLLAMA_HOST)
        models = client.list()
        model_names = [m.model for m in models.models]
        found = any(OLLAMA_MODEL in name for name in model_names)
        if found:
            print(f"  [OK]   Model '{OLLAMA_MODEL}' registered in Ollama")
        else:
            print(f"  [FAIL] Model '{OLLAMA_MODEL}' not found.")
            print(f"         Fix: run EnvironmentSetup.py")
            all_ok = False
    except Exception as e:
        print(f"  [WARN] Could not verify model: {e}")

    # Check 3: Dataset files exist
    if os.path.exists(ACTIVE_DATASET):
        print(f"  [OK]   MedHop dataset found: {os.path.basename(ACTIVE_DATASET)}")
    else:
        print(f"  [FAIL] MedHop dataset not found: {ACTIVE_DATASET}")
        all_ok = False

    if os.path.exists(DRUGBANK_VOCAB):
        print(f"  [OK]   DrugBank vocab found")
    else:
        print(f"  [WARN] DrugBank vocab not found — drug names may show as IDs")

    # Check 4: medhop.json (processed dataset)
    if os.path.exists(MEDHOP_FILE):
        with open(MEDHOP_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [OK]   medhop.json ready — {len(data)} questions")
    else:
        print(f"  [WARN] medhop.json not found — will run load_dataset.py first")

    # Check 5: outputs dir
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"  [OK]   Outputs directory ready")

    return all_ok


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────

def run_load_dataset():
    """Run load_dataset.py to prepare medhop.json."""
    print("\n--- Step 1: Preparing Dataset ---")

    if os.path.exists(MEDHOP_FILE):
        with open(MEDHOP_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [OK]   medhop.json already exists — {len(data)} questions")
        return data

    # Need to run load_dataset
    print("  [INFO] medhop.json not found — loading from source...")
    from src.load_dataset import load_drugbank_vocab, load_medhop, save_dataset
    from config.settings  import ACTIVE_DATASET, DRUGBANK_VOCAB, MAX_QUESTIONS

    vocab = load_drugbank_vocab(DRUGBANK_VOCAB)
    data  = load_medhop(ACTIVE_DATASET, vocab, MAX_QUESTIONS)
    save_dataset(data, MEDHOP_FILE)
    return data


# ─────────────────────────────────────────────
# STEP 2 — RUN INFERENCE
# ─────────────────────────────────────────────

def run_inference_step(data: list, max_questions: int = None):
    """Run full inference pipeline."""
    print("\n--- Step 2: Running Inference ---")

    # Apply question limit if set
    if max_questions is not None:
        data = data[:max_questions]
        print(f"  [INFO] Limited to {max_questions} questions")

    # Import and run pipeline
    from src.inference_pipeline import (
        load_existing_predictions,
        save_predictions,
        save_logs,
        calculate_em,
        print_progress,
    )
    from src.prompt_builder import build_prompt, extract_drug_id
    from src.llm_runner     import get_ollama_client, run_inference

    client   = get_ollama_client()
    existing = load_existing_predictions()

    predictions   = list(existing.values())
    correct_count = sum(
        1 for p in predictions
        if p.get("prediction", "").upper() == p.get("answer", "").upper()
        and p.get("success", False)
    )

    total          = len(data)
    failed_count   = 0
    inference_times = []
    pipeline_start = time.time()

    remaining = total - len(existing)
    print(f"  Total : {total} | Done: {len(existing)} | Remaining: {remaining}")
    print()

    for i, record in enumerate(data):
        question_id = record["id"]

        if question_id in existing:
            continue

        prompt = build_prompt(record)
        result = run_inference(client, prompt, question_id)

        if result["success"]:
            prediction = extract_drug_id(result["raw_response"], record["candidates"])
        else:
            prediction = ""
            failed_count += 1

        is_correct = (
            prediction.strip().upper() == record["answer"].strip().upper()
            and result["success"]
        )
        if is_correct:
            correct_count += 1

        if result["inference_time"] > 0:
            inference_times.append(result["inference_time"])

        pred_record = {
            "question_id":     question_id,
            "query":           record["query"],
            "query_drug_id":   record["query_drug_id"],
            "query_drug_name": record["query_drug_name"],
            "candidates":      record["candidates"],
            "prediction":      prediction,
            "raw_response":    result["raw_response"],
            "answer":          record["answer"],
            "answer_name":     record.get("answer_name", record["answer"]),
            "is_correct":      is_correct,
            "inference_time":  result["inference_time"],
            "success":         result["success"],
            "error":           result["error"],
        }

        predictions.append(pred_record)
        existing[question_id] = pred_record
        save_predictions(predictions)

        from config.settings import LOG_EVERY_N
        if (i + 1) % LOG_EVERY_N == 0 or (i + 1) == total:
            print_progress(
                i, total, record, prediction, is_correct,
                result["inference_time"], correct_count, pipeline_start
            )

    # Save logs
    total_time = time.time() - pipeline_start
    avg_time   = sum(inference_times) / len(inference_times) if inference_times else 0

    logs = {
        "run_info": {
            "model":           OLLAMA_MODEL,
            "dataset":         MEDHOP_FILE,
            "total_time_min":  round(total_time / 60, 2),
            "avg_time_sec":    round(avg_time, 3),
            "timestamp":       datetime.now().isoformat(),
        },
        "em_results":          calculate_em(predictions),
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

    return predictions


# ─────────────────────────────────────────────
# STEP 3 — EVALUATE AND REPORT
# ─────────────────────────────────────────────

def run_evaluation(predictions: list):
    """Run evaluation and generate report."""
    print("\n--- Step 3: Evaluation & Report ---")

    from src.utils import (
        print_evaluation_report,
        save_evaluation_report,
    )

    print_evaluation_report(predictions, stage="Baseline 1")
    save_evaluation_report(predictions, stage="baseline1")


# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────

def reset_predictions():
    """Delete previous prediction files."""
    files = [PREDICTIONS_FILE, LOGS_FILE]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  [OK]   Deleted: {f}")
    print("  [OK]   Reset complete — ready for fresh run")


# ─────────────────────────────────────────────
# PRINT BANNER
# ─────────────────────────────────────────────

def print_banner(mode: str):
    print()
    print("╔" + "═" * 58 + "╗")
    print("║   Biomedical Multi-Hop QA — Baseline 1" + " " * 19 + "║")
    print("║   Direct LLM QA with BioMistral-7B-GGUF" + " " * 17 + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║   Mode    : {mode:<46}║")
    print(f"║   Model   : {OLLAMA_MODEL:<46}║")
    print(f"║   Time    : {datetime.now().strftime('%Y-%m-%d %H:%M'):<46}║")
    print("╚" + "═" * 58 + "╝")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine mode
    if args.reset:
        mode = "RESET"
    elif args.report_only:
        mode = "REPORT ONLY"
    elif args.test:
        mode = "TEST (5 questions)"
    elif args.questions:
        mode = f"PARTIAL ({args.questions} questions)"
    else:
        mode = "FULL RUN (342 questions)"

    print_banner(mode)
    start_time = time.time()

    # ── RESET mode ──
    if args.reset:
        print("--- Resetting Previous Run ---")
        reset_predictions()
        print("\n  Run again without --reset to start fresh.\n")
        return

    # ── REPORT ONLY mode ──
    if args.report_only:
        if not os.path.exists(PREDICTIONS_FILE):
            print("  [FAIL] No predictions found. Run without --report-only first.")
            sys.exit(1)
        with open(PREDICTIONS_FILE, encoding="utf-8") as f:
            predictions = json.load(f)
        print(f"  [OK]   Loaded {len(predictions)} existing predictions")
        run_evaluation(predictions)
        return

    # ── FULL / TEST / PARTIAL mode ──

    # Pre-flight checks
    if not preflight_checks():
        print("\n  [FAIL] Pre-flight checks failed. Fix issues above and retry.\n")
        sys.exit(1)

    # Step 1: Load dataset
    data = run_load_dataset()

    # Apply question limit
    max_questions = None
    if args.test:
        max_questions = 5
    elif args.questions:
        max_questions = args.questions

    # Step 2: Run inference
    predictions = run_inference_step(data, max_questions)

    # Step 3: Evaluate
    run_evaluation(predictions)

    # Final timing
    total_time = time.time() - start_time
    print(f"  Total pipeline time: {total_time/60:.1f} minutes")
    print()
    print("  Output files:")
    print(f"    📄 {PREDICTIONS_FILE}")
    print(f"    📄 {LOGS_FILE}")
    report_path = os.path.join(OUTPUTS_DIR, "baseline1_evaluation_report.json")
    print(f"    📄 {report_path}")
    print()
    print("  ✅  Baseline 1 complete!")
    print()


if __name__ == "__main__":
    main()
