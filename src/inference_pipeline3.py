"""
src/inference_pipeline3.py
===========================
Biomedical Multi-Hop QA Project — Baseline 3

The main orchestrator that runs the full inference pipeline 
with MedCPT semantic retrieval.

Usage:
    py -3.10 src/inference_pipeline3.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    PREDICTIONS_FILE_B3,
    LOGS_FILE_B3,
    OUTPUTS_DIR,
    MAX_QUESTIONS,
    LOG_EVERY_N,
    OLLAMA_MODEL_NAME,  # ✅ الاسم الصحيح
    MEDCPT_TOP_K,
)
from src.prompt_builder import build_prompt_with_context, extract_drug_id
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.retriever_semantic import retrieve_semantic, check_answer_in_retrieved


# ─────────────────────────────────────────────
# LOAD DATASET
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
        print(f"  [INFO] Limited to {MAX_QUESTIONS} questions")

    print(f"  [OK]   Loaded {len(data)} questions from medhop.json")
    return data


# ─────────────────────────────────────────────
# RESUME SUPPORT
# ─────────────────────────────────────────────

def load_existing_predictions() -> dict:
    """Load predictions from a previous interrupted run."""
    if not os.path.exists(PREDICTIONS_FILE_B3):
        return {}

    try:
        with open(PREDICTIONS_FILE_B3, encoding="utf-8") as f:
            existing = json.load(f)
        done = {r["question_id"]: r for r in existing}
        print(f"  [INFO] Found {len(done)} predictions from previous run — will resume")
        return done
    except Exception:
        return {}


# ─────────────────────────────────────────────
# SAVE FUNCTIONS
# ─────────────────────────────────────────────

def save_predictions(predictions: list):
    """Save predictions list to JSON."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PREDICTIONS_FILE_B3, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def save_logs(logs: dict):
    """Save run logs to JSON."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LOGS_FILE_B3, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# CALCULATE EM SCORE
# ─────────────────────────────────────────────

def calculate_em(predictions: list) -> dict:
    """Calculate Exact Match (EM) score."""
    total = len(predictions)
    correct = 0
    failed = 0
    skipped = 0

    for pred in predictions:
        if not pred.get("success", False):
            failed += 1
            continue

        prediction = pred.get("prediction", "").strip().upper()
        answer = pred.get("answer", "").strip().upper()

        if not prediction:
            skipped += 1
            continue

        if prediction == answer:
            correct += 1

    answered = total - failed - skipped
    em_score = (correct / answered * 100) if answered > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": answered - correct,
        "failed": failed,
        "skipped": skipped,
        "answered": answered,
        "em_score": round(em_score, 2),
    }


# ─────────────────────────────────────────────
# PRINT PROGRESS
# ─────────────────────────────────────────────

def print_progress(i: int, total: int, record: dict,
                   prediction: str, is_correct: bool,
                   inference_time: float, retrieval_time: float,
                   correct_count: int, recall_at_k: int):
    """Print progress line."""
    current_em = correct_count / (i + 1) * 100
    status = "✅" if is_correct else "❌"

    # تعديل: إضافة Recall@K إلى السطر المطبوع
    print(
        f"  [{i+1:>3}/{total}] {status} "
        f"{record['id']:<12} | "
        f"pred: {prediction:<10} | "
        f"ans: {record['answer']:<10} | "
        f"RetK: {recall_at_k} | "  # تمت إضافة هذا السطر
        f"ret: {retrieval_time:.2f}s | "
        f"inf: {inference_time:.1f}s | "
        f"EM: {current_em:.1f}%"
    )


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    """Full Baseline 3 inference pipeline."""

    print("\n" + "=" * 60)
    print("  Baseline 3 — MedCPT Semantic RAG Pipeline")
    print("=" * 60)
    print(f"  Model    : {OLLAMA_MODEL_NAME}")
    print(f"  Retrieval: MedCPT (Semantic)")
    print(f"  Top-K    : {MEDCPT_TOP_K}")
    print(f"  Dataset  : {MEDHOP_FILE}")
    print(f"  Output   : {PREDICTIONS_FILE_B3}")
    print()

    pipeline_start = time.time()

    # ── Step 1: Connect to Ollama ──
    print("--- Step 1: Connecting to Ollama ---")
    client = get_ollama_client()
    if not check_model_available(client):
        sys.exit(1)
    print(f"  [OK]   Connected — model '{OLLAMA_MODEL_NAME}' ready\n")

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
    print("--- Step 4: Running MedCPT Retrieval + Inference ---")
    print(f"  Total questions : {total}")
    print(f"  Already done    : {len(existing)}")
    print(f"  Remaining       : {total - len(existing)}")
    print()

    predictions = list(existing.values())
    correct_count = sum(
        1 for p in predictions
        if p.get("prediction", "").upper() == p.get("answer", "").upper()
        and p.get("success", False)
    )
    failed_count = 0
    inference_times = []
    retrieval_times = []
    
    # جديد: لحساب Recall@K
    total_recall_hits = 0

    for i, record in enumerate(data):
        question_id = record["id"]

        # Skip if already done
        if question_id in existing:
            continue

        # ── MedCPT Retrieval ──
        retrieval_start = time.time()
        
        query = record.get("query", "")
        drug_name = record.get("query_drug_name", "")
        supports = record.get("supports", [])
        
        retrieved = retrieve_semantic(
            query=query,
            supports=supports,
            drug_name=drug_name,
            top_k=MEDCPT_TOP_K
        )
        
        retrieval_time = time.time() - retrieval_start
        retrieval_times.append(retrieval_time)

        # جديد: حساب Recall@K (هل الإجابة موجودة في المستندات المسترجعة؟)
        answer_names = []
        ans_name = record.get("answer_name")
        if ans_name:
            answer_names.append(ans_name)
        
        # نتحقق إذا كان اسم الإجابة موجود في المستندات المسترجعة
        recall_hit = 1 if check_answer_in_retrieved(retrieved, answer_names) else 0
        total_recall_hits += recall_hit

        # Build prompt with context
        prompt = build_prompt_with_context(record, retrieved)

        # Run inference
        result = run_inference(client, prompt, question_id)

        # Extract Drug ID
        if result["success"]:
            prediction = extract_drug_id(result["raw_response"], record["candidates"])
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
            "question_id": question_id,
            "query": record["query"],
            "query_drug_id": record["query_drug_id"],
            "query_drug_name": record["query_drug_name"],
            "candidates": record["candidates"],
            "prediction": prediction,
            "raw_response": result["raw_response"],
            "answer": record["answer"],
            "answer_name": record.get("answer_name", record["answer"]),
            "is_correct": is_correct,
            "recall_at_k": recall_hit, # جديد: حفظ قيمة الـ recall
            "retrieval_time": round(retrieval_time, 4),
            "inference_time": result["inference_time"],
            "total_time": round(retrieval_time + result["inference_time"], 4),
            "retrieved_count": len(retrieved),
            "success": result["success"],
            "error": result["error"],
        }

        predictions.append(pred_record)
        existing[question_id] = pred_record

        # Save incrementally
        save_predictions(predictions)

        # Print progress
        if (i + 1) % LOG_EVERY_N == 0 or (i + 1) == total:
            print_progress(
                i, total, record,
                prediction, is_correct,
                result["inference_time"], retrieval_time,
                correct_count, recall_hit # جديد: تمرير قيمة الـ recall
            )

    # ── Step 5: Calculate final EM ──
    print()
    print("--- Step 5: Calculating Results ---")
    em_results = calculate_em(predictions)

    total_time = time.time() - pipeline_start
    avg_inf_time = (sum(inference_times) / len(inference_times)) if inference_times else 0
    avg_ret_time = (sum(retrieval_times) / len(retrieval_times)) if retrieval_times else 0
    
    # جديد: حساب متوسط Recall@K
    avg_recall = (total_recall_hits / total) * 100 if total > 0 else 0.0

    # ── Step 6: Save logs ──
    logs = {
        "run_info": {
            "baseline": "Baseline 3 - MedCPT Semantic RAG",
            "model": OLLAMA_MODEL_NAME,
            "medcpt_top_k": MEDCPT_TOP_K,
            "dataset": MEDHOP_FILE,
            "total_time_min": round(total_time / 60, 2),
            "avg_retrieval_sec": round(avg_ret_time, 4),
            "avg_inference_sec": round(avg_inf_time, 3),
            "avg_total_sec": round(avg_ret_time + avg_inf_time, 3),
            "avg_recall_at_k": round(avg_recall, 2), # جديد
        },
        "em_results": em_results,
        "per_question_times": [
            {
                "question_id": p["question_id"],
                "retrieval_time": p.get("retrieval_time", 0),
                "inference_time": p.get("inference_time", 0),
                "total_time": p.get("total_time", 0),
                "is_correct": p["is_correct"],
                "recall_at_k": p.get("recall_at_k", 0), # جديد
                "success": p["success"],
            }
            for p in predictions
        ],
    }
    save_logs(logs)

    # ── Step 7: Print final summary ──
    print()
    print("=" * 60)
    print("  BASELINE 3 — FINAL RESULTS")
    print("=" * 60)
    print(f"  Method          : MedCPT Semantic RAG (Top-{MEDCPT_TOP_K})")
    print()
    print(f"  Total questions : {em_results['total']}")
    print(f"  Answered        : {em_results['answered']}")
    print(f"  Correct (EM=1)  : {em_results['correct']}")
    print(f"  Wrong   (EM=0)  : {em_results['wrong']}")
    print(f"  Failed          : {em_results['failed']}")
    print()
    print(f"  ✅  Exact Match (EM) Score : {em_results['em_score']}%")
    print(f"  📊  Recall@{MEDCPT_TOP_K} Score        : {avg_recall:.2f}%") # جديد
    print()
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Avg retrieval   : {avg_ret_time:.3f} seconds")
    print(f"  Avg inference   : {avg_inf_time:.2f} seconds")
    print(f"  Avg total       : {avg_ret_time + avg_inf_time:.2f} seconds")
    print()
    print(f"  Predictions saved : {PREDICTIONS_FILE_B3}")
    print(f"  Logs saved        : {LOGS_FILE_B3}")
    print("=" * 60)
    print()

    return em_results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()