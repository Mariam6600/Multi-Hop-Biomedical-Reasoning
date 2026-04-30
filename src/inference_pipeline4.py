"""
src/inference_pipeline4.py
===========================
Biomedical Multi-Hop QA Project — Stage 4: Entity Bridging

Pipeline:
  Question
    → Step 1: MedCPT retrieval (K=5, same as Baseline 3)
    → Step 2: Bridge extraction via LLM ("what is the mechanism of Drug X?")
    → Step 3: MedCPT retrieval using bridge query
    → Step 4: Merge + deduplicate original + bridge docs
    → Step 5: FewShot + context prompt → BioMistral → Answer

Why this order?
  - Step 1 catches documents that directly mention the drug (drug-centric docs)
  - Step 2 identifies the biological "hop" (enzyme/pathway/receptor)
  - Step 3 catches documents about that mechanism and drugs sharing it
  - Merging both sets gives the model complete context for multi-hop reasoning

Key design decisions:
  - All original Baseline 3 results preserved (different output files)
  - Bridge extraction capped at 80 tokens → fast
  - Deduplication by text (not index) → safe across retrieval passes
  - Full resume support (pick up where we left off)
  - bridge_entity logged per question for analysis

Usage:
    py -3.10 src/inference_pipeline4.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    PREDICTIONS_FILE_STAGE4,
    LOGS_FILE_STAGE4,
    OUTPUTS_DIR,
    MAX_QUESTIONS,
    LOG_EVERY_N,
    OLLAMA_MODEL_NAME,
    MEDCPT_TOP_K,
    STAGE4_BRIDGE_TOP_K,
    STAGE4_FINAL_TOP_K,
)
from src.prompt_builder import build_prompt_fewshot_with_context, extract_drug_id
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.retriever_semantic import retrieve_semantic, check_answer_in_retrieved
from src.bridge_extractor import extract_bridge_entity, build_bridge_query


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
    if not os.path.exists(PREDICTIONS_FILE_STAGE4):
        return {}

    try:
        with open(PREDICTIONS_FILE_STAGE4, encoding="utf-8") as f:
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
    """Save predictions list to JSON (incremental)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PREDICTIONS_FILE_STAGE4, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def save_logs(logs: dict):
    """Save run logs to JSON."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LOGS_FILE_STAGE4, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# DOCUMENT MERGING & DEDUPLICATION
# ─────────────────────────────────────────────

def merge_and_deduplicate(
    original_docs: list,
    bridge_docs: list,
    final_top_k: int,
) -> list:
    """
    Merge two sets of retrieved documents, deduplicate by text,
    and return the top-N by score.

    Strategy:
      1. Combine both lists
      2. Remove exact duplicates (same text content)
      3. Sort by score descending
      4. Return top final_top_k

    Args:
        original_docs: Results from the first retrieval (drug query)
        bridge_docs:   Results from the bridge retrieval (mechanism query)
        final_top_k:   Maximum number of docs to return

    Returns:
        Merged, deduplicated, sorted list of doc dicts
    """
    seen_texts = set()
    merged = []

    # Process original docs first (they get priority in case of ties)
    for doc in original_docs:
        text_key = doc["text"].strip()[:100]  # use first 100 chars as key
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append({**doc, "source": "original"})

    # Process bridge docs — deduplicate against originals
    for doc in bridge_docs:
        text_key = doc["text"].strip()[:100]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            merged.append({**doc, "source": "bridge"})

    # Sort by score descending
    merged.sort(key=lambda x: x["score"], reverse=True)

    # Return top-N
    return merged[:final_top_k]


# ─────────────────────────────────────────────
# CALCULATE EM SCORE
# ─────────────────────────────────────────────

def calculate_em(predictions: list) -> dict:
    """Calculate Exact Match (EM) and bridge statistics."""
    total   = len(predictions)
    correct = 0
    failed  = 0
    skipped = 0
    bridge_success = 0

    for pred in predictions:
        if pred.get("bridge_success"):
            bridge_success += 1

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
        "total":          total,
        "correct":        correct,
        "wrong":          answered - correct,
        "failed":         failed,
        "skipped":        skipped,
        "answered":       answered,
        "em_score":       round(em_score, 2),
        "bridge_success": bridge_success,
        "bridge_rate":    round(bridge_success / total * 100, 1) if total > 0 else 0.0,
    }


# ─────────────────────────────────────────────
# PRINT PROGRESS
# ─────────────────────────────────────────────

def print_progress(
    i: int,
    total: int,
    record: dict,
    prediction: str,
    is_correct: bool,
    bridge_time: float,
    retrieval_time: float,
    inference_time: float,
    correct_count: int,
    recall_at_k: int,
    bridge_clean: str,
):
    """Print one progress line with bridge info."""
    current_em = correct_count / (i + 1) * 100
    status     = "✅" if is_correct else "❌"
    bridge_short = bridge_clean[:30] + "..." if len(bridge_clean) > 30 else bridge_clean

    print(
        f"  [{i+1:>3}/{total}] {status} "
        f"{record['id']:<12} | "
        f"pred: {prediction:<10} | "
        f"ans: {record['answer']:<10} | "
        f"Ret@K: {recall_at_k} | "
        f"bridge: {bridge_time:.1f}s | "
        f"ret: {retrieval_time:.1f}s | "
        f"inf: {inference_time:.1f}s | "
        f"EM: {current_em:.1f}%"
    )
    if bridge_clean:
        print(f"           Bridge: \"{bridge_short}\"")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    """Full Stage 4 — Entity Bridging inference pipeline."""

    print("\n" + "=" * 60)
    print("  Stage 4 — Entity Bridging + MedCPT Pipeline")
    print("=" * 60)
    print(f"  Model        : {OLLAMA_MODEL_NAME}")
    print(f"  Retrieval    : MedCPT (Semantic)")
    print(f"  Bridge Top-K : {STAGE4_BRIDGE_TOP_K}")
    print(f"  Final Top-K  : {STAGE4_FINAL_TOP_K}")
    print(f"  Prompt       : FewShot + Context")
    print(f"  Dataset      : {MEDHOP_FILE}")
    print(f"  Output       : {PREDICTIONS_FILE_STAGE4}")
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
    data  = load_data()
    total = len(data)
    print()

    # ── Step 3: Check for previous run ──
    print("--- Step 3: Checking Previous Progress ---")
    existing = load_existing_predictions()
    print()

    # ── Step 4: Run pipeline ──
    print("--- Step 4: Entity Bridging + Retrieval + Inference ---")
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

    bridge_times    = []
    retrieval_times = []
    inference_times = []
    total_recall    = 0
    failed_count    = 0

    for i, record in enumerate(data):
        question_id = record["id"]

        if question_id in existing:
            continue

        query      = record.get("query", "")
        drug_id    = record["query_drug_id"]
        drug_name  = record.get("query_drug_name", drug_id)
        supports   = record.get("supports", [])
        ans_name   = record.get("answer_name", "")

        # ── Step A: Original retrieval (drug query) ──
        ret_start = time.time()
        original_docs = retrieve_semantic(
            query=query,
            supports=supports,
            drug_name=drug_name,
            top_k=MEDCPT_TOP_K,
        )
        retrieval_time = time.time() - ret_start

        # ── Step B: Bridge extraction ──
        bridge_start = time.time()
        bridge_result = extract_bridge_entity(
            client=client,
            drug_name=drug_name,
            drug_id=drug_id,
            question_id=question_id,
        )
        bridge_time = time.time() - bridge_start
        bridge_times.append(bridge_time)

        # ── Step C: Bridge retrieval ──
        bridge_docs = []
        if bridge_result["success"] and bridge_result["bridge_query"] != drug_name:
            ret2_start = time.time()
            bridge_docs = retrieve_semantic(
                query=bridge_result["bridge_query"],
                supports=supports,
                drug_name=drug_name,
                top_k=STAGE4_BRIDGE_TOP_K,
            )
            retrieval_time += time.time() - ret2_start

        retrieval_times.append(retrieval_time)

        # ── Step D: Merge + deduplicate ──
        final_docs = merge_and_deduplicate(
            original_docs=original_docs,
            bridge_docs=bridge_docs,
            final_top_k=STAGE4_FINAL_TOP_K,
        )

        # ── Check Recall@K ──
        recall_hit = 1 if (ans_name and check_answer_in_retrieved(
            final_docs, [ans_name]
        )) else 0
        total_recall += recall_hit

        # ── Step E: Build FewShot + Context prompt ──
        prompt = build_prompt_fewshot_with_context(record, final_docs)

        # ── Step F: LLM inference ──
        inf_result = run_inference(client, prompt, question_id)

        if inf_result["success"]:
            prediction = extract_drug_id(inf_result["raw_response"], record["candidates"])
        else:
            prediction = ""
            failed_count += 1

        inference_times.append(inf_result["inference_time"])

        is_correct = (
            prediction.strip().upper() == record["answer"].strip().upper()
            and inf_result["success"]
        )
        if is_correct:
            correct_count += 1

        # ── Build prediction record ──
        pred_record = {
            "question_id":      question_id,
            "query":            query,
            "query_drug_id":    drug_id,
            "query_drug_name":  drug_name,
            "candidates":       record["candidates"],
            "prediction":       prediction,
            "raw_response":     inf_result["raw_response"],
            "answer":           record["answer"],
            "answer_name":      ans_name,
            "is_correct":       is_correct,
            "recall_at_k":      recall_hit,
            # Bridge-specific fields
            "bridge_raw":       bridge_result["bridge_raw"],
            "bridge_clean":     bridge_result["bridge_clean"],
            "bridge_query":     bridge_result["bridge_query"],
            "bridge_success":   bridge_result["success"],
            "bridge_docs_count": len(bridge_docs),
            "final_docs_count": len(final_docs),
            # Timing
            "bridge_time":      round(bridge_time, 3),
            "retrieval_time":   round(retrieval_time, 3),
            "inference_time":   inf_result["inference_time"],
            "total_time":       round(
                bridge_time + retrieval_time + inf_result["inference_time"], 3
            ),
            # Status
            "success":  inf_result["success"],
            "error":    inf_result["error"],
            "prompt_type": "entity_bridging_fewshot",
        }

        predictions.append(pred_record)
        existing[question_id] = pred_record

        save_predictions(predictions)

        # Print progress every LOG_EVERY_N questions
        if (i + 1) % LOG_EVERY_N == 0 or (i + 1) == total:
            print_progress(
                i=i,
                total=total,
                record=record,
                prediction=prediction,
                is_correct=is_correct,
                bridge_time=bridge_time,
                retrieval_time=retrieval_time,
                inference_time=inf_result["inference_time"],
                correct_count=correct_count,
                recall_at_k=recall_hit,
                bridge_clean=bridge_result["bridge_clean"],
            )

    # ── Step 5: Calculate results ──
    print()
    print("--- Step 5: Calculating Results ---")
    em_results = calculate_em(predictions)

    total_time    = time.time() - pipeline_start
    avg_bridge    = sum(bridge_times) / len(bridge_times) if bridge_times else 0
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_recall    = (total_recall / total) * 100 if total > 0 else 0.0

    # ── Save logs ──
    logs = {
        "run_info": {
            "stage":            "Stage 4 - Entity Bridging + MedCPT",
            "model":            OLLAMA_MODEL_NAME,
            "medcpt_top_k":     MEDCPT_TOP_K,
            "bridge_top_k":     STAGE4_BRIDGE_TOP_K,
            "final_top_k":      STAGE4_FINAL_TOP_K,
            "prompt_type":      "fewshot_with_context",
            "dataset":          MEDHOP_FILE,
            "total_time_min":   round(total_time / 60, 2),
            "avg_bridge_sec":   round(avg_bridge, 3),
            "avg_retrieval_sec": round(avg_retrieval, 3),
            "avg_inference_sec": round(avg_inference, 3),
            "avg_total_sec":    round(avg_bridge + avg_retrieval + avg_inference, 3),
            "avg_recall_at_k":  round(avg_recall, 2),
        },
        "em_results": em_results,
        "per_question_times": [
            {
                "question_id":  p["question_id"],
                "bridge_time":  p.get("bridge_time", 0),
                "retrieval_time": p.get("retrieval_time", 0),
                "inference_time": p.get("inference_time", 0),
                "total_time":   p.get("total_time", 0),
                "is_correct":   p["is_correct"],
                "recall_at_k":  p.get("recall_at_k", 0),
                "bridge_success": p.get("bridge_success", False),
                "bridge_clean": p.get("bridge_clean", ""),
                "success":      p["success"],
            }
            for p in predictions
        ],
    }
    save_logs(logs)

    # ── Print final summary ──
    print()
    print("=" * 60)
    print("  STAGE 4 — ENTITY BRIDGING — FINAL RESULTS")
    print("=" * 60)
    print(f"  Method          : Entity Bridging + MedCPT")
    print(f"  Bridge Top-K    : {STAGE4_BRIDGE_TOP_K} | Final Top-K: {STAGE4_FINAL_TOP_K}")
    print()
    print(f"  Total questions : {em_results['total']}")
    print(f"  Answered        : {em_results['answered']}")
    print(f"  Correct (EM=1)  : {em_results['correct']}")
    print(f"  Wrong   (EM=0)  : {em_results['wrong']}")
    print(f"  Failed          : {em_results['failed']}")
    print()
    print(f"  ✅  Exact Match (EM)   : {em_results['em_score']}%")
    print(f"  📊  Recall@K           : {avg_recall:.2f}%")
    print(f"  🔗  Bridge success     : {em_results['bridge_success']}/{em_results['total']} ({em_results['bridge_rate']}%)")
    print()
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Avg bridge      : {avg_bridge:.2f}s")
    print(f"  Avg retrieval   : {avg_retrieval:.2f}s")
    print(f"  Avg inference   : {avg_inference:.2f}s")
    print(f"  Avg total/Q     : {avg_bridge+avg_retrieval+avg_inference:.2f}s")
    print()
    print(f"  Predictions : {PREDICTIONS_FILE_STAGE4}")
    print(f"  Logs        : {LOGS_FILE_STAGE4}")
    print("=" * 60)
    print()

    # ── Compare with Baseline 3 if available ──
    _print_comparison(em_results["em_score"], avg_recall)

    return em_results


# ─────────────────────────────────────────────
# COMPARISON HELPER
# ─────────────────────────────────────────────

def _print_comparison(stage4_em: float, stage4_recall: float):
    """Print side-by-side comparison with Baseline 3 K=5."""
    import json as _json

    b3_path = os.path.join(OUTPUTS_DIR, "baseline3_k5_logs.json")
    if not os.path.exists(b3_path):
        return

    try:
        with open(b3_path) as f:
            b3 = _json.load(f)
        b3_em     = b3["em_results"]["em_score"]
        b3_recall = b3["run_info"].get("avg_recall_at_k", 0)

        delta_em     = stage4_em - b3_em
        delta_recall = stage4_recall - b3_recall
        sign_em      = "+" if delta_em >= 0 else ""
        sign_r       = "+" if delta_recall >= 0 else ""

        print("  ── Comparison with Baseline 3 (MedCPT K=5) ──")
        print(f"  EM:     B3={b3_em}%  →  Stage4={stage4_em}%  ({sign_em}{delta_em:.2f}%)")
        print(f"  Recall: B3={b3_recall}%  →  Stage4={stage4_recall:.2f}%  ({sign_r}{delta_recall:.2f}%)")
        print()
    except Exception:
        pass


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()