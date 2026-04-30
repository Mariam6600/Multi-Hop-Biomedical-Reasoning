"""
src/inference_pipeline_hybrid.py
=================================
Biomedical Multi-Hop QA Project — Hybrid Retrieval Pipeline

Combines BM25 (Lexical) and MedCPT (Semantic) retrieval methods
using Reciprocal Rank Fusion (RRF).

Usage:
    py -3.10 src/inference_pipeline3-1_hybrid.py
"""

import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    PREDICTIONS_FILE_HYBRID,
    LOGS_FILE_HYBRID,
    OUTPUTS_DIR,
    MAX_QUESTIONS,
    LOG_EVERY_N,
    OLLAMA_MODEL_NAME,
    HYBRID_TOP_K,
    HYBRID_CANDIDATES,
    RRF_K_CONSTANT,
    MEDCPT_WEIGHT,
    BM25_WEIGHT,
)
from src.prompt_builder import build_prompt_with_context, extract_drug_id
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.retriever import retrieve_top_k  # BM25
from src.retriever_semantic import retrieve_semantic, check_answer_in_retrieved # MedCPT


# ─────────────────────────────────────────────
# HYBRID RETRIEVAL LOGIC (RRF)
# ─────────────────────────────────────────────

def weighted_combination(medcpt_results, bm25_results, medcpt_w=0.70, bm25_w=0.30):
    """
    Combine MedCPT and BM25 results using weighted scores.
    
    MedCPT scores are cosine similarities (0 to 1).
    BM25 scores are normalized to (0 to 1) before combining.
    
    Final score = medcpt_w * medcpt_score + bm25_w * bm25_score_normalized
    """
    # Normalize BM25 scores to 0-1 range
    bm25_max = max((r["score"] for r in bm25_results), default=1.0)
    bm25_min = min((r["score"] for r in bm25_results), default=0.0)
    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

    # Build score maps
    medcpt_map = {r["text"]: r["score"] for r in medcpt_results}
    bm25_map   = {r["text"]: (r["score"] - bm25_min) / bm25_range for r in bm25_results}

    # Collect all unique texts
    all_texts = set(medcpt_map.keys()) | set(bm25_map.keys())

    # Calculate weighted score for each text
    combined = {}
    for text in all_texts:
        m_score = medcpt_map.get(text, 0.0)
        b_score = bm25_map.get(text, 0.0)
        combined[text] = medcpt_w * m_score + bm25_w * b_score

    # Sort by combined score
    sorted_texts = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{"text": t, "score": s, "rank": i+1} for i, (t, s) in enumerate(sorted_texts)]


def reciprocal_rank_fusion(lists_of_results, k=60):
    """Kept for reference — not used in EXP10 (Weighted)."""
    rrf_scores = {}
    for results in lists_of_results:
        for rank, item in enumerate(results, 1):
            text = item["text"]
            if text not in rrf_scores:
                rrf_scores[text] = 0
            rrf_scores[text] += 1 / (k + rank)
    sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": text, "score": score, "rank": i+1} for i, (text, score) in enumerate(sorted_texts)]


def retrieve_hybrid(query, supports, drug_name, top_k):
    """
    Retrieve using Hybrid strategy (BM25 + MedCPT).
    """
    # 1. Retrieve candidates from BM25
    bm25_results = retrieve_top_k(
        query=query,
        supports=supports,
        drug_name=drug_name,
        top_k=HYBRID_CANDIDATES
    )
    
    # 2. Retrieve candidates from MedCPT
    medcpt_results = retrieve_semantic(
        query=query,
        supports=supports,
        drug_name=drug_name,
        top_k=HYBRID_CANDIDATES
    )
    
    # 3. Weighted combination (MedCPT 70% + BM25 30%)
    fused_results = weighted_combination(
        medcpt_results=medcpt_results,
        bm25_results=bm25_results,
        medcpt_w=MEDCPT_WEIGHT,
        bm25_w=BM25_WEIGHT,
    )
    
    # 4. Return top-K final results
    return fused_results[:top_k]


# ─────────────────────────────────────────────
# MAIN PIPELINE (Similar structure to previous)
# ─────────────────────────────────────────────

# [Include load_data, load_existing_predictions, save_predictions, calculate_em functions here]
# (They are identical to previous pipelines, just copied here for completeness)

def load_data() -> list:
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found")
        sys.exit(1)
    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if MAX_QUESTIONS is not None:
        data = data[:MAX_QUESTIONS]
    print(f"  [OK]   Loaded {len(data)} questions")
    return data

def load_existing_predictions() -> dict:
    if not os.path.exists(PREDICTIONS_FILE_HYBRID):
        return {}
    try:
        with open(PREDICTIONS_FILE_HYBRID, encoding="utf-8") as f:
            existing = json.load(f)
        done = {r["question_id"]: r for r in existing}
        print(f"  [INFO] Found {len(done)} predictions — resuming")
        return done
    except: return {}

def save_predictions(predictions: list):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(PREDICTIONS_FILE_HYBRID, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

def save_logs(logs: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(LOGS_FILE_HYBRID, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

def calculate_em(predictions: list) -> dict:
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("prediction","").strip().upper() == p.get("answer","").strip().upper() and p.get("success"))
    answered = sum(1 for p in predictions if p.get("success"))
    em_score = (correct / answered * 100) if answered > 0 else 0.0
    return {"total": total, "correct": correct, "em_score": round(em_score, 2)}


def print_progress(i, total, record, prediction, is_correct, inf_time, ret_time, correct_count, recall_at_k):
    status = "✅" if is_correct else "❌"
    print(
        f"  [{i+1:>3}/{total}] {status} {record['id']:<12} | "
        f"pred: {prediction:<10} | ans: {record['answer']:<10} | "
        f"RetK: {recall_at_k} | ret: {ret_time:.2f}s | inf: {inf_time:.1f}s | "
        f"EM: {correct_count/(i+1)*100:.1f}%"
    )


def run_pipeline():
    print("\n" + "=" * 60)
    print("  HYBRID RETRIEVAL — BM25 + MedCPT (RRF)")
    print("=" * 60)
    print(f"  Model    : {OLLAMA_MODEL_NAME}")
    print(f"  Strategy : Hybrid Weighted (MedCPT={MEDCPT_WEIGHT} + BM25={BM25_WEIGHT})")
    print(f"  Top-K    : {HYBRID_TOP_K}")
    print(f"  Output   : {PREDICTIONS_FILE_HYBRID}")
    print()

    pipeline_start = time.time()

    print("--- Step 1: Connecting to Ollama ---")
    client = get_ollama_client()
    if not check_model_available(client): sys.exit(1)
    
    print("--- Step 2: Loading Dataset ---")
    data = load_data()
    total = len(data)

    print("--- Step 3: Checking Previous Progress ---")
    existing = load_existing_predictions()

    print("--- Step 4: Running Hybrid Retrieval + Inference ---")
    predictions = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("prediction","").upper() == p.get("answer","").upper() and p.get("success"))
    
    retrieval_times = []
    inference_times = []
    total_recall_hits = 0

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing: continue

        # --- Hybrid Retrieval ---
        ret_start = time.time()
        retrieved = retrieve_hybrid(
            query=record.get("query",""),
            supports=record.get("supports",[]),
            drug_name=record.get("query_drug_name",""),
            top_k=HYBRID_TOP_K
        )
        ret_time = time.time() - ret_start
        retrieval_times.append(ret_time)

        # Check Recall
        ans_names = [record.get("answer_name")] if record.get("answer_name") else []
        recall_hit = 1 if check_answer_in_retrieved(retrieved, ans_names) else 0
        total_recall_hits += recall_hit

        # Inference
        prompt = build_prompt_with_context(record, retrieved)
        result = run_inference(client, prompt, qid)
        
        prediction = extract_drug_id(result["raw_response"], record["candidates"]) if result["success"] else ""
        is_correct = (prediction.upper() == record["answer"].upper() and result["success"])
        if is_correct: correct_count += 1
        if result["inference_time"] > 0: inference_times.append(result["inference_time"])

        # Save
        pred_record = {
            "question_id": qid, "query": record["query"], "prediction": prediction,
            "answer": record["answer"], "is_correct": is_correct, "recall_at_k": recall_hit,
            "retrieval_time": ret_time, "inference_time": result["inference_time"],
            "success": result["success"]
        }
        predictions.append(pred_record)
        existing[qid] = pred_record
        save_predictions(predictions)

        if (i+1) % LOG_EVERY_N == 0:
            print_progress(i, total, record, prediction, is_correct, result["inference_time"], ret_time, correct_count, recall_hit)

    # --- Final Results ---
    print("\n--- Final Results ---")
    em_res = calculate_em(predictions)
    avg_ret = sum(retrieval_times)/len(retrieval_times) if retrieval_times else 0
    avg_inf = sum(inference_times)/len(inference_times) if inference_times else 0
    avg_recall = (total_recall_hits / total) * 100 if total > 0 else 0

    logs = {
        "run_info": {
            "method": "Hybrid (BM25+MedCPT)", "top_k": HYBRID_TOP_K,
            "em_score": em_res["em_score"], "recall": avg_recall
        }
    }
    save_logs(logs)

    print(f"  ✅ EM: {em_res['em_score']}%")
    print(f"  📊 Recall@{HYBRID_TOP_K}: {avg_recall:.2f}%")
    print(f"  ⏱️  Avg Time: {avg_ret + avg_inf:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()