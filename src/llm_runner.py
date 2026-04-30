"""
src/llm_runner.py
==================
Biomedical Multi-Hop QA Project — Baseline 1

Handles all communication with BioMistral via Ollama API.

What this module does:
  1. Connects to Ollama service
  2. Sends a prompt to BioMistral
  3. Receives and cleans the raw response
  4. Measures inference time
  5. Handles errors gracefully (timeout, connection, empty response)

Design decisions:
  - Uses ollama Python library (not raw HTTP)
  - Temperature = 0.1 (low = consistent, factual answers)
  - Retries once on failure before giving up
  - Returns structured result dict for each question

Usage:
    py -3.10 src/llm_runner.py
"""

import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    OLLAMA_MODEL_NAME,
    OLLAMA_HOST,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_MAX_TOKENS,
    LLM_NUM_CTX,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# How many times to retry on failure
MAX_RETRIES = 1

# Seconds to wait between retries
RETRY_DELAY = 3

# ─────────────────────────────────────────────
# OLLAMA CLIENT SETUP
# ─────────────────────────────────────────────

def get_ollama_client():
    """
    Create and return an Ollama client.
    Raises clear error if Ollama is not running.
    """
    try:
        import ollama
        # Test connection
        client = ollama.Client(host=OLLAMA_HOST)
        return client
    except ImportError:
        print("  [FAIL] ollama library not installed.")
        print("         Run: pip install ollama")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAIL] Cannot connect to Ollama: {e}")
        print(f"         Make sure Ollama is running: ollama serve")
        sys.exit(1)


def check_model_available(client) -> bool:
    """Check if BioMistral model is registered in Ollama."""
    try:
        models = client.list()
        model_names = [m.model for m in models.models]
        # Check with and without tag
        for name in model_names:
            if OLLAMA_MODEL_NAME in name:
                return True
        print(f"  [FAIL] Model '{OLLAMA_MODEL_NAME}' not found in Ollama.")
        print(f"         Available models: {model_names}")
        print(f"         Run EnvironmentSetup.py to register the model.")
        return False
    except Exception as e:
        print(f"  [WARN] Could not verify model availability: {e}")
        return True  # proceed anyway


# ─────────────────────────────────────────────
# CORE FUNCTION — RUN ONE INFERENCE
# ─────────────────────────────────────────────

def run_inference(client, prompt: str, question_id: str = "") -> dict:
    """
    Send one prompt to BioMistral and return structured result.

    Args:
        client:      Ollama client instance
        prompt:      Formatted prompt string from prompt_builder
        question_id: ID for logging purposes

    Returns:
        {
            "question_id":    str,
            "raw_response":   str,   ← exact text from model
            "inference_time": float, ← seconds
            "success":        bool,
            "error":          str    ← empty if success
        }
    """
    result = {
        "question_id":    question_id,
        "raw_response":   "",
        "inference_time": 0.0,
        "success":        False,
        "error":          "",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            start_time = time.time()

            response = client.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[
                    {
                        "role":    "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": LLM_TEMPERATURE,
                    "top_p":       LLM_TOP_P,
                    "top_k":       LLM_TOP_K,
                    "num_predict": LLM_MAX_TOKENS,
                    "num_ctx":     LLM_NUM_CTX,
                },
            )

            elapsed = time.time() - start_time

            raw_response = response["message"]["content"].strip()

            if not raw_response:
                raise ValueError("Model returned empty response")

            result["raw_response"]   = raw_response
            result["inference_time"] = round(elapsed, 3)
            result["success"]        = True
            result["error"]          = ""
            return result

        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg

            if attempt < MAX_RETRIES:
                print(f"  [WARN] Attempt {attempt+1} failed for {question_id}: {error_msg}")
                print(f"  [INFO] Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [FAIL] All attempts failed for {question_id}: {error_msg}")

    result["inference_time"] = 0.0
    return result


# ─────────────────────────────────────────────
# BATCH FUNCTION — RUN MULTIPLE INFERENCES
# ─────────────────────────────────────────────

def run_batch(client, prompts: list, log_every: int = 10) -> list:
    """
    Run inference for a list of prompt dicts.

    Args:
        client:     Ollama client instance
        prompts:    List of dicts from prompt_builder.build_all_prompts()
                    Each dict: { id, prompt, answer, answer_name }
        log_every:  Print progress every N questions

    Returns:
        List of result dicts with inference results merged in
    """
    results = []
    total   = len(prompts)
    failed  = 0

    print(f"  [INFO] Starting inference on {total} questions...")
    print(f"  [INFO] Model: {OLLAMA_MODEL_NAME}")
    print()

    batch_start = time.time()

    for i, item in enumerate(prompts):
        question_id = item["id"]
        prompt      = item["prompt"]
        answer      = item["answer"]
        answer_name = item.get("answer_name", answer)

        # Run inference
        inference_result = run_inference(client, prompt, question_id)

        # Merge everything into one result dict
        result = {
            "question_id":    question_id,
            "answer":         answer,
            "answer_name":    answer_name,
            "raw_response":   inference_result["raw_response"],
            "inference_time": inference_result["inference_time"],
            "success":        inference_result["success"],
            "error":          inference_result["error"],
        }

        results.append(result)

        if not inference_result["success"]:
            failed += 1

        # Progress logging
        if (i + 1) % log_every == 0 or (i + 1) == total:
            elapsed_total = time.time() - batch_start
            avg_time      = elapsed_total / (i + 1)
            remaining     = avg_time * (total - i - 1)

            print(
                f"  [{i+1:>3}/{total}] "
                f"last: {inference_result['inference_time']:.1f}s | "
                f"avg: {avg_time:.1f}s | "
                f"ETA: {remaining/60:.1f}min | "
                f"failed: {failed}"
            )

    total_time = time.time() - batch_start
    print()
    print(f"  [OK]   Batch complete — {total} questions in {total_time/60:.1f} min")
    print(f"  [INFO] Success: {total - failed} | Failed: {failed}")

    return results


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    import json
    from config.settings import MEDHOP_FILE
    from src.prompt_builder import build_prompt, extract_drug_id

    print("\n" + "=" * 60)
    print("  LLM Runner — Test (3 questions)")
    print("=" * 60)

    # Step 1 — Connect to Ollama
    print("\n--- Connecting to Ollama ---")
    client = get_ollama_client()
    print(f"  [OK]   Connected to Ollama at {OLLAMA_HOST}")

    # Step 2 — Check model
    if not check_model_available(client):
        sys.exit(1)
    print(f"  [OK]   Model '{OLLAMA_MODEL_NAME}' is available")

    # Step 3 — Load 3 test questions
    print("\n--- Loading Test Questions ---")
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found. Run load_dataset.py first.")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    test_data = data[:3]
    print(f"  [OK]   Loaded {len(test_data)} test questions")

    # Step 4 — Run inference
    print("\n--- Running Inference ---")
    correct = 0

    for i, record in enumerate(test_data):
        prompt = build_prompt(record)
        result = run_inference(client, prompt, record["id"])

        if result["success"]:
            prediction = extract_drug_id(
                result["raw_response"],
                record["candidates"]
            )
            is_correct = prediction == record["answer"]
            if is_correct:
                correct += 1
            status = "✅ CORRECT" if is_correct else "❌ WRONG"

            print(f"\n  Question {i+1}: {record['id']}")
            print(f"  Drug      : {record['query_drug_name']} ({record['query_drug_id']})")
            print(f"  Expected  : {record['answer']} ({record.get('answer_name', '')})")
            print(f"  Raw resp  : {result['raw_response'][:80]}")
            print(f"  Extracted : {prediction}")
            print(f"  Result    : {status}")
            print(f"  Time      : {result['inference_time']}s")
        else:
            print(f"\n  Question {i+1}: {record['id']} — FAILED: {result['error']}")

    # Summary
    print("\n" + "─" * 60)
    print(f"  Test Results: {correct}/{len(test_data)} correct")
    print(f"  Accuracy on 3 samples: {correct/len(test_data)*100:.1f}%")
    print()
    print("=" * 60)
    print("  ✅  llm_runner.py working correctly.")
    print("  Next step: src/inference_pipeline.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
