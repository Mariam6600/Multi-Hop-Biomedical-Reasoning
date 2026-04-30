"""
src/inference_pipeline5.py
===========================
Biomedical Multi-Hop QA — Semantic Expansion Pipeline
يدعم أربعة أنواع من الـ retriever:

  "expanded"        ← Term-Count العادي (الإصدار المُصلح)
  "semantic"        ← MedCPT cosine similarity
  "weighted_struct" ← Structured Expansion مع أوزان حسب الفئة (تجربة جديدة)
  hybrid (مستقبلاً)

للتبديل: غيّر EXP_RETRIEVER في الأسفل
"""

import json
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    OUTPUTS_DIR,
    DIAG_SAMPLE_SIZE,
    ACTIVE_PROVIDER,
    OLLAMA_MODEL_NAME,
)

# ─────────────────────────────────────────────
# اختيار runner حسب ACTIVE_PROVIDER
# ─────────────────────────────────────────────

if ACTIVE_PROVIDER == "ollama":
    from src.llm_runner import (
        get_ollama_client     as _get_client,
        check_model_available as _check_model,
        run_inference         as _run_inference,
    )
    def get_client():      return _get_client()
    def check_model(c):    return _check_model(c)
    def do_inference(c, p, q): return _run_inference(c, p, q)
    def active_model_name():   return OLLAMA_MODEL_NAME
else:
    from src.llm_runner_api import (
        get_api_client        as _get_client,
        check_model_available as _check_model,
        run_inference         as _run_inference,
    )
    from config.settings import API_MODEL
    def get_client():      return _get_client()
    def check_model(c):    return _check_model(c)
    def do_inference(c, p, q): return _run_inference(c, p, q)
    def active_model_name():   return API_MODEL

# ─────────────────────────────────────────────
# استيراد المكونات
# ─────────────────────────────────────────────

from src.query_expander import expand_query
from src.retriever_expanded import retrieve_expanded
from src.retriever_semantic import retrieve_semantic
from src.prompt_builder import extract_drug_id
from src.gold_chain_retriever import retrieve_gold_chain, retrieve_query_drug
from src.query_expander_structured import get_weighted_terms
from src.retriever_hybrid_scored import retrieve_hybrid_scored
from src.retriever_combined import retrieve_combined


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

def _format_docs(retrieved: list) -> str:
    if not retrieved:
        return "No supporting evidence available."
    parts = []
    for r in retrieved:
        text = r["text"]
        if len(text) > 400:
            text = text[:400] + "..."
        parts.append(f"[{r['rank']}] {text}")
    return "\n".join(parts)


def build_prompt_direct(record: dict, retrieved: list) -> str:
    docs      = _format_docs(retrieved)
    drug_id   = record["query_drug_id"]
    drug_name = record.get("query_drug_name", drug_id)
    candidates_text = "\n".join(
        f"- {name} ({cid})" if name != cid else f"- {cid}"
        for cid, name in zip(
            record["candidates"],
            record.get("candidate_names", record["candidates"])
        )
    )
    return f"""You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the supporting evidence to identify which drug interacts with {drug_name}
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""


def build_prompt_fewshot(record: dict, retrieved: list) -> str:
    docs      = _format_docs(retrieved)
    drug_id   = record["query_drug_id"]
    drug_name = record.get("query_drug_name", drug_id)
    candidates_text = "\n".join(
        f"- {name} ({cid})" if name != cid else f"- {cid}"
        for cid, name in zip(
            record["candidates"],
            record.get("candidate_names", record["candidates"])
        )
    )
    return f"""You are a biomedical expert specializing in drug interactions.

Example:
Drug: Fluoxetine (DB00472) — inhibits the CYP2D6 enzyme
Supporting evidence: "Fluoxetine is a potent inhibitor of CYP2D6, affecting the metabolism of drugs like desipramine."
Interacting drug: Desipramine (DB01151) — because it is metabolized by CYP2D6

Now answer the following:

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the example reasoning pattern above
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""


PROMPT_BUILDERS = {
    "direct":  build_prompt_direct,
    "fewshot": build_prompt_fewshot,
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(n) -> list:
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)
    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data[:n] if n else data


def _model_short_name() -> str:
    name = active_model_name()
    name = re.sub(r"[/:\\.]", "_", name).strip("_")
    return name[-35:] if len(name) > 35 else name


def output_path(retriever: str, k: int, prompt: str) -> tuple:
    base = f"phase1_{retriever}_k{k}_{prompt}_{_model_short_name()}"
    pred = os.path.join(OUTPUTS_DIR, f"{base}_predictions.json")
    logs = os.path.join(OUTPUTS_DIR, f"{base}_logs.json")
    return pred, logs


def load_existing(pred_file: str) -> dict:
    if not os.path.exists(pred_file):
        return {}
    try:
        with open(pred_file, encoding="utf-8") as f:
            existing = json.load(f)
        done = {r["question_id"]: r for r in existing}
        print(f"  [INFO] Resuming — {len(done)} already done")
        return done
    except Exception:
        return {}


def save_preds(preds: list, path: str):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)


def calculate_em(preds: list) -> dict:
    total    = len(preds)
    failed   = sum(1 for p in preds if not p.get("success", False))
    answered = total - failed
    correct  = sum(
        1 for p in preds
        if p.get("success", False)
        and p.get("prediction", "").strip().upper() == p.get("answer", "").strip().upper()
        and p.get("prediction", "").strip() != ""
    )
    nm_hits  = sum(1 for p in preds if p.get("nm_hit", False) and p.get("success", False))
    return {
        "total":    total,
        "answered": answered,
        "correct":  correct,
        "failed":   failed,
        "nm_hits":  nm_hits,
        "em_score": round(correct / answered * 100, 2) if answered > 0 else 0.0,
        "nm_score": round(nm_hits  / answered * 100, 2) if answered > 0 else 0.0,
    }

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_experiment(
    retriever_type: str = "expanded",
    top_k: int = 5,
    prompt_type: str = "direct",
    sample_size: int = None,
    verbose: bool = True,
) -> dict:

    if sample_size is None:
        sample_size = DIAG_SAMPLE_SIZE

    # Import weighted_struct retriever only if needed (lazy import)
    if retriever_type == "weighted_struct":
        from src.query_expander_structured import expand_query_structured
        from src.retriever_weighted_struct import retrieve_weighted_struct

    pred_file, logs_file = output_path(retriever_type, top_k, prompt_type)
    build_prompt = PROMPT_BUILDERS.get(prompt_type, build_prompt_direct)
    model_name   = active_model_name()

    print("\n" + "=" * 60)
    print("  Phase 1 — Semantic Expansion Pipeline")
    print("=" * 60)
    print(f"  Provider  : {ACTIVE_PROVIDER}")
    print(f"  Model     : {model_name}")
    print(f"  Retriever : {retriever_type}")
    print(f"  Top-K     : {top_k}")
    print(f"  Prompt    : {prompt_type}")
    print(f"  Sample    : {sample_size} questions")
    print(f"  Output    : {pred_file}")
    print()

    pipeline_start = time.time()

    print(f"--- Connecting to {ACTIVE_PROVIDER} ---")
    client = get_client()
    if not check_model(client):
        sys.exit(1)
    print()

    print("--- Loading Dataset ---")
    data  = load_data(sample_size)
    total = len(data)
    print(f"  [OK]   {total} questions loaded")
    print()

    existing      = load_existing(pred_file)
    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct", False))
    nm_count      = sum(1 for p in predictions if p.get("nm_hit", False))

    ret_times, inf_times = [], []

    remaining = total - len(existing)
    print(f"--- Running ({remaining} remaining) ---\n")

    if ACTIVE_PROVIDER != "ollama" and remaining > 0:
        _rate = client.get("rate", 20.0) if isinstance(client, dict) else 20.0
        print(f"  [INFO] Estimated time: ~{remaining * _rate / 60:.1f} min\n")

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            continue

        drug_name = record.get("query_drug_name", "")
        supports  = record.get("supports", [])
        ans_name  = record.get("answer_name", "")

        # ── Expansion + Retrieval ──────────────────
        ret_start = time.time()

        if retriever_type == "expanded":
            exp_result = expand_query(drug_name)
            expanded_terms = exp_result["terms"] if exp_result["success"] else []
            retrieved = retrieve_expanded(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                expanded_terms=expanded_terms,
                top_k=top_k,
            )

        elif retriever_type == "weighted_struct":
            exp_result = expand_query_structured(drug_name)
            weighted_terms = exp_result.get("weighted_terms", [])
            if not weighted_terms:
                # Fallback to flat terms with weight=1.0
                weighted_terms = [{"term": t, "weight": 1.0, "category": "unknown"}
                                   for t in exp_result.get("terms", [])]
            retrieved = retrieve_weighted_struct(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                weighted_terms=weighted_terms,
                top_k=top_k,
            )

        elif retriever_type == "gold_chain":
            retrieved = retrieve_gold_chain(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                answer_id=record["answer"],
                answer_name=record.get("answer_name", ""),
                top_k=top_k,
            )

        elif retriever_type == "query_drug":
            # تصفية بناءً على دواء السؤال — واقعية ولا تحتاج الإجابة
            retrieved = retrieve_query_drug(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                drug_id=record.get("query_drug_id", ""),
                top_k=top_k,
            )

        elif retriever_type == "hybrid_scored":
            # 1. جلب المصطلحات العادية (Flat Terms)
            exp_result = expand_query(drug_name)
            flat_terms_list = exp_result["terms"] if exp_result["success"] else []
            
            # 2. جلب المصطلحات الموزونة (Weighted Terms)
            weighted = get_weighted_terms(drug_name)
            
            retrieved = retrieve_hybrid_scored(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                flat_terms=flat_terms_list,  # <- استخدام المتغير الصحيح
                weighted_terms=weighted,
                top_k=top_k,
            )    

        elif retriever_type == "combined":
            exp_result = expand_query(drug_name)
            expanded_terms = exp_result["terms"] if exp_result["success"] else []
            retrieved = retrieve_combined(
                query=record.get("query",""), supports=supports,
                drug_name=drug_name, drug_id=record.get("query_drug_id",""),
                expanded_terms=expanded_terms, top_k=top_k,
            )

        else:  # semantic
            retrieved = retrieve_semantic(
                query=record.get("query", ""),
                supports=supports,
                drug_name=drug_name,
                top_k=top_k,
            )

        ret_times.append(time.time() - ret_start)

        # ── Prompt + Inference ─────────────────────
        prompt = build_prompt(record, retrieved)
        inf_result = do_inference(client, prompt, qid)
        inf_times.append(inf_result["inference_time"])

        # ── Strip thinking blocks ──────────────────
        if inf_result["success"]:
            raw = inf_result["raw_response"]
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", raw, flags=re.DOTALL).strip()
            if raw:
                inf_result["raw_response"] = raw

        # ── Extract prediction ─────────────────────
        if inf_result["success"]:
            prediction = extract_drug_id(inf_result["raw_response"], record["candidates"])
        else:
            prediction = ""

        is_correct = (
            bool(prediction)
            and prediction.upper() == record["answer"].upper()
            and inf_result["success"]
        )
        if is_correct:
            correct_count += 1

        nm_hit = (
            bool(ans_name)
            and ans_name.lower() in inf_result["raw_response"].lower()
            and inf_result["success"]
        )
        if nm_hit:
            nm_count += 1

        # ── Save record ────────────────────────────
        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "raw_response":    inf_result["raw_response"],
            "answer":          record["answer"],
            "answer_name":     ans_name,
            "is_correct":      is_correct,
            "nm_hit":          nm_hit,
            "model":           model_name,
            "provider":        ACTIVE_PROVIDER,
            "retriever":       retriever_type,
            "top_k":           top_k,
            "prompt_type":     prompt_type,
            "success":         inf_result["success"],
            "error":           inf_result["error"],
            "inference_time":  inf_result["inference_time"],
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done_so_far = i + 1
        if verbose and (done_so_far % 5 == 0 or done_so_far == total):
            answered_now = sum(1 for p in predictions if p.get("success", False))
            em_now = (correct_count / answered_now * 100) if answered_now > 0 else 0.0
            nm_now = (nm_count     / answered_now * 100) if answered_now > 0 else 0.0
            elapsed = time.time() - pipeline_start
            print(
                f"  [{done_so_far:>3}/{total}] {'OK' if is_correct else '--'} "
                f"{qid:<12} | EM: {em_now:.1f}% | NM: {nm_now:.1f}% | "
                f"Elapsed: {elapsed/60:.1f}m"
            )

    # ── Final results ──────────────────────────────
    em         = calculate_em(predictions)
    total_time = time.time() - pipeline_start

    print()
    print("=" * 60)
    print(f"  RESULT: retriever={retriever_type} | K={top_k} | prompt={prompt_type}")
    print(f"  Provider      : {ACTIVE_PROVIDER}")
    print(f"  Model         : {model_name}")
    print(f"  Strict EM%    : {em['em_score']}%  ({em['correct']}/{em['total']})")
    print(f"  Lenient EM%   : {round(em['correct']/em['answered']*100,2) if em['answered']>0 else 0}%  ({em['correct']}/{em['answered']} answered)")
    print(f"  NM Score      : {em['nm_score']}%")
    print(f"  Failed        : {em['failed']}")
    print(f"  Total time    : {total_time/60:.1f} min")
    print(f"  Output        : {pred_file}")
    print("=" * 60 + "\n")

    logs = {
        "experiment": {
            "retriever":   retriever_type,
            "top_k":       top_k,
            "prompt":      prompt_type,
            "provider":    ACTIVE_PROVIDER,
            "model":       model_name,
            "sample_size": sample_size,
        },
        "results": em,
        "timing": {
            "total_minutes":   round(total_time / 60, 2),
            "avg_inference_s": round(sum(inf_times) / len(inf_times), 2) if inf_times else 0,
            "avg_retrieval_s": round(sum(ret_times) / len(ret_times), 2) if ret_times else 0,
        },
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return em


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # # ════════════════════════════════════════
    # # تجربة 1 — Fixed Term-Count (B3-EXP-FIXED)
    # # نفس B3-EXP10 بس بكاش مُصلح (30 مصطلح نظيف)
    # # ════════════════════════════════════════
    # EXP_RETRIEVER = "expanded"
    # EXP_TOP_K     = 5
    # EXP_PROMPT    = "direct"

    # # ════════════════════════════════════════
    # # تجربة 2 — Structured Weighted (B3-EXP-STRUCT)
    # # مصطلحات مصنّفة بأوزان حسب الفئة الطبية
    # # ════════════════════════════════════════
    # EXP_RETRIEVER = "weighted_struct"
    # EXP_TOP_K     = 3
    # EXP_PROMPT    = "direct"

    # # ════════════════════════════════════════
    # # تجربة 3 — hybrid_scored 
    # # 0.3 × MedCPT + 0.7 × weighted term count 
    # # ════════════════════════════════════════
    # EXP_RETRIEVER = "hybrid_scored"
    # EXP_TOP_K     = 3
    # EXP_PROMPT    = "direct"
    

    # ════════════════════════════════════════
    # تجربة 4 — gold_chain (للتحليل النظري فقط — upper bound)
    # تصفية بناءً على الإجابة الصحيحة → تحتاج معرفة الجواب
    # ════════════════════════════════════════
    # EXP_RETRIEVER = "gold_chain"
    # EXP_TOP_K     = 5
    # EXP_PROMPT    = "direct"

    # # ════════════════════════════════════════
    # # تجربة 5 — query_drug (جديدة — واقعية)
    # # تصفية بناءً على اسم/ID الدواء في السؤال
    # # لا تحتاج معرفة الإجابة → قابلة للإنتاج
    # # ════════════════════════════════════════
    # EXP_RETRIEVER = "query_drug"
    # EXP_TOP_K     = 5
    # EXP_PROMPT    = "direct"


    # # ════════════════════════════════════════
    # # تجربة 6 — Combined Retriever 
    # # score = term_count_score + 2 × (query_drug_name in doc)
    # # ════════════════════════════════════════
    EXP_RETRIEVER = "combined"
    EXP_TOP_K     = 3
    EXP_PROMPT    = "direct"


    EXP_SAMPLE = DIAG_SAMPLE_SIZE   # 50 للاختبار، None للكامل

    run_experiment(
        retriever_type = EXP_RETRIEVER,
        top_k          = EXP_TOP_K,
        prompt_type    = EXP_PROMPT,
        sample_size    = EXP_SAMPLE,
    )