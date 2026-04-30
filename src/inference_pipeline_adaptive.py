"""
src/inference_pipeline_adaptive.py
=====================================
Biomedical Multi-Hop QA — Adaptive Experiments (B4-EXP10+)

Experiments:
================================================================
  B4-EXP10 : Adaptive Top-K (Internal Only)
             Based on B4-EXP4 (33.33%) + Adaptive K
             k changes dynamically based on max similarity score

  B4-EXP11 : Wikipedia V2 External Retrieval (Wiki Only)
             Wikipedia V2 as external source instead of internal supports
             Uses bridge_info + candidate_names for targeted retrieval
             Applies hybrid_scored on Wikipedia V2 chunks

  B4-EXP12 : Internal + WikiV2 + Source-Aware Scoring + Adaptive K
             Internal supports + Wikipedia V2 chunks
             Source-Aware Scoring via score_combined_retrieval()
             Adaptive K applied on combined scored results

  B4-EXP13 : Signal-Based Adaptive Filter
             Signal-based filtering directly on B4-EXP4 base
             Drug_A AND (Candidate OR Bridge) signal filter
             Score-gap filter with adaptive k (max=5)

How it builds on B4-EXP4:
  - Same bridge extraction (from bridge_cache.json)
  - Same PROMPT_COT_WITH_GUIDED (best prompt at 33.33%)
  - Same hybrid_scored scoring
  - Addition: Adaptive K + Wikipedia V2 + Source-Aware Scoring

Key Changes from Previous Version:
  - Wikipedia V1 → V2 (retrieve_wikipedia_v2 with bridge_info + candidate_names)
  - B4-EXP12 (Combined) now uses Source-Aware Scoring instead of naive concatenation
  - Added _adaptive_k_on_scored() helper for pre-scored combined results
  - Renumbered experiments: B4-EXP10, B4-EXP11, B4-EXP12, B4-EXP13
  - Removed run_wiki_task_aware (old EXPERIMENT 5, redundant)

Usage:
    # B4-EXP10 — Adaptive K (Internal Only)
    py -3.10 src/inference_pipeline_adaptive.py --exp atk

    # B4-EXP11 — Wikipedia V2 Only
    py -3.10 src/inference_pipeline_adaptive.py --exp wiki

    # B4-EXP12 — Combined + Source-Aware Scoring (Recommended)
    py -3.10 src/inference_pipeline_adaptive.py --exp combined

    # B4-EXP13 — Signal-Based Filter
    py -3.10 src/inference_pipeline_adaptive.py --exp signal

    # With custom thresholds (after running score_diagnostics.py)
    py -3.10 src/inference_pipeline_adaptive.py --exp atk --high 0.70 --mid 0.45 --low 0.25

    # Test sample
    py -3.10 src/inference_pipeline_adaptive.py --exp combined --sample 50
"""

import json, os, sys, time, re, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE, OUTPUTS_DIR, DIAG_SAMPLE_SIZE, OLLAMA_MODEL_NAME,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MAX_TOKENS, LLM_NUM_CTX,
    ACTIVE_PROVIDER,
)

# ── LLM runner (supports ollama + API) ──
if ACTIVE_PROVIDER == "ollama":
    from src.llm_runner import get_ollama_client as _get_client, check_model_available as _check_model, run_inference as _run_inference
    def get_client(): return _get_client()
    def check_model(c): return _check_model(c)
    def do_inference(c, p, q): return _run_inference(c, p, q)
    def active_model(): return OLLAMA_MODEL_NAME
else:
    from src.llm_runner_api import get_api_client as _get_client, check_model_available as _check_model, run_inference as _run_inference
    from config.settings import API_MODEL
    def get_client(): return _get_client()
    def check_model(c): return _check_model(c)
    def do_inference(c, p, q): return _run_inference(c, p, q)
    def active_model(): return API_MODEL

# ── Core components ──
from src.query_expander import expand_query
from src.query_expander_structured import get_weighted_terms
from src.retriever_hybrid_scored import retrieve_hybrid_scored
from src.retriever_adaptive_k import retrieve_adaptive_k, decide_k_from_gap
from src.retriever_wikipedia_v2 import retrieve_wikipedia_v2, prefetch_wikipedia_v2
from src.retriever_wikipedia_v2 import score_combined_retrieval
from src.prompt_builder import extract_drug_id

# ──────────────────────────────────────────────────────
# BRIDGE CACHE
# ──────────────────────────────────────────────────────

BRIDGE_CACHE_FILE = os.path.join(OUTPUTS_DIR, "bridge_cache.json")
BRIDGE_MAX_TOKENS = 150


def _load_bridge_cache():
    if os.path.exists(BRIDGE_CACHE_FILE):
        try:
            with open(BRIDGE_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}


def _get_bridge_from_cache(record: dict, cache: dict) -> str:
    """
    [FIXED v2] Fetch bridge entity from saved cache.

    التغيير الجوهري عن النسخة السابقة:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    النسخة القديمة كانت تستخدم:
        raw = cache.get("mechanism::{drug}") OR cache.get("2hop_mechanism::{drug}")

    المشكلة: `2hop_mechanism::` يحتوي على ردود طويلة غير منظّفة من نموذج قديم
    مثال: "Based on the provided evidence, specifically citation [2]..."
    هذا كان يُعطي bridge رديء يُضرّ الاسترجاع → EXP10 كان 26% بدل 33.33%

    الحل: استخدام `mechanism::` فقط — نفس ما يفعله B4-EXP4 (33.33%)
    التحقق: كل 221 دواء في الداتاسيت لديه مفتاح mechanism:: في bridge_cache الحالي

    ★ FIX: mechanism:: ONLY — same key as B4-EXP4 baseline (33.33% EM)
    """
    drug_name = record.get("query_drug_name", "")
    # استخدام mechanism:: فقط — نفس B4-EXP4 بالضبط
    key = f"mechanism::{drug_name}"
    raw = cache.get(key, "")
    if not raw:
        return ""
    # تنظيف بسيط: احذف prefix إن وُجد، خذ أول سطر
    raw = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE).strip()
    first_line = raw.split("\n")[0].strip()
    return first_line[:100]


def _bridge_to_weighted_terms(bridge_info: str) -> list:
    """
    Convert bridge phrase into individual weighted terms — same as B4-EXP4.

    B4-EXP4 in inference_pipeline_entity_bridging.py:
        clean = bridge_info.replace('inhibits','').replace('blocks','')...
        bridge_terms = [w for w in clean.split() if len(w) > 2]
        # each term -> weight=5.0 individually

    Previous problem: sending the full phrase -> no matches in documents
    Solution: split into words -> each word gets weight=5.0 independently
    """
    if not bridge_info:
        return []
    clean = (bridge_info
             .replace("inhibits", "")
             .replace("blocks", "")
             .replace("acts", "")
             .replace("via", "")
             .replace("through", ""))
    terms = [w.strip() for w in clean.split() if len(w.strip()) > 2]
    return [{"term": t, "weight": 5.0, "category": "mechanism"} for t in terms]


def _run_bridge_inference(client, drug_name: str, docs_text: str, qid: str) -> str:
    """Extract bridge with LLM if not found in cache — works with Ollama and API providers."""
    prompt = f"""You are a biomedical expert. What is the PRIMARY mechanism of action of {drug_name}?

Evidence:
{docs_text}

Answer in ONE short phrase (e.g., "inhibits CYP2D6", "blocks MAO-A"):
MECHANISM:"""
    result = {
        "question_id": qid, "raw_response": "",
        "inference_time": 0.0, "success": False, "error": ""
    }
    for attempt in range(2):
        try:
            inf_result = do_inference(client, prompt, f"{qid}_bridge")
            raw = inf_result.get("raw_response", "").strip()
            raw = re.sub(r"<think.*?</think >", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE).strip()
            if raw:
                return raw
        except Exception as e:
            result["error"] = str(e)
            if attempt == 0:
                time.sleep(2)
    return ""


# ──────────────────────────────────────────────────────
# PROMPT — Same as B4-EXP4 (PROMPT_COT_ENRICHED = 33.33%)
# ──────────────────────────────────────────────────────

def _format_docs(retrieved: list) -> str:
    """Format documents — same as B4-EXP4."""
    if not retrieved:
        return "No supporting evidence available."
    parts = []
    for r in retrieved:
        text = r["text"] if isinstance(r, dict) else str(r)
        if len(text) > 400:
            text = text[:400] + "..."
        rank = r.get("rank", "?") if isinstance(r, dict) else "?"
        parts.append(f"[{rank}] {text}")
    return "\n".join(parts)


# This is the verbatim prompt from inference_pipeline_entity_bridging.py
# PROMPT_COT_WITH_GUIDED = PROMPT_COT_ENRICHED (alias)
_PROMPT_COT_ENRICHED_TEMPLATE = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- {drug_name} has mechanism: {bridge_info}
- Look for a candidate that is AFFECTED BY this mechanism (metabolized by same enzyme, targets same receptor, shares same pathway)
- The interacting drug does NOT need to have the same mechanism — it just needs to be affected by it
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""


def build_prompt_cot_enriched(record: dict, retrieved: list, bridge_info: str) -> str:
    """
    PROMPT_COT_ENRICHED — verbatim from B4-EXP4 (33.33%).
    No modifications to this prompt.
    """
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
    bridge_line = bridge_info if bridge_info else "unknown"
    return _PROMPT_COT_ENRICHED_TEMPLATE.format(
        docs=docs,
        drug_name=drug_name,
        drug_id=drug_id,
        bridge_info=bridge_line,
        candidates_text=candidates_text,
    )


# ──────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────

def load_data(sample=None):
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)
    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data[:sample] if sample else data


def load_existing(pred_file):
    if not os.path.exists(pred_file):
        return {}
    try:
        with open(pred_file, encoding="utf-8") as f:
            existing = json.load(f)
        done = {r["question_id"]: r for r in existing}
        print(f"  [INFO] Resuming — {len(done)} already done")
        return done
    except:
        return {}


def save_preds(preds, path):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)


def calculate_em(preds):
    total   = len(preds)
    failed  = sum(1 for p in preds if not p.get("success", False))
    correct = sum(
        1 for p in preds
        if p.get("success") and
           p.get("prediction", "").strip().upper() == p.get("answer", "").strip().upper() and
           p.get("prediction", "").strip()
    )
    answered = total - failed
    return {
        "total":    total,
        "answered": answered,
        "correct":  correct,
        "failed":   failed,
        "em_score": round(correct / answered * 100, 2) if answered > 0 else 0.0,
    }


def _model_short(name):
    name = re.sub(r"[/:\\.]", "_", name).strip("_")
    return name[-30:] if len(name) > 30 else name


# ──────────────────────────────────────────────────────
# HELPER: Adaptive K on Pre-Scored Results
# ──────────────────────────────────────────────────────

def _adaptive_k_on_scored(
    scored_docs: list,
    gap_high: float = 0.3085,
    gap_mid: float = 0.1301,
    gap_low: float = 0.0577,
) -> tuple:
    """
    Adaptive K on pre-scored documents (from Source-Aware Scoring).
    Same gap-based logic as retrieve_adaptive_k but works on already-scored docs.

    Args:
        scored_docs: List of dicts with 'score' key, already sorted by score descending
        gap_high: p75 threshold -> k=2
        gap_mid:  p50 threshold (used for mid tier)
        gap_low:  p25 threshold -> k=5

    Returns:
        (selected_docs, adaptive_k, gap_value, decision_info)
    """
    from src.retriever_adaptive_k import MIN_K, DEFAULT_K, MAX_K

    if not scored_docs:
        return [], MIN_K, 0.0, {"reason": "no_docs", "k": MIN_K}

    scores = [d.get("score", 0.0) for d in scored_docs]

    if len(scores) < 2:
        return scored_docs[:MIN_K], MIN_K, scores[0] if scores else 0.0, {"reason": "single_doc", "k": MIN_K}

    gap = scores[0] - scores[1]
    from src.retriever_adaptive_k import decide_k_from_gap
    adaptive_k, tier, _ = decide_k_from_gap(scores[0], scores[1], gap_high, gap_mid, gap_low)

    adaptive_k = max(MIN_K, min(MAX_K, adaptive_k))
    adaptive_k = min(adaptive_k, len(scored_docs))

    selected = scored_docs[:adaptive_k]

    decision_info = {
        "method": "score_gap_on_combined",
        "adaptive_k": adaptive_k,
        "gap": round(gap, 4),
        "tier": tier,
        "top1_score": round(scores[0], 4),
        "all_scores": [round(s, 4) for s in scores[:10]],
        "n_available": len(scored_docs),
    }

    return selected, adaptive_k, gap, decision_info


# ──────────────────────────────────────────────────────
# EXPERIMENT B4-EXP10 — ADAPTIVE TOP-K (Internal Only)
# ──────────────────────────────────────────────────────

def run_adaptive_topk(
    sample_size=None,
    gap_high=0.3085,   # p75 from 342 questions (score_diagnostics)
    gap_mid=0.1301,    # p50 from 342 questions
    gap_low=0.0577,    # p25 from 342 questions
    verbose=True,
):
    """
    Adaptive Top-K on internal supports — B4-EXP10.
    Based on B4-EXP4 entirely + Adaptive K instead of fixed K=3.

    Logic: Score-Gap (no Absolute Threshold):
      gap = score[rank1] - score[rank2]
      gap >= 0.3085 -> k=2  (25% of questions — rank1 is clear)
      gap >= 0.0577 -> k=3  (50% of questions — normal case)
      gap <  0.0577 -> k=5  (25% of questions — docs are close)
    """
    exp_name  = "adaptive_atk"
    model_s   = _model_short(active_model())
    pred_file = os.path.join(OUTPUTS_DIR, f"adaptive_{exp_name}_{model_s}_predictions.json")
    logs_file = pred_file.replace("_predictions.json", "_logs.json")

    print("\n" + "="*60)
    print("  EXPERIMENT: Adaptive Top-K / B4-EXP10 (Internal Only)")
    print("="*60)
    print(f"  Model      : {active_model()}")
    print(f"  Method     : Score-Gap (data-driven thresholds from score_diagnostics)")
    print(f"  Gaps       : HIGH={gap_high} -> k=2 | MID/LOW -> k=3 | VERY_LOW={gap_low} -> k=5")
    print(f"  Expected   : k=2 ~25% | k=3 ~50% | k=5 ~25%")
    print(f"  Output     : {pred_file}")
    print()

    pipeline_start = time.time()
    client         = get_client()
    if not check_model(client):
        sys.exit(1)

    data          = load_data(sample_size)
    bridge_cache  = _load_bridge_cache()
    total         = len(data)
    existing      = load_existing(pred_file)
    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct", False))
    inf_times     = []
    k_distribution = {2: 0, 3: 0, 5: 0}

    print(f"  Questions  : {total} | Done: {len(existing)} | Remaining: {total-len(existing)}")
    print()

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            k_used = existing[qid].get("adaptive_k", 3)
            k_distribution[k_used] = k_distribution.get(k_used, 0) + 1
            continue

        drug_name = record.get("query_drug_name", "")
        query     = record.get("query", "")
        supports  = record.get("supports", [])
        ans_name  = record.get("answer_name", "")

        # ── Get bridge (same as B4-EXP4) ──
        bridge = _get_bridge_from_cache(record, bridge_cache)

        # ── Build weighted terms (same as B4-EXP4) ──
        exp_result     = expand_query(drug_name)
        flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
        weighted_terms = get_weighted_terms(drug_name) or []

        if bridge:
            # Same as B4-EXP4: split bridge phrase into words, each gets weight=5.0
            weighted_terms = _bridge_to_weighted_terms(bridge) + weighted_terms

        # ── Adaptive K Retrieval (Gap-Based) ──
        retrieved, adaptive_k, gap_val, decision_info = retrieve_adaptive_k(
            query=query,
            supports=supports,
            drug_name=drug_name,
            flat_terms=flat_terms,
            weighted_terms=weighted_terms,
            gap_high=gap_high,
            gap_mid=gap_mid,
            gap_low=gap_low,
        )

        k_distribution[adaptive_k] = k_distribution.get(adaptive_k, 0) + 1

        # ── Prompt + Inference (same as B4-EXP4) ──
        prompt     = build_prompt_cot_enriched(record, retrieved, bridge)
        inf_result = do_inference(client, prompt, qid)
        inf_times.append(inf_result["inference_time"])

        if inf_result["success"]:
            raw = re.sub(r"<think.*?</think >", "", inf_result["raw_response"], flags=re.DOTALL).strip()
            if raw:
                inf_result["raw_response"] = raw

        prediction = extract_drug_id(inf_result["raw_response"], record["candidates"]) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == record["answer"].upper() and inf_result["success"]
        if is_correct:
            correct_count += 1

        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "answer":          record["answer"],
            "answer_name":     ans_name,
            "is_correct":      is_correct,
            "bridge_info":     bridge,
            "adaptive_k":      adaptive_k,
            "gap_value":       round(gap_val, 4),
            "k_tier":          decision_info.get("tier", ""),
            "top1_score":      decision_info.get("top1_score", 0.0),
            "all_scores":      decision_info.get("all_scores", []),
            "raw_response":    inf_result["raw_response"],
            "model":           active_model(),
            "provider":        ACTIVE_PROVIDER,
            "experiment":      "b4exp10_adaptive_atk",
            "success":         inf_result["success"],
            "error":           inf_result["error"],
            "inference_time":  inf_result["inference_time"],
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 10 == 0 or done == total):
            answered_now = sum(1 for p in predictions if p.get("success", False))
            em_now = correct_count / answered_now * 100 if answered_now > 0 else 0.0
            elapsed = time.time() - pipeline_start
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} "
                  f"k={adaptive_k}({decision_info.get('tier','?')}) gap={gap_val:.3f} | EM: {em_now:.1f}% | {elapsed/60:.1f}m")

    # ── Final results ──
    em         = calculate_em(predictions)
    total_time = time.time() - pipeline_start

    print()
    print("="*60)
    print(f"  RESULT: Adaptive Top-K / B4-EXP10")
    print(f"  EM Score  : {em['em_score']}%  ({em['correct']}/{em['total']})")
    print(f"  Baseline  : 33.33% (B4-EXP4) | {'↑ IMPROVED' if em['em_score'] > 33.33 else '↓'}")
    print(f"  Time      : {total_time/60:.1f} min")
    print()
    print("  K Distribution:")
    for k, cnt in sorted(k_distribution.items()):
        pct = cnt / total * 100 if total else 0
        print(f"    k={k}: {cnt:>4} ({pct:.1f}%)")
    print("="*60 + "\n")

    logs = {
        "experiment":   "b4exp10_adaptive_atk",
        "model":        active_model(),
        "method":       "score_gap",
        "gap_thresholds": {"high": gap_high, "mid": gap_mid, "low": gap_low},
        "results":      em,
        "k_distribution": k_distribution,
        "baseline_em":  33.33,
        "timing": {
            "total_minutes":   round(total_time / 60, 2),
            "avg_inference_s": round(sum(inf_times) / len(inf_times), 2) if inf_times else 0,
        },
    }
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return em


# ──────────────────────────────────────────────────────
# EXPERIMENT B4-EXP11 — WIKIPEDIA V2 ONLY
# ──────────────────────────────────────────────────────

def run_wikipedia_only(
    sample_size=None,
    top_k=3,
    methodology="all",
    verbose=True,
):
    """
    Wikipedia V2 as external retrieval source only (instead of internal supports).
    Uses retrieve_wikipedia_v2 with bridge_info + candidate_names for targeted retrieval.
    Applies hybrid_scored on Wikipedia V2 chunks.
    k=3 (fixed — for fair comparison with B4-EXP4).

    methodology controls which Wikipedia retrieval methods to apply:
      "all"             ← كل المنهجيات معاً (B4-EXP11)
      "query2doc"       ← Query2Doc فقط (B4-EXP11a)
      "sequential"      ← Sequential Retrieval فقط (B4-EXP11b)
      "medical_sections" ← Medical Sections فقط (B4-EXP11c)
      "entity_pair"     ← Entity Pair Search فقط (B4-EXP11d)
    """
    method_suffix = f"_{methodology}" if methodology != "all" else ""
    exp_name  = f"b4exp11_wiki_v2{method_suffix}"
    model_s   = _model_short(active_model())
    pred_file = os.path.join(OUTPUTS_DIR, f"adaptive_{exp_name}_k{top_k}_{model_s}_predictions.json")
    logs_file = pred_file.replace("_predictions.json", "_logs.json")

    print("\n" + "="*60)
    print("  EXPERIMENT: Wikipedia V2 External Retrieval / B4-EXP11")
    print("="*60)
    print(f"  Model      : {active_model()}")
    print(f"  K          : {top_k}")
    print(f"  Output     : {pred_file}")
    print()

    pipeline_start = time.time()
    client         = get_client()
    if not check_model(client):
        sys.exit(1)

    # Prefetch Wikipedia V2 for all drugs
    data = load_data(sample_size)
    bridge_cache = _load_bridge_cache()
    print("  [WikiV2] Prefetching Wikipedia V2 articles...")
    prefetch_wikipedia_v2(data, bridge_cache=bridge_cache, methodology=methodology, verbose=verbose)

    total         = len(data)
    existing      = load_existing(pred_file)
    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct", False))
    inf_times     = []
    wiki_hit_count = 0

    print(f"\n  Questions  : {total} | Done: {len(existing)} | Remaining: {total-len(existing)}")
    print()

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            if existing[qid].get("wiki_chunks", 0) > 0:
                wiki_hit_count += 1
            continue

        drug_name  = record.get("query_drug_name", "")
        query      = record.get("query", "")
        ans_name   = record.get("answer_name", "")
        bridge     = _get_bridge_from_cache(record, bridge_cache)
        candidate_names = record.get("candidate_names", record["candidates"])

        # ── Wikipedia V2 chunks ──
        wiki_chunks = retrieve_wikipedia_v2(
            drug_name,
            bridge_info=bridge,
            candidate_names=candidate_names,
            methodology=methodology,
        )
        if wiki_chunks:
            wiki_hit_count += 1

        # ── Build weighted terms ──
        exp_result     = expand_query(drug_name)
        flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
        weighted_terms = get_weighted_terms(drug_name) or []

        if bridge:
            # Same as B4-EXP4: each word from bridge phrase becomes a separate term with weight 5.0
            weighted_terms = _bridge_to_weighted_terms(bridge) + weighted_terms

        # ── Retrieve from Wikipedia V2 (if available) or fallback to internal ──
        if wiki_chunks:
            retrieved = retrieve_hybrid_scored(
                query=query,
                supports=wiki_chunks,   # <- Wikipedia V2 instead of internal supports
                drug_name=drug_name,
                flat_terms=flat_terms,
                weighted_terms=weighted_terms,
                top_k=top_k,
            )
        else:
            # Fallback to internal supports if Wikipedia V2 is empty
            supports = record.get("supports", [])
            retrieved = retrieve_hybrid_scored(
                query=query,
                supports=supports,
                drug_name=drug_name,
                flat_terms=flat_terms,
                weighted_terms=weighted_terms,
                top_k=top_k,
            )

        # ── Prompt + Inference ──
        prompt     = build_prompt_cot_enriched(record, retrieved, bridge)
        inf_result = do_inference(client, prompt, qid)
        inf_times.append(inf_result["inference_time"])

        if inf_result["success"]:
            raw = re.sub(r"<think.*?</think >", "", inf_result["raw_response"], flags=re.DOTALL).strip()
            if raw:
                inf_result["raw_response"] = raw

        prediction = extract_drug_id(inf_result["raw_response"], record["candidates"]) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == record["answer"].upper() and inf_result["success"]
        if is_correct:
            correct_count += 1

        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "answer":          record["answer"],
            "answer_name":     ans_name,
            "is_correct":      is_correct,
            "bridge_info":     bridge,
            "wiki_chunks":     len(wiki_chunks),
            "source":          "wiki_v2" if wiki_chunks else "internal_fallback",
            "raw_response":    inf_result["raw_response"],
            "model":           active_model(),
            "provider":        ACTIVE_PROVIDER,
            "experiment":      "b4exp11_wiki_v2",
            "top_k":           top_k,
            "success":         inf_result["success"],
            "error":           inf_result["error"],
            "inference_time":  inf_result["inference_time"],
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 10 == 0 or done == total):
            answered_now = sum(1 for p in predictions if p.get("success", False))
            em_now = correct_count / answered_now * 100 if answered_now > 0 else 0.0
            elapsed = time.time() - pipeline_start
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} "
                  f"wiki={len(wiki_chunks)} | EM: {em_now:.1f}% | {elapsed/60:.1f}m")

    # ── Final results ──
    em         = calculate_em(predictions)
    total_time = time.time() - pipeline_start

    print()
    print("="*60)
    print(f"  RESULT: Wikipedia V2 External Retrieval / B4-EXP11")
    print(f"  EM Score  : {em['em_score']}%  ({em['correct']}/{em['total']})")
    print(f"  Wiki hits : {wiki_hit_count}/{total} ({wiki_hit_count/total*100:.1f}%)")
    print(f"  Baseline  : 33.33% (B4-EXP4) | {'↑ IMPROVED' if em['em_score'] > 33.33 else '↓'}")
    print(f"  Time      : {total_time/60:.1f} min")
    print("="*60 + "\n")

    logs = {
        "experiment":    "b4exp11_wiki_v2",
        "model":         active_model(),
        "wiki_version":  "v2",
        "top_k":         top_k,
        "results":       em,
        "wiki_hit_rate": round(wiki_hit_count / total * 100, 2) if total else 0,
        "baseline_em":   33.33,
        "timing":        {"total_minutes": round(total_time / 60, 2),
                          "avg_inference_s": round(sum(inf_times) / len(inf_times), 2) if inf_times else 0},
    }
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return em


# ──────────────────────────────────────────────────────
# EXPERIMENT B4-EXP12 — COMBINED (Internal + WikiV2 + Source-Aware Scoring + Adaptive K)
# ──────────────────────────────────────────────────────

def run_combined(
    sample_size=None,
    gap_high=0.3085,
    gap_mid=0.1301,
    gap_low=0.0577,
    methodology="all",
    verbose=True,
):
    """
    Full combination: internal supports + Wikipedia V2 + Source-Aware Scoring + Adaptive K.

    Pipeline:
      1. Internal retrieval via retrieve_hybrid_scored (top_k=5)
      2. Wikipedia V2 retrieval via retrieve_wikipedia_v2 (with bridge_info + candidate_names)
      3. Source-Aware Scoring via score_combined_retrieval() — scores docs from both sources
         considering their source type, bridge relevance, and candidate mentions
      4. Adaptive K on combined scored results via _adaptive_k_on_scored()

    methodology controls Wikipedia retrieval methods (same as run_wikipedia_only).

    This replaces the old naive concatenation (supports + wiki_chunks) with proper
    cross-source scoring that accounts for document provenance and task relevance.
    """
    method_suffix = f"_{methodology}" if methodology != "all" else ""
    exp_name  = f"b4exp12_combined_v2{method_suffix}"
    model_s   = _model_short(active_model())
    pred_file = os.path.join(OUTPUTS_DIR, f"adaptive_{exp_name}_{model_s}_predictions.json")
    logs_file = pred_file.replace("_predictions.json", "_logs.json")

    print("\n" + "="*60)
    print("  EXPERIMENT: Combined + Source-Aware Scoring / B4-EXP12")
    print("="*60)
    print(f"  Model      : {active_model()}")
    print(f"  Method     : Internal + WikiV2 + Source-Aware Scoring + Adaptive K")
    print(f"  Gap        : HIGH={gap_high} | MID={gap_mid} | LOW={gap_low}")
    print(f"  Output     : {pred_file}")
    print()

    pipeline_start = time.time()
    client         = get_client()
    if not check_model(client):
        sys.exit(1)

    data = load_data(sample_size)
    bridge_cache = _load_bridge_cache()
    print("  [WikiV2] Prefetching Wikipedia V2 articles...")
    prefetch_wikipedia_v2(data, bridge_cache=bridge_cache, methodology=methodology, verbose=verbose)

    total         = len(data)
    existing      = load_existing(pred_file)
    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct", False))
    inf_times     = []
    k_distribution = {2: 0, 3: 0, 5: 0}
    wiki_added_count = 0

    print(f"\n  Questions  : {total} | Done: {len(existing)} | Remaining: {total-len(existing)}")
    print()

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            k_used = existing[qid].get("adaptive_k", 3)
            k_distribution[k_used] = k_distribution.get(k_used, 0) + 1
            continue

        drug_name  = record.get("query_drug_name", "")
        query      = record.get("query", "")
        supports   = record.get("supports", [])
        ans_name   = record.get("answer_name", "")
        bridge     = _get_bridge_from_cache(record, bridge_cache)
        candidate_names = record.get("candidate_names", record["candidates"])

        # ── Build weighted terms ──
        exp_result     = expand_query(drug_name)
        flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
        weighted_terms = get_weighted_terms(drug_name) or []

        if bridge:
            weighted_terms = _bridge_to_weighted_terms(bridge) + weighted_terms

        # ── Step 1: Internal retrieval ──
        internal_retrieved = retrieve_hybrid_scored(
            query=query,
            supports=supports,
            drug_name=drug_name,
            flat_terms=flat_terms,
            weighted_terms=weighted_terms,
            top_k=5,
        )

        # ── Step 2: Wikipedia V2 retrieval ──
        wiki_chunks = retrieve_wikipedia_v2(
            drug_name,
            bridge_info=bridge,
            candidate_names=candidate_names,
            methodology=methodology,
        )
        n_wiki = len(wiki_chunks)
        if wiki_chunks:
            wiki_added_count += 1

        # ── Step 3: Source-Aware Scoring (from retriever_wikipedia_v2.py) ──
        combined = score_combined_retrieval(
            internal_retrieved=internal_retrieved,
            wiki_retrieved=wiki_chunks,
            drug_name=drug_name,
            bridge_info=bridge,
            candidate_names=candidate_names,
        )

        # ── Step 4: Adaptive K on combined scored results ──
        retrieved, adaptive_k, gap_val, decision_info = _adaptive_k_on_scored(
            combined, gap_high=gap_high, gap_mid=gap_mid, gap_low=gap_low
        )

        k_distribution[adaptive_k] = k_distribution.get(adaptive_k, 0) + 1

        # ── Prompt + Inference ──
        prompt     = build_prompt_cot_enriched(record, retrieved, bridge)
        inf_result = do_inference(client, prompt, qid)
        inf_times.append(inf_result["inference_time"])

        if inf_result["success"]:
            raw = re.sub(r"<think.*?</think >", "", inf_result["raw_response"], flags=re.DOTALL).strip()
            if raw:
                inf_result["raw_response"] = raw

        prediction = extract_drug_id(inf_result["raw_response"], record["candidates"]) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == record["answer"].upper() and inf_result["success"]
        if is_correct:
            correct_count += 1

        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "answer":          record["answer"],
            "answer_name":     ans_name,
            "is_correct":      is_correct,
            "bridge_info":     bridge,
            "adaptive_k":      adaptive_k,
            "gap_value":       round(gap_val, 4),
            "k_tier":          decision_info.get("tier", ""),
            "top1_score":      decision_info.get("top1_score", 0.0),
            "wiki_chunks":     n_wiki,
            "n_combined_scored": len(combined),
            "scoring_method":  "source_aware",
            "raw_response":    inf_result["raw_response"],
            "model":           active_model(),
            "provider":        ACTIVE_PROVIDER,
            "experiment":      "b4exp12_combined_v2",
            "success":         inf_result["success"],
            "error":           inf_result["error"],
            "inference_time":  inf_result["inference_time"],
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 10 == 0 or done == total):
            answered_now = sum(1 for p in predictions if p.get("success", False))
            em_now = correct_count / answered_now * 100 if answered_now > 0 else 0.0
            elapsed = time.time() - pipeline_start
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} "
                  f"k={adaptive_k}({decision_info.get('tier','?')}) wiki={n_wiki} | EM: {em_now:.1f}% | {elapsed/60:.1f}m")

    # ── Final results ──
    em         = calculate_em(predictions)
    total_time = time.time() - pipeline_start

    print()
    print("="*60)
    print(f"  RESULT: Combined + Source-Aware Scoring / B4-EXP12")
    print(f"  EM Score  : {em['em_score']}%  ({em['correct']}/{em['total']})")
    print(f"  Wiki coverage: {wiki_added_count}/{total} ({wiki_added_count/total*100:.1f}%)")
    print(f"  Baseline  : 33.33% (B4-EXP4) | {'↑ IMPROVED' if em['em_score'] > 33.33 else '↓'}")
    print(f"  Time      : {total_time/60:.1f} min")
    print()
    print("  K Distribution:")
    for k, cnt in sorted(k_distribution.items()):
        pct = cnt / total * 100 if total else 0
        print(f"    k={k}: {cnt:>4} ({pct:.1f}%)")
    print("="*60 + "\n")

    logs = {
        "experiment":    "b4exp12_combined_v2",
        "model":         active_model(),
        "method":        "source_aware_scoring",
        "wiki_version":  "v2",
        "gap_thresholds": {"high": gap_high, "mid": gap_mid, "low": gap_low},
        "results":       em,
        "k_distribution": k_distribution,
        "wiki_coverage":  round(wiki_added_count / total * 100, 2) if total else 0,
        "baseline_em":   33.33,
        "timing":        {"total_minutes": round(total_time / 60, 2),
                          "avg_inference_s": round(sum(inf_times) / len(inf_times), 2) if inf_times else 0},
    }
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return em


# ──────────────────────────────────────────────────────
# EXPERIMENT B4-EXP13 — ADAPTIVE K + SIGNAL-BASED FILTER
# (Applying Signal-Based Filter directly on B4-EXP4)
# ──────────────────────────────────────────────────────

def run_adaptive_signal(
    sample_size=None,
    fetch_k=10,
    max_k=5,
    gap_ratio=0.6,
    verbose=True,
):
    """
    B4-EXP4 + Signal-Based Adaptive K (no arbitrary thresholds).

    Steps:
      1. Same as B4-EXP4 (bridge extraction + hybrid scoring)
      2. Fetch fetch_k=10 instead of k=3
      3. Filter: Drug_A AND (Candidate OR Bridge)  <- no threshold
      4. Boost for documents mentioning Candidate or Bridge
      5. Score-gap filter -> k is determined automatically (max=5)

    Goal: Compare with B4-EXP4 (33.33%) using a more logical filter
    """
    exp_name  = "b4exp13_signal"
    model_s   = _model_short(active_model())
    pred_file = os.path.join(OUTPUTS_DIR, f"adaptive_{exp_name}_{model_s}_predictions.json")
    logs_file = pred_file.replace("_predictions.json", "_logs.json")

    print("\n" + "="*60)
    print("  EXPERIMENT: Signal-Based Adaptive Filter / B4-EXP13")
    print("="*60)
    print(f"  Model      : {active_model()}")
    print(f"  Fetch K    : {fetch_k} -> Signal filter -> Score-gap (max={max_k})")
    print(f"  Filter     : Drug_A AND (Candidate OR Bridge)")
    print(f"  Output     : {pred_file}")
    print()

    pipeline_start = time.time()
    client         = get_client()
    if not check_model(client):
        sys.exit(1)

    from src.retriever_task_aware import retrieve_adaptive_signal

    data          = load_data(sample_size)
    bridge_cache  = _load_bridge_cache()
    total         = len(data)
    existing      = load_existing(pred_file)
    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct", False))
    inf_times     = []
    k_stats       = []

    print(f"  Questions  : {total} | Done: {len(existing)} | Remaining: {total-len(existing)}")
    print()

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            k_stats.append(existing[qid].get("k_used", 3))
            continue

        drug_name  = record.get("query_drug_name", "")
        query      = record.get("query", "")
        supports   = record.get("supports", [])
        ans_name   = record.get("answer_name", "")
        bridge     = _get_bridge_from_cache(record, bridge_cache)

        # Candidate names (for signal filter)
        candidate_names = record.get("candidate_names", record["candidates"])

        # Weighted terms
        exp_result     = expand_query(drug_name)
        flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
        weighted_terms = get_weighted_terms(drug_name) or []
        if bridge:
            weighted_terms = _bridge_to_weighted_terms(bridge) + weighted_terms

        bridge_terms = [bridge] if bridge else []

        # ── Adaptive Signal Retrieval ──
        retrieved, k_used, filter_stats = retrieve_adaptive_signal(
            query=query,
            supports=supports,
            drug_a=drug_name,
            bridge_terms=bridge_terms,
            candidate_names=candidate_names,
            flat_terms=flat_terms,
            weighted_terms=weighted_terms,
            fetch_k=fetch_k,
            max_k=max_k,
            gap_ratio=gap_ratio,
        )
        k_stats.append(k_used)

        # ── Prompt + Inference ──
        prompt     = build_prompt_cot_enriched(record, retrieved, bridge)
        inf_result = do_inference(client, prompt, qid)
        inf_times.append(inf_result["inference_time"])

        if inf_result["success"]:
            raw = re.sub(r"<think.*?</think >", "", inf_result["raw_response"], flags=re.DOTALL).strip()
            if raw:
                inf_result["raw_response"] = raw

        prediction = extract_drug_id(inf_result["raw_response"], record["candidates"]) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == record["answer"].upper() and inf_result["success"]
        if is_correct:
            correct_count += 1

        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "answer":          record["answer"],
            "answer_name":     ans_name,
            "is_correct":      is_correct,
            "bridge_info":     bridge,
            "k_used":          k_used,
            "filter_stats":    filter_stats,
            "raw_response":    inf_result["raw_response"],
            "model":           active_model(),
            "provider":        ACTIVE_PROVIDER,
            "experiment":      "b4exp13_signal",
            "success":         inf_result["success"],
            "error":           inf_result["error"],
            "inference_time":  inf_result["inference_time"],
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 10 == 0 or done == total):
            answered_now = sum(1 for p in predictions if p.get("success", False))
            em_now = correct_count / answered_now * 100 if answered_now > 0 else 0.0
            elapsed = time.time() - pipeline_start
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} "
                  f"k={k_used} | EM: {em_now:.1f}% | {elapsed/60:.1f}m")

    # ── Final results ──
    em         = calculate_em(predictions)
    total_time = time.time() - pipeline_start
    avg_k      = sum(k_stats) / len(k_stats) if k_stats else 0

    print()
    print("="*60)
    print(f"  RESULT: Signal-Based Adaptive Filter / B4-EXP13")
    print(f"  EM Score  : {em['em_score']}%  ({em['correct']}/{em['total']})")
    print(f"  Avg K     : {avg_k:.2f}  (B4-EXP4 was fixed k=3)")
    print(f"  Baseline  : 33.33% (B4-EXP4) | {'↑ IMPROVED' if em['em_score'] > 33.33 else '↓'}")
    print(f"  Time      : {total_time/60:.1f} min")
    print("="*60 + "\n")

    logs = {
        "experiment":  "b4exp13_signal",
        "model":       active_model(),
        "fetch_k":     fetch_k,
        "max_k":       max_k,
        "gap_ratio":   gap_ratio,
        "results":     em,
        "avg_k_used":  round(avg_k, 2),
        "baseline_em": 33.33,
        "timing":      {"total_minutes": round(total_time/60, 2),
                        "avg_inference_s": round(sum(inf_times)/len(inf_times), 2) if inf_times else 0},
    }
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    return em


# ──────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Experiments — B4-EXP10+ (Adaptive K + Wikipedia V2 + Source-Aware)")
    parser.add_argument("--exp",
        choices=["atk", "wiki", "combined", "signal"],
        default="combined",
        help="B4-EXP10=atk | B4-EXP11=wiki | B4-EXP12=combined | B4-EXP13=signal")
    parser.add_argument("--sample", type=int, default=None,  help="Number of questions (None=all)")
    parser.add_argument("--methodology",
        choices=["all", "query2doc", "sequential", "medical_sections", "entity_pair"],
        default="all",
        help="Wikipedia V2 methodology: all|query2doc|sequential|medical_sections|entity_pair")
    parser.add_argument("--gap-high", dest="gap_high", type=float, default=0.3085,
                        help="High gap threshold -> k=2 (p75 from score_diagnostics)")
    parser.add_argument("--gap-mid",  dest="gap_mid",  type=float, default=0.1301,
                        help="Mid gap threshold -> k=3 (p50 from score_diagnostics)")
    parser.add_argument("--gap-low",  dest="gap_low",  type=float, default=0.0577,
                        help="Low gap threshold -> k=5 (p25 from score_diagnostics)")
    parser.add_argument("--k",      type=int,   default=3,   help="Top-K for Wikipedia experiment (fixed)")
    args = parser.parse_args()

    # ── Print header ──
    exp_labels = {
        "atk": "B4-EXP10: Adaptive Top-K (Internal Only)",
        "wiki": "B4-EXP11: Wikipedia V2 External Retrieval",
        "combined": "B4-EXP12: Combined + Source-Aware Scoring",
        "signal": "B4-EXP13: Signal-Based Adaptive Filter",
    }

    print()
    print("╔" + "═"*58 + "╗")
    print("║   Biomedical Multi-Hop QA — Adaptive Experiments" + " "*9 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║   Experiment  : {args.exp:<41}║")
    print(f"║   Label       : {exp_labels.get(args.exp, 'Unknown')[:41]:<41}║")
    print(f"║   Model       : {active_model()[:41]:<41}║")
    print(f"║   Sample      : {'All' if not args.sample else str(args.sample):<41}║")
    print("╚" + "═"*58 + "╝")

    if args.exp == "atk":
        run_adaptive_topk(
            sample_size=args.sample,
            gap_high=args.gap_high,
            gap_mid=args.gap_mid,
            gap_low=args.gap_low,
        )
    elif args.exp == "wiki":
        run_wikipedia_only(
            sample_size=args.sample,
            top_k=args.k,
            methodology=args.methodology,
        )
    elif args.exp == "combined":
        run_combined(
            sample_size=args.sample,
            gap_high=args.gap_high,
            gap_mid=args.gap_mid,
            gap_low=args.gap_low,
            methodology=args.methodology,
        )
    elif args.exp == "signal":
        run_adaptive_signal(
            sample_size=args.sample,
        )
