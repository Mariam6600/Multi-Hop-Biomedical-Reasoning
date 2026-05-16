"""
src/inference_pipeline_scoring.py
===================================
BioMed Multi-Hop QA — Score Generation (V3)

الفكرة:
  بدل طلب ترتيب (ranking) → نطلب من الـ LLM score لكل مرشح من الـ 9.
  كل path تستخدم retriever مختلف + cross-encoder re-ranking.

Pipeline لكل سؤال (لكل path):
  1. Retrieve Top-10   ← retriever مختلف لكل path
  2. Cross-Encoder     ← يعيد ترتيب الـ 10 ويأخذ أفضل 3
  3. LLM Scoring       ← استدعاء واحد: يعطي score لكل مرشح من الـ 9
  Output: {"DB001": 0.85, "DB002": 0.12, ...}

3 Paths:
  A → BM25-only        (keyword matching)
  B → MedCPT-only      (semantic / dense)
  C → Hybrid           (BM25 + MedCPT + Bridge boosting)

التشغيل:
  py -3.10 src/inference_pipeline_scoring.py --path A
  py -3.10 src/inference_pipeline_scoring.py --path B
  py -3.10 src/inference_pipeline_scoring.py --path C
  # اختبار سريع:
  py -3.10 src/inference_pipeline_scoring.py --path A --sample 20

ملاحظة:
  بعد تشغيل الـ 3 paths → شغّل train_meta_classifier.py
"""

import json
import os
import sys
import re
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE,
    MEDHOP_TRAIN,
    MEDHOP_DEV,
    OUTPUTS_DIR,
    OLLAMA_MODEL_NAME,
)
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.reranker import rerank_documents
from src.query_expander import expand_query
from src.query_expander_structured import get_weighted_terms

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RETRIEVE_TOP_N  = 10    # عدد الوثائق قبل الـ cross-encoder
FINAL_TOP_K     = 3     # عدد الوثائق بعد الـ cross-encoder
BRIDGE_CACHE    = os.path.join(OUTPUTS_DIR, "bridge_cache.json")

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# Scoring Prompt (for Path A and B)
PROMPT_SCORE = """\
You are a biomedical expert specializing in drug-drug interactions.

Supporting Evidence (top 3 most relevant documents):
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

TASK: Distribute exactly 100 points across the {n_cands} candidate drugs below.
The points reflect how likely each drug is to interact with {drug_name} via its mechanism.

Rules:
  - Total points MUST equal 100
  - Every candidate MUST receive at least 1 point (no zeros allowed)
  - The most likely candidate should receive the most points
  - Use the supporting evidence AND your biomedical knowledge

The interacting drug is one AFFECTED BY {drug_name}'s mechanism:
  - Metabolized or inhibited by the same enzyme
  - Acts on the same receptor or pathway
  - Shares the same biological target

Candidates:
{candidates_numbered}

Return ONLY a valid JSON object mapping each DrugBank ID to its integer points.
No explanation, no text, just the JSON.

Example format: {{"DB00001": 45, "DB00002": 20, "DB00003": 10, ...}} (must sum to 100)

Your allocation:"""

# Phase 2 Guided Ranking Prompt (for Path C - better performance: 33.33% vs 12.69%)
PROMPT_RANKING = """\
You are a biomedical expert specializing in drug-drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

TASK: Rank ALL {n_cands} candidate drugs from MOST to LEAST likely to interact with {drug_name}.

The interacting drug is one that is AFFECTED BY {drug_name}'s mechanism:
  - Metabolized by the same enzyme
  - Acts on the same receptor or biological target
  - Shares the same biological pathway

Candidates:
{candidates_numbered}

Instructions:
  - Use the supporting evidence AND your biomedical knowledge
  - Place the most likely interacting drug FIRST
  - Include ALL {n_cands} candidates in your ranking (no omissions)
  - Return ONLY a JSON array of DrugBank IDs in ranked order

Your ranking (JSON array, {n_cands} IDs, most-likely first):"""

# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_bridge_cache() -> dict:
    if not os.path.exists(BRIDGE_CACHE):
        return {}
    with open(BRIDGE_CACHE, encoding="utf-8") as f:
        return json.load(f)

def get_bridge(drug_name: str, cache: dict) -> str:
    raw = (
        cache.get(f"mechanism::{drug_name}")
        or cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    if not raw:
        return "unknown mechanism"
    raw = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"<\|[^|]+\|>", "", raw).strip()
    return raw.split("\n")[0].strip()[:120] or "unknown mechanism"

# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVERS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_bm25(record: dict, top_n: int) -> list:
    from src.retriever import retrieve_top_k
    return retrieve_top_k(
        query=record.get("query", ""),
        supports=record.get("supports", []),
        drug_name=record.get("query_drug_name", ""),
        top_k=top_n,
    )

def retrieve_dense(record: dict, top_n: int) -> list:
    from src.retriever_semantic import retrieve_semantic
    return retrieve_semantic(
        query=record.get("query", ""),
        supports=record.get("supports", []),
        drug_name=record.get("query_drug_name", ""),
        top_k=top_n,
    )

def retrieve_hybrid(record: dict, bridge_info: str, top_n: int) -> list:
    from src.retriever_hybrid_scored import retrieve_hybrid_scored
    drug_name      = record.get("query_drug_name", "")
    exp            = expand_query(drug_name)
    flat_terms     = exp.get("terms", []) if exp.get("success") else []
    weighted_terms = get_weighted_terms(drug_name)

    if bridge_info and bridge_info not in ("", "unknown mechanism"):
        bridge_clean = re.sub(
            r"\b(inhibits|blocks|acts|via|the|a|an|of|by|on)\b",
            "", bridge_info, flags=re.IGNORECASE
        ).strip()
        for w in bridge_clean.split():
            w = w.strip(".,;:")
            if len(w) > 2:
                weighted_terms.append({"term": w, "weight": 5.0, "type": "bridge_entity"})
                flat_terms.append(w)

    return retrieve_hybrid_scored(
        query=record.get("query", ""),
        supports=record.get("supports", []),
        drug_name=drug_name,
        flat_terms=flat_terms,
        weighted_terms=weighted_terms,
        top_k=top_n,
    )

def do_retrieval(record: dict, path: str, bridge_info: str) -> list:
    """Dispatcher لاختيار الـ retriever حسب الـ path."""
    if path == "A":
        return retrieve_bm25(record, RETRIEVE_TOP_N)
    elif path == "B":
        return retrieve_dense(record, RETRIEVE_TOP_N)
    else:  # C
        return retrieve_hybrid(record, bridge_info, RETRIEVE_TOP_N)

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def format_docs(docs: list, max_chars: int = 350) -> str:
    if not docs:
        return "No supporting evidence available."
    return "\n".join(
        f"[{d['rank']}] {d['text'][:max_chars]}{'...' if len(d['text']) > max_chars else ''}"
        for d in docs
    )

def build_scoring_prompt(record: dict, docs: list, bridge_info: str) -> str:
    """Build scoring prompt for Path A and B."""
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    candidates = record.get("candidates", [])
    cand_names = record.get("candidate_names") or candidates
    numbered   = "\n".join(
        f"  {i+1}. {n} ({c})" if n != c else f"  {i+1}. {c}"
        for i, (c, n) in enumerate(zip(candidates, cand_names))
    )
    return PROMPT_SCORE.format(
        docs=format_docs(docs),
        drug_name=drug_name,
        drug_id=drug_id,
        bridge_info=bridge_info or "unknown mechanism",
        candidates_numbered=numbered,
        n_cands=len(candidates),
    )

def build_ranking_prompt(record: dict, docs: list, bridge_info: str) -> str:
    """Build ranking prompt for Path C (Phase 2 Guided - 33.33% EM)."""
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    candidates = record.get("candidates", [])
    cand_names = record.get("candidate_names") or candidates
    numbered   = "\n".join(
        f"  {i+1}. {n} ({c})" if n != c else f"  {i+1}. {c}"
        for i, (c, n) in enumerate(zip(candidates, cand_names))
    )
    return PROMPT_RANKING.format(
        docs=format_docs(docs),
        drug_name=drug_name,
        drug_id=drug_id,
        bridge_info=bridge_info or "unknown mechanism",
        candidates_numbered=numbered,
        n_cands=len(candidates),
    )

# ══════════════════════════════════════════════════════════════════════════════
# SCORE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_scores(raw: str, candidates: list) -> dict:
    """
    استخراج الـ scores من استجابة الـ LLM.

    الـ prompt الجديد يطلب نقاطاً صحيحة تجمع 100.
    نستخرجها ثم نُنورمها إلى 0.0-1.0 بقسمتها على المجموع.

    الاستراتيجيات بالترتيب:
      1. JSON object بنقاط صحيحة → نورم
      2. Pattern matching (DB: رقم) → نورم
      3. Fallback: ترتيب ظهور الـ IDs → scores تنازلية
    """
    candidates_up = {c.upper() for c in candidates}
    raw_points    = {c.upper(): 0.0 for c in candidates}

    # ── Strategy 1: JSON parse ────────────────────────────────────────────────
    # نبحث عن آخر JSON object في الاستجابة (بعد أي تفكير)
    json_matches = list(re.finditer(r'\{[^{}]+\}', raw, re.DOTALL))
    for jm in reversed(json_matches):  # ابدأ من الأخير
        try:
            parsed = json.loads(jm.group())
            found  = 0
            temp   = {}
            for key, val in parsed.items():
                key_up = key.strip().upper()
                if key_up in candidates_up:
                    try:
                        pts = max(0.0, float(val))
                        temp[key_up] = pts
                        found += 1
                    except (ValueError, TypeError):
                        pass
            if found >= len(candidates) // 2:
                raw_points.update(temp)
                # تأكد أن كل مرشح له نقطة واحدة على الأقل
                for c in candidates_up:
                    if raw_points[c] == 0.0:
                        raw_points[c] = 1.0
                return _normalize(raw_points)
        except (json.JSONDecodeError, ValueError):
            continue

    # ── Strategy 2: Pattern matching ─────────────────────────────────────────
    pattern = re.compile(r'\b(DB\d{5})\b[\s:=,]+([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
    found   = 0
    for m in pattern.finditer(raw):
        cid = m.group(1).upper()
        if cid in candidates_up:
            try:
                raw_points[cid] = max(1.0, float(m.group(2)))
                found += 1
            except ValueError:
                pass
    if found >= len(candidates) // 2:
        for c in candidates_up:
            if raw_points[c] == 0.0:
                raw_points[c] = 1.0
        return _normalize(raw_points)

    # ── Strategy 3: Fallback — ترتيب الظهور → scores تنازلية ─────────────────
    ranked_ids, seen = [], set()
    for m in re.finditer(r'\bDB\d{5}\b', raw, re.IGNORECASE):
        cid = m.group().upper()
        if cid in candidates_up and cid not in seen:
            ranked_ids.append(cid)
            seen.add(cid)

    n = len(candidates)
    if ranked_ids:
        for rank, cid in enumerate(ranked_ids):
            raw_points[cid] = max(1.0, n - rank)
    # أي مرشح لم يُذكر → نقطة واحدة
    for c in candidates_up:
        if raw_points[c] == 0.0:
            raw_points[c] = 1.0

    return _normalize(raw_points)


def _normalize(points: dict) -> dict:
    """نورمة النقاط إلى مجموع 1.0."""
    total = sum(points.values())
    if total <= 0:
        n = len(points)
        return {k: 1.0 / n for k in points}
    return {k: round(v / total, 4) for k, v in points.items()}

# ══════════════════════════════════════════════════════════════════════════════
# RANKING PARSER (for Path C - Phase 2 Guided)
# ══════════════════════════════════════════════════════════════════════════════

def parse_ranking(raw_response: str, candidates: list) -> list:
    """
    استخراج الترتيب المُقترح من استجابة الـ LLM (Phase 2 Guided).

    الاستراتيجية:
      1. ابحث عن كل DrugBank ID بالترتيب في الاستجابة
      2. أي مرشح لم يُذكر → يُضاف في النهاية بترتيبه الأصلي
      3. الإخراج دائماً قائمة بطول len(candidates)
    """
    candidates_set = {c.upper() for c in candidates}
    found    = []
    seen     = set()

    # ابحث عن كل DB + 5 أرقام بترتيب الظهور
    for m in re.finditer(r'\bDB\d{5}\b', raw_response, re.IGNORECASE):
        cid = m.group().upper()
        if cid in candidates_set and cid not in seen:
            found.append(cid)
            seen.add(cid)

    # أضف المرشحين المفقودين في نهاية القائمة (بترتيبهم الأصلي)
    for c in candidates:
        cu = c.upper()
        if cu not in seen:
            found.append(cu)
            seen.add(cu)

    return found  # دائماً len = len(candidates)

def ranking_to_scores(ranked: list) -> dict:
    """
    تحويل الترتيب إلى scores منورمة (للتوافق مع باقي الكود).
    
    المرشح الأول يحصل على أعلى score، والأخير على أقل score.
    Scores منورمة بحيث مجموعها = 1.0
    """
    n = len(ranked)
    if n == 0:
        return {}
    
    # استخدام rank-weighted scores: (n - rank) / sum(1..n)
    # المرشح الأول: (n-0) = n، الثاني: (n-1)، ..., الأخير: 1
    raw_scores = {cid: (n - i) for i, cid in enumerate(ranked)}
    
    # نورمة
    total = sum(raw_scores.values())
    return {k: round(v / total, 4) for k, v in raw_scores.items()}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_file: str, n: int = None) -> list:
    """Load data from specified dataset file (train or dev)."""
    if not os.path.exists(dataset_file):
        print(f"  [FAIL] Dataset not found: {dataset_file}")
        sys.exit(1)
    with open(dataset_file, encoding="utf-8") as f:
        data = json.load(f)
    return data[:n] if n else data

def load_existing(path_file: str) -> dict:
    if not os.path.exists(path_file):
        return {}
    try:
        with open(path_file, encoding="utf-8") as f:
            return {r["question_id"]: r for r in json.load(f)}
    except Exception:
        return {}

def save_json(obj, filepath: str):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

PATH_LABELS = {
    "A": "BM25-only (keyword matching)",
    "B": "MedCPT Dense (semantic)",
    "C": "Hybrid Bridge-Boosted",
}

# Model assignment per path (for diversity)
PATH_MODELS = {
    "A": "biomistral-7b",   # Path A (BM25): BioMistral (7B - medical specialist)
    "B": "qwen2.5-7b",      # Path B (MedCPT): Qwen 2.5 (7B)
    "C": "qwen3.5-9b",      # Path C (Hybrid): Qwen 3.5 (9B - most advanced)
}

def run_pipeline(path: str = "A", sample_size: int = None, verbose: bool = True):
    """
    Run scoring pipeline for a specific path.
    
    Args:
        path: Retrieval path (A, B, or C)
        sample_size: Number of questions to process (None = all)
        verbose: Print progress
    """
    assert path in ("A", "B", "C"), "path يجب أن يكون: A | B | C"

    # Use model from settings (will be changed by batch file)
    model_name = OLLAMA_MODEL_NAME
    
    exp_tag   = f"scoring_v2_path{path}_k{FINAL_TOP_K}_{model_name}"
    pred_file = os.path.join(OUTPUTS_DIR, f"{exp_tag}_predictions.json")
    logs_file = os.path.join(OUTPUTS_DIR, f"{exp_tag}_logs.json")

    print("\n" + "═" * 68)
    if path == "C":
        print(f"  Path {path} — Phase 2 Guided Ranking: {PATH_LABELS[path]}")
        print(f"  Strategy   : Ranking (33.33% EM on dev.json vs 12.69% Scoring)")
    else:
        print(f"  Scoring Pipeline — Path {path}: {PATH_LABELS[path]}")
    print(f"  Dataset    : train.json (MEDHOP_FILE)")
    print(f"  Retrieval  : Top-{RETRIEVE_TOP_N} → Cross-Encoder → Top-{FINAL_TOP_K}")
    if path == "C":
        print(f"  LLM Output : ranked list of 9 candidates (most→least likely)")
    else:
        print(f"  LLM Output : scores for all 9 candidates (0.0–1.0)")
    print(f"  Model      : {model_name}")
    print(f"  Output     : {pred_file}")
    print("═" * 68 + "\n")

    # ── Connect ──────────────────────────────────────────────────────────────
    print("--- Step 1: Connecting ---")
    client = get_ollama_client()
    if not check_model_available(client, model_name):
        sys.exit(1)
    print(f"  [OK]   Model ready\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("--- Step 2: Loading Data ---")
    data         = load_data(MEDHOP_FILE, sample_size)
    bridge_cache = load_bridge_cache()
    print(f"  [OK]   {len(data)} questions | {len(bridge_cache)} bridge entries\n")

    # ── Resume ───────────────────────────────────────────────────────────────
    print("--- Step 3: Checking Progress ---")
    existing    = load_existing(pred_file)
    predictions = list(existing.values())
    print(f"  Done: {len(existing)} | Remaining: {len(data) - len(existing)}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    if path == "C":
        print("--- Step 4: Phase 2 Guided Ranking ---\n")
    else:
        print("--- Step 4: Score Generation ---\n")
    t_start    = time.time()
    ret_times  = []
    inf_times  = []

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing:
            continue

        drug_name  = record.get("query_drug_name", "")
        candidates = record.get("candidates", [])
        cand_names = record.get("candidate_names") or candidates
        answer     = record.get("answer", "")

        bridge_info = get_bridge(drug_name, bridge_cache)

        # ── Retrieval (Top-10) ────────────────────────────────────────────────
        t_ret    = time.time()
        raw_docs = do_retrieval(record, path, bridge_info)
        ret_time = round(time.time() - t_ret, 3)
        ret_times.append(ret_time)

        # ── Cross-Encoder Re-ranking (Top-10 → Top-3) ────────────────────────
        reranked = rerank_documents(
            query=record.get("query", ""),
            bridge_info=bridge_info,
            retrieved_docs=raw_docs,
            final_k=FINAL_TOP_K,
        )

        # ── Build Prompt (Scoring for A/B, Ranking for C) ────────────────────
        if path == "C":
            # Path C: Phase 2 Guided Ranking (33.33% EM on dev.json)
            prompt = build_ranking_prompt(record, reranked, bridge_info)
        else:
            # Path A/B: Scoring Strategy
            prompt = build_scoring_prompt(record, reranked, bridge_info)

        # ── ONE LLM Call ──────────────────────────────────────────────────────
        t_inf    = time.time()
        inf_res  = run_inference(client, prompt, qid, model_name=model_name)
        inf_time = round(time.time() - t_inf, 3)
        inf_times.append(inf_time)

        # ── Parse Response (Scores for A/B, Ranking for C) ───────────────────
        raw = inf_res.get("raw_response", "")
        if path == "C":
            # Path C: Parse ranking and convert to scores
            if inf_res["success"]:
                ranked_list = parse_ranking(raw, candidates)
                scores_dict = ranking_to_scores(ranked_list)
            else:
                # Fallback: equal scores
                scores_dict = {c.upper(): 1.0/len(candidates) for c in candidates}
                ranked_list = list(candidates)
        else:
            # Path A/B: Parse scores
            if inf_res["success"]:
                scores_dict = parse_scores(raw, candidates)
            else:
                # Fallback: equal scores
                scores_dict = {c.upper(): 1.0/len(candidates) for c in candidates}
            # Derive ranking from scores
            ranked_list = sorted(scores_dict, key=scores_dict.get, reverse=True)

        # الإجابة المتوقعة = المرشح بأعلى score (أو الأول في الترتيب)
        best_cand  = ranked_list[0] if ranked_list else max(scores_dict, key=scores_dict.get)
        is_correct = (best_cand.upper() == answer.upper() and inf_res["success"])

        # Recall@K بناءً على الترتيب
        recall_results  = {
            f"recall@{k}": answer.upper() in [r.upper() for r in ranked_list[:k]]
            for k in range(1, len(candidates) + 1)
        }

        pred_record = {
            "question_id":     qid,
            "query":           record.get("query", ""),
            "query_drug_id":   record.get("query_drug_id", ""),
            "query_drug_name": drug_name,
            "candidates":      candidates,
            "candidate_names": cand_names,
            "answer":          answer,
            "answer_name":     record.get("answer_name", ""),
            # الـ scores: المخرج الجديد والأهم
            "scores":          {c.upper(): scores_dict.get(c.upper(), 0.0) for c in candidates},
            # ranked list مشتقة من الـ scores أو الترتيب المباشر (Path C)
            "ranked_candidates": ranked_list,
            "prediction":      best_cand,
            "is_correct":      is_correct,
            **recall_results,
            "bridge_info":     bridge_info,
            "docs_retrieved":  len(raw_docs),
            "docs_reranked":   len(reranked),
            "raw_response":    raw,
            "success":         inf_res["success"],
            "error":           inf_res.get("error", ""),
            "model":           model_name,
            "path":            path,
            "retriever":       PATH_LABELS[path],
            "retrieval_time":  ret_time,
            "inference_time":  inf_time,
        }

        predictions.append(pred_record)
        existing[qid] = pred_record
        save_json(list(existing.values()), pred_file)

        # ── Progress ──────────────────────────────────────────────────────────
        if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(data)):
            done = [p for p in predictions if p.get("success")]
            em   = sum(1 for p in done if p["is_correct"]) / len(done) * 100 if done else 0
            r3   = sum(1 for p in predictions if p.get("recall@3")) / len(predictions) * 100
            status = "✅" if is_correct else "❌"
            print(
                f"  [{i+1:>3}/{len(data)}] {status} {qid:<12} | "
                f"best:{best_cand:<10} ans:{answer:<10} | "
                f"EM:{em:.1f}% R@3:{r3:.1f}% | "
                f"ret:{ret_time:.1f}s inf:{inf_time:.1f}s"
            )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n--- Step 5: Summary ---\n")

    from src.inference_pipeline_ranking import compute_recall_curve
    recall_curve = compute_recall_curve(predictions)
    total_time   = time.time() - t_start

    logs = {
        "run_info": {
            "experiment":      f"Scoring Path {path} — {PATH_LABELS[path]}",
            "path":            path,
            "retriever":       PATH_LABELS[path],
            "model":           model_name,
            "retrieve_top_n":  RETRIEVE_TOP_N,
            "final_top_k":     FINAL_TOP_K,
            "total_questions": len(predictions),
            "total_time_min":  round(total_time / 60, 2),
            "avg_retrieval_s": round(sum(ret_times)/len(ret_times), 3) if ret_times else 0,
            "avg_inference_s": round(sum(inf_times)/len(inf_times), 3) if inf_times else 0,
        },
        "recall_curve": recall_curve,
    }
    save_json(logs, logs_file)

    print("═" * 68)
    print(f"  PATH {path} — {PATH_LABELS[path]}")
    print("═" * 68)
    for k in range(1, 10):
        val = recall_curve.get(f"recall@{k}", 0)
        bar = "█" * int(val / 3.5)
        em_tag = "  ← EM" if k == 1 else ""
        print(f"  K={k:<4}  {val:>7.2f}%  {bar}{em_tag}")
    print(f"\n  Total time : {total_time/60:.1f} min")
    print(f"  Saved to   : {pred_file}")
    print("═" * 68 + "\n")

    return recall_curve, pred_file


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score Generation — 3 Diverse Paths + Cross-Encoder"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help=(
            "الـ retrieval path:\n"
            "  A → BM25-only\n"
            "  B → MedCPT Dense\n"
            "  C → Hybrid Bridge-Boosted"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="عدد الأسئلة (للاختبار السريع، افتراضي = كل البيانات)",
    )
    args = parser.parse_args()
    run_pipeline(path=args.path, sample_size=args.sample)
