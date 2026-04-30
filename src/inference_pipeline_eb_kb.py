"""
src/inference_pipeline_eb_kb.py
=================================
B4-EXP4 + KB Evidence  ←  تجربة عزل أثر قاعدة المعرفة

الفرق الوحيد عن B4-EXP4:
  ✦ يُضيف قسم [KB Evidence] للـ Prompt قبل قائمة المرشحين
  ✦ يُرتب المرشحين حسب درجة KB (الأقوى أولاً)

كل شيء آخر مطابق 100% لـ B4-EXP4:
  • نفس النموذج المحلي (qwen3.5-9b)
  • نفس الـ Retriever   (hybrid_scored)
  • نفس K=3
  • نفس Temperature   (من settings.py)
  • نفس bridge_info logic

الهدف: قياس أثر KB وحده — إذا ارتفعت النتيجة عن 33.33% نكمل.

التشغيل:
    py -3.10 src/inference_pipeline_eb_kb.py

المخرج:
    outputs/eb_kb_guided_retrieval_k3_qwen3_5-9b_predictions.json
"""

import json, os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE, OUTPUTS_DIR, DIAG_SAMPLE_SIZE,
    ACTIVE_PROVIDER, OLLAMA_MODEL_NAME,
)

# ─── runner (local Ollama فقط) ───────────────────────────────────────────────
if ACTIVE_PROVIDER != "ollama":
    print(f"  [FAIL] هذا الملف مخصص للتشغيل المحلي (ollama).")
    print(f"  ACTIVE_PROVIDER = '{ACTIVE_PROVIDER}' في settings.py")
    print(f"  غيّريه إلى 'ollama' لتشغيل التجربة المحلية.")
    sys.exit(1)

from src.llm_runner import (
    get_ollama_client     as get_client,
    check_model_available as check_model,
    run_inference         as do_inference,
)
from src.query_expander            import expand_query
from src.query_expander_structured import get_weighted_terms
from src.retriever_hybrid_scored   import retrieve_hybrid_scored
from src.prompt_builder            import extract_drug_id
from src.kb_builder                import load_kb, get_kb_evidence_text

# ─── إعدادات التجربة ──────────────────────────────────────────────────────────
EXP_NAME   = "eb_kb_guided_retrieval"
TOP_K      = 3        # مطابق B4-EXP4
KB_TOP_N   = 3        # عدد المرشحين المُميَّزين من KB

BRIDGE_CACHE = os.path.join(OUTPUTS_DIR, "bridge_cache.json")
OUTPUT_PREDS = os.path.join(OUTPUTS_DIR, f"{EXP_NAME}_k{TOP_K}_{OLLAMA_MODEL_NAME}_predictions.json")
OUTPUT_LOGS  = os.path.join(OUTPUTS_DIR, f"{EXP_NAME}_k{TOP_K}_{OLLAMA_MODEL_NAME}_logs.json")

# ─────────────────────────────────────────────────────────────────────────────
# Prompt — نفس B4-EXP4 (PROMPT_COT_ENRICHED) + قسم KB مُضاف
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_B4_EXP4_WITH_KB = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

{kb_section}
Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- {drug_name} has mechanism: {bridge_info}
- Look for a candidate that is AFFECTED BY this mechanism (metabolized by same enzyme, targets same receptor, shares same pathway)
- The interacting drug does NOT need to have the same mechanism — it just needs to be affected by it
- If the KB Evidence above marks a candidate with [KB: high DDI risk], prioritize it
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# مساعدات
# ─────────────────────────────────────────────────────────────────────────────

def _load_bridge_cache() -> dict:
    if os.path.exists(BRIDGE_CACHE):
        with open(BRIDGE_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_bridge(drug_name: str, cache: dict) -> str:
    raw = cache.get(f"mechanism::{drug_name}", "")
    if not raw: return ""
    raw = re.sub(r"MECHANISM:\s*", "", raw, flags=re.IGNORECASE).strip()
    return raw.split("\n")[0].strip()[:100]


def _do_retrieval(record: dict, bridge_info: str) -> list:
    """نفس retrieval منطق B4-EXP4 بالضبط."""
    drug_name      = record.get("query_drug_name", "")
    exp            = expand_query(drug_name)
    flat_terms     = exp.get("terms", []) if exp.get("success") else []
    weighted_terms = get_weighted_terms(drug_name)
    # bridge boost — مطابق B4-EXP4
    if bridge_info and bridge_info not in ("", "unknown"):
        clean = (bridge_info.replace("inhibits","").replace("blocks","")
                            .replace("acts","").replace("via",""))
        for w in clean.split():
            if len(w.strip()) > 2:
                weighted_terms.append({"term": w.strip(), "weight": 5.0, "type": "bridge_entity"})
                flat_terms.append(w.strip())
    return retrieve_hybrid_scored(
        query=record.get("query", ""), supports=record.get("supports", []),
        drug_name=drug_name, flat_terms=flat_terms,
        weighted_terms=weighted_terms, top_k=TOP_K,
    )


def _format_docs(retrieved: list) -> str:
    if not retrieved: return "No supporting evidence available."
    parts = []
    for r in retrieved:
        text = r.get("text", "") if isinstance(r, dict) else str(r)
        if len(text) > 400: text = text[:400] + "..."
        rank = r.get("rank", "") if isinstance(r, dict) else ""
        parts.append(f"[{rank}] {text}")
    return "\n".join(parts)


def _build_prompt(record: dict, retrieved: list, bridge_info: str, kb: dict) -> str:
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    candidates = record.get("candidates", [])
    cand_names = record.get("candidate_names", candidates)

    # ─── KB evidence ───────────────────────────────────────────────────────
    kb_top, kb_text = get_kb_evidence_text(drug_id, candidates, kb, top_n=KB_TOP_N)
    kb_top_set = set(kb_top)

    # إذا لم يكن هناك أي evidence هيكلي، لا نُضيف القسم (نظافة الـ prompt)
    has_kb_info = any(
        any(score > 0 for score in [])
        for _ in [1]
    )
    # فحص بسيط: هل في أي overlap بين الدواء والمرشحين في KB؟
    from src.kb_builder import score_candidate
    any_kb_score = any(score_candidate(drug_id, c, kb)["score"] > 0 for c in candidates)

    if any_kb_score:
        kb_section = kb_text + "\n\n"
    else:
        kb_section = ""   # لا نُزعج النموذج بـ section فارغة

    # ─── ترتيب المرشحين: KB الأقوى أولاً ──────────────────────────────────
    cand_pairs = list(zip(candidates, cand_names))
    kb_first   = [(c, n) for c, n in cand_pairs if c in kb_top_set]
    rest       = [(c, n) for c, n in cand_pairs if c not in kb_top_set]
    ordered    = kb_first + rest

    cand_lines = []
    for c, n in ordered:
        tag = " [KB: high DDI risk]" if c in kb_top_set and any_kb_score else ""
        if n and n != c:
            cand_lines.append(f"- {n} ({c}){tag}")
        else:
            cand_lines.append(f"- {c}{tag}")
    candidates_text = "\n".join(cand_lines)

    return PROMPT_B4_EXP4_WITH_KB.format(
        docs            = _format_docs(retrieved),
        kb_section      = kb_section,
        drug_name       = drug_name,
        drug_id         = drug_id,
        bridge_info     = bridge_info or "unknown",
        candidates_text = candidates_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# الدالة الرئيسية
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(sample_size=None, verbose=True):
    print("\n" + "="*60)
    print("  B4-EXP4 + KB Evidence (Local — qwen3.5-9b)")
    print("  الفرق الوحيد: KB Evidence مُضاف للـ Prompt")
    print("="*60)

    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)

    # ─── تحميل KB ────────────────────────────────────────────────────────────
    print("\n  [KB] Loading knowledge base...")
    kb = load_kb()
    if not kb:
        print("  [WARN] KB فارغ — تأكدي من تشغيل kb_builder.py أولاً")
        print("         py -3.10 src/kb_builder.py")
        sys.exit(1)
    print(f"  [KB] {len(kb):,} drugs indexed ✓")

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if sample_size:
        data = data[:sample_size]

    bridge_cache = _load_bridge_cache()

    # ─── Resume ──────────────────────────────────────────────────────────────
    existing = {}
    if os.path.exists(OUTPUT_PREDS):
        try:
            with open(OUTPUT_PREDS, encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("success"):
                        existing[r["question_id"]] = r
            if existing:
                print(f"  [INFO] Resume: {len(existing)}/{len(data)} منجز")
        except Exception:
            pass

    print(f"\n  Model     : {OLLAMA_MODEL_NAME}")
    print(f"  Retriever : hybrid_scored (نفس B4-EXP4)")
    print(f"  K         : {TOP_K} (نفس B4-EXP4)")
    print(f"  KB Top-N  : {KB_TOP_N}")
    print(f"  Total     : {len(data)} | Done: {len(existing)}")
    print(f"  Output    : {OUTPUT_PREDS}\n")

    # ─── اتصال ───────────────────────────────────────────────────────────────
    client = get_client()
    if not check_model(client):
        sys.exit(1)

    predictions   = list(existing.values())
    correct_count = sum(r.get("is_correct", False) for r in predictions)
    start_time    = time.time()

    # ─── الحلقة الرئيسية ──────────────────────────────────────────────────
    for i, record in enumerate(data):
        qid = record.get("id", record.get("question_id", f"q{i}"))
        if qid in existing:
            continue

        drug_name   = record.get("query_drug_name", "")
        answer      = record.get("answer", "")
        bridge_info = _get_bridge(drug_name, bridge_cache)
        retrieved   = _do_retrieval(record, bridge_info)
        prompt      = _build_prompt(record, retrieved, bridge_info, kb)

        inf_result  = do_inference(client, prompt, qid)

        # ─── تنظيف thinking blocks ────────────────────────────────────────
        raw = inf_result.get("raw_response", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", raw, flags=re.DOTALL).strip()

        prediction = extract_drug_id(raw, record.get("candidates", [])) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == answer.upper() and inf_result["success"]
        if is_correct: correct_count += 1

        pred_record = {
            "question_id":     qid,
            "query_drug_name": drug_name,
            "prediction":      prediction,
            "raw_response":    raw,
            "answer":          answer,
            "answer_name":     record.get("answer_name", ""),
            "is_correct":      is_correct,
            "bridge_info":     bridge_info,
            "model":           OLLAMA_MODEL_NAME,
            "provider":        "ollama",
            "retriever":       "hybrid_scored+KB",
            "top_k":           TOP_K,
            "strategy":        "guided_retrieval+KB",
            "success":         inf_result["success"],
            "error":           inf_result.get("error", ""),
            "inference_time":  inf_result.get("inference_time", 0),
        }
        predictions.append(pred_record)
        existing[qid] = pred_record

        # حفظ فوري
        with open(OUTPUT_PREDS, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        # Progress
        done = i + 1
        if verbose and (done % 5 == 0 or done == len(data)):
            answered = sum(1 for p in predictions if p.get("success"))
            em       = correct_count / answered * 100 if answered else 0
            elapsed  = (time.time() - start_time) / 60
            mark     = "✓" if is_correct else "✗"
            print(f"  [{done:>3}/{len(data)}] {mark} {qid:<14} | EM: {em:.1f}% | {elapsed:.1f}m")

    # ─── النتائج النهائية ──────────────────────────────────────────────────
    total    = len(predictions)
    failed   = sum(1 for p in predictions if not p.get("success", True))
    correct  = sum(1 for p in predictions if p.get("is_correct"))
    answered = total - failed
    em_s     = correct / total    * 100 if total    else 0
    em_l     = correct / answered * 100 if answered else 0

    print(f"\n{'='*60}")
    print(f"  RESULT: B4-EXP4 + KB Evidence (Local)")
    print(f"  EM (Strict):   {em_s:.2f}%  ({correct}/{total})")
    print(f"  EM (Lenient):  {em_l:.2f}%  ({correct}/{answered})")
    print(f"  Failed:        {failed}")
    print(f"  vs B4-EXP4 (33.33%): {'↑ +' if em_s>33.33 else '↓ '}{abs(em_s-33.33):.2f}%")
    print(f"{'='*60}\n")

    # حفظ logs
    with open(OUTPUT_LOGS, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":    EXP_NAME,
            "model":         OLLAMA_MODEL_NAME,
            "strategy":      "B4-EXP4 + KB Evidence only",
            "changes_from_b4exp4": ["Added KB Evidence section to prompt",
                                    "Reordered candidates by KB score"],
            "unchanged":     ["model", "retriever=hybrid_scored",
                              "K=3", "temperature", "bridge_info", "prompt_base"],
            "results":       {"total":total,"correct":correct,
                              "failed":failed,"em_score":round(em_s,2)},
            "baseline_b4exp4": 33.33,
            "improvement":   round(em_s - 33.33, 2),
        }, f, ensure_ascii=False, indent=2)

    return em_s


if __name__ == "__main__":
    run_experiment(sample_size=DIAG_SAMPLE_SIZE)
