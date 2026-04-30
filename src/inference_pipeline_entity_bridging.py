"""
src/inference_pipeline_entity_bridging.py
==========================================
Biomedical Multi-Hop QA — Phase 2: Entity Bridging & Query Decomposition

الموقف الأكاديمي:
  كل التجارب في هذا الملف تنتمي لمقاربة "Entity Bridging" وهي نوع من
  Query Decomposition المتخصص لـ MedHop. السؤال لا يُقسَّم نصياً، بل
  مسار الإجابة يُقسَّم إلى خطوتين عبر الـ bridge entity (آلية الدواء).

  الـ Bridge Entity هو "الإجابة الوسطية" التي تربط الدواء A بالدواء B.
  مثال: Moclobemide → [inhibits MAO-A] → Tetrabenazine

الأساليب (كلها تستخدم نفس الـ Bridge، لكن تختلف في كيفية توظيفه):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  المجموعة A — Prompt-Based (الآلية تُضاف للـ Prompt)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  no_decomp         Control Baseline — بدون bridge (للمقارنة)
  enriched          B4-EXP2 — الآلية كـ hint نصي بسيط                 → 24.27%
  cot_enriched      B4-EXP3 — الآلية + CoT يوجّه الاستدلال              → 32.16%
  forced_reasoning  B4-EXP6 — Prompt كنموذج step-by-step إجباري        → 16.67%
  bridge_pivoted    B5-EXP4 — السؤال يتحول "أي مرشح يتأثر بـ [bridge]?" (جديد)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  المجموعة B — Retrieval-Based (الآلية تُضاف للـ Retrieval)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  guided_retrieval  B4-EXP4 — الآلية كـ مصطلح بحث بوزن 5.0            → 33.33% ✓ أفضل
  reranked          B4-EXP5 — Guided + Cross-Encoder Re-Ranking         → 24.85%
  dual_retrieval    B4-EXP7 — بحثان منفصلان: الدواء + الآلية + Merge   → 17.25%
  cand_filtered     B4-EXP9 — Guided K=10 → فلتر بناء على ذكر المرشحين → K=3

النتائج الكاملة (من error_analysis_all.txt):
  - 100% من الأخطاء = Wrong Candidate في كل التجارب
  - لا Hallucination، لا Format Error
  - 33.33% هو الحد الأعلى للنموذج المحلي Qwen3.5-9B
"""

import json, os, sys, time, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE, OUTPUTS_DIR, DIAG_SAMPLE_SIZE, OLLAMA_MODEL_NAME,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MAX_TOKENS, LLM_NUM_CTX,
)
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.query_expander import expand_query
from src.retriever_expanded import retrieve_expanded
from src.retriever_hybrid_scored import retrieve_hybrid_scored
from src.query_expander_structured import get_weighted_terms
from src.prompt_builder import extract_drug_id
from src.retriever_combined import retrieve_combined
from src.reranker import rerank_documents
from src.retriever_semantic import retrieve_semantic

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
BRIDGE_CACHE_FILE  = os.path.join(OUTPUTS_DIR, "bridge_cache.json")
BRIDGE_MAX_TOKENS  = 150
RERANK_FETCH_K     = 10   # عدد الوثائق قبل Re-Ranking
CAND_FILTER_FETCH_K = 10  # عدد الوثائق قبل فلترة المرشحين (B4-EXP9)

# ══════════════════════════════════════════════════════════════
# BRIDGE CACHE
# ══════════════════════════════════════════════════════════════

def _load_bridge_cache():
    if os.path.exists(BRIDGE_CACHE_FILE):
        try:
            with open(BRIDGE_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

def _save_bridge_cache(cache):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(BRIDGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def run_inference_bridge(client, prompt, qid):
    """استدعاء سريع للنموذج لاستخراج الـ bridge (آلية الدواء) — tokens محدودة."""
    result = {"question_id": qid, "raw_response": "", "inference_time": 0.0, "success": False, "error": ""}
    for attempt in range(2):
        try:
            start = time.time()
            resp = client.chat(model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P,
                         "top_k": LLM_TOP_K, "num_predict": BRIDGE_MAX_TOKENS, "num_ctx": LLM_NUM_CTX})
            raw = resp["message"]["content"].strip()
            if not raw: raise ValueError("Empty response")
            result.update({"raw_response": raw, "inference_time": round(time.time()-start, 3), "success": True, "error": ""})
            return result
        except Exception as e:
            result["error"] = str(e)
            if attempt == 0: time.sleep(2)
    return result

BRIDGE_PROMPT = """\
You are a biomedical expert. What is the PRIMARY mechanism of action of {drug_name}?

Evidence:
{docs}

Answer in ONE short phrase (e.g., "inhibits CYP2D6", "blocks MAO-A"):
MECHANISM:"""

# ══════════════════════════════════════════════════════════════
# PROMPTS — المجموعة A: Prompt-Based
# ══════════════════════════════════════════════════════════════

# Control (بدون bridge)
PROMPT_NO_DECOMP = """\
You are a biomedical expert specializing in drug interactions.

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

# B4-EXP2: الآلية كـ hint بسيط
# الناتج: 24.27% | السبب: النموذج يأخذ الـ hint لكن الوثائق لا تدعمه
PROMPT_ENRICHED = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the supporting evidence to identify which drug interacts with {drug_name}
- The mechanism above is a hint — the interacting drug may be affected by this mechanism
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""

# B4-EXP3: CoT يوجّه الاستدلال حول الـ bridge
# الناتج: 32.16% | السبب: توجيه أفضل للنموذج نحو العلاقة الصحيحة
PROMPT_COT_ENRICHED = """\
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

# B4-EXP6: Prompt كنموذج step-by-step إجباري
# الناتج: 16.67% | السبب: الـ Prompt الطويل شتّت النموذج (17.5% Format Error)
PROMPT_FORCED_REASONING = """\
You are a biomedical expert. Analyze this drug interaction problem step by step.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Mechanism of action: {bridge_info}

Candidates:
{candidates_text}

Task: Find which ONE candidate drug interacts with {drug_name}.

Step 1 - Mechanism analysis:
{drug_name} works by: {bridge_info}
Which candidates could be affected by this mechanism?

Step 2 - Evidence check:
Read the supporting evidence. Which candidate is MENTIONED or RELATED to {drug_name}'s mechanism?

Step 3 - Final answer:
Based on steps 1 and 2, the interacting drug is:

Answer (DrugBank ID only, format DBxxxxx):"""

# B5-EXP4: السؤال يتحول من multi-hop إلى single-hop حول الـ bridge
# مستوحى من UETQuintet (Rank 2, 83.8% BioCreative IX 2025)
# الـ bridge entity يصبح محور السؤال بدلاً من الدواء الأصلي
PROMPT_BRIDGE_PIVOTED = """\
You are a biomedical expert specializing in drug mechanisms and interactions.

Supporting Evidence:
{docs}

The drug {drug_name} ({drug_id}) works by: {bridge_info}

Key question: Which of the following candidate drugs is AFFECTED BY or INTERACTS WITH the mechanism "{bridge_info}"?

A drug is "affected by" a mechanism if it:
- Is metabolized by the same enzyme
- Acts on the same receptor or target
- Shares the same biological pathway

Candidates:
{candidates_text}

Answer with the DrugBank ID only (format: DBxxxxx):"""

# ══════════════════════════════════════════════════════════════
# PROMPTS — المجموعة B: Retrieval-Based
# ══════════════════════════════════════════════════════════════

# B4-EXP4 / B4-EXP5: CoT مع Retrieval موجَّه بالـ bridge
# نفس الـ Prompt لـ guided_retrieval و reranked — الفرق في الـ retrieval strategy
PROMPT_COT_WITH_GUIDED = PROMPT_COT_ENRICHED  # alias واضح

# B4-EXP7: بحثان منفصلان + دمج الوثائق
# FewShot مثال + سياق مدمج (وثائق الدواء + وثائق الآلية)
PROMPT_DUAL_RETRIEVAL = """\
You are a biomedical expert specializing in drug interactions.

Example:
Drug: Fluoxetine (DB00472) — inhibits the CYP2D6 enzyme
Supporting evidence: "Fluoxetine is a potent inhibitor of CYP2D6, affecting the metabolism of drugs like desipramine."
Interacting drug: Desipramine (DB01151) — because it is metabolized by CYP2D6

Now answer the following:

Supporting Evidence (drug documents + mechanism documents combined):
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- {drug_name} works by: {bridge_info}
- The evidence includes documents about both {drug_name} AND its mechanism
- Find the candidate AFFECTED BY or INTERACTING WITH this mechanism
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""

# ══════════════════════════════════════════════════════════════
# STRATEGY → PROMPT MAPPING
# ══════════════════════════════════════════════════════════════
STRATEGY_PROMPTS = {
    # المجموعة A: Prompt-Based
    "no_decomp":         PROMPT_NO_DECOMP,         # Control — بدون bridge
    "enriched":          PROMPT_ENRICHED,           # B4-EXP2 → 24.27%
    "cot_enriched":      PROMPT_COT_ENRICHED,       # B4-EXP3 → 32.16%
    "forced_reasoning":  PROMPT_FORCED_REASONING,   # B4-EXP6 → 16.67%
    "bridge_pivoted":    PROMPT_BRIDGE_PIVOTED,     # B5-EXP4 → (قيد التنفيذ)

    # المجموعة B: Retrieval-Based
    "guided_retrieval":  PROMPT_COT_WITH_GUIDED,    # B4-EXP4 → 33.33% ✓ أفضل
    "reranked":          PROMPT_COT_WITH_GUIDED,    # B4-EXP5 → 24.85%
    "dual_retrieval":    PROMPT_DUAL_RETRIEVAL,     # B4-EXP7 → 17.25%
    "cand_filtered":     PROMPT_COT_WITH_GUIDED,    # B4-EXP9 → (جديدة)
}

# الاستراتيجيات التي تحتاج bridge_info
NEEDS_BRIDGE = {
    "enriched", "cot_enriched", "forced_reasoning", "bridge_pivoted",
    "guided_retrieval", "reranked", "dual_retrieval", "cand_filtered"
}

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _format_docs(retrieved, max_chars=400):
    if not retrieved: return "No supporting evidence available."
    return "\n".join(f"[{r['rank']}] {r['text'][:max_chars]}{'...' if len(r['text'])>max_chars else ''}" for r in retrieved)

def _clean(text):
    if not text: return ""
    text = re.sub(r"<\|endoftext\|>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|im_start\|>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text.strip("* ").strip()

def _extract_bridge(raw):
    if not raw: return ""
    raw = _clean(raw)
    m = re.search(r"MECHANISM:\s*(.+)", raw, re.IGNORECASE)
    if m: return _clean(m.group(1).split("\n")[0])[:150]
    for line in raw.splitlines():
        line = line.strip()
        if line and len(line) > 3 and not line.startswith("You "): return _clean(line)[:150]
    return raw[:150]

def load_data(n=None):
    with open(MEDHOP_FILE, encoding="utf-8") as f: data = json.load(f)
    return data[:n] if n else data

def _model_short():
    return re.sub(r"[/:\\.]+", "_", OLLAMA_MODEL_NAME).strip("_")[-30:]

def save_preds(preds, path):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(preds, f, indent=2, ensure_ascii=False)

def calc_metrics(preds):
    total = len(preds); failed = sum(1 for p in preds if not p.get("success")); answered = total - failed
    correct = sum(1 for p in preds if p.get("success") and p.get("prediction","").strip().upper() == p.get("answer","").strip().upper() and p.get("prediction","").strip())
    nm = sum(1 for p in preds if p.get("success") and p.get("nm_hit"))
    return {"total": total, "answered": answered, "failed": failed, "correct": correct,
            "strict_em": round(correct/total*100, 2) if total else 0,
            "lenient_em": round(correct/answered*100, 2) if answered else 0,
            "nm_hits": nm, "nm_score": round(nm/answered*100, 2) if answered else 0}

def merge_and_deduplicate(original_docs, bridge_docs, final_top_k=5):
    """دمج وثائق البحث الأصلي + وثائق الـ bridge مع إزالة المكررات."""
    seen, merged = set(), []
    for doc in original_docs:
        t = doc.get("text","")
        if t and t not in seen: seen.add(t); merged.append({**doc, "source": "original"})
    for doc in bridge_docs:
        t = doc.get("text","")
        if t and t not in seen: seen.add(t); merged.append({**doc, "source": "bridge"})
    merged.sort(key=lambda x: x.get("score",0), reverse=True)
    for i, d in enumerate(merged[:final_top_k], 1): d["rank"] = i
    return merged[:final_top_k]


def filter_by_candidates(docs, candidates, cand_names, final_k=3):
    """
    B4-EXP9 — Candidate-Aware Filtering
    ─────────────────────────────────────
    من K=10 وثائق مسترجعة، نُبقي الوثائق التي تذكر على الأقل مرشحاً واحداً.
    الوثائق المتبقية أكثر صلة بالسؤال لأنها تحتوي معلومات عن المرشحين.
    إذا لم تبقَ وثائق كافية (< final_k)، نُكمل بأعلى الوثائق score.

    الفرق عن B4-EXP5 (Cross-Encoder): لا يحتاج نموذج re-ranking خارجي،
    مجرد text matching — أسرع وأخف.

    المرجع: Candidate-Aware Retrieval — BioCreative MedHop best practices
    """
    cand_set = set()
    for cid, cname in zip(candidates, cand_names):
        cand_set.add(cid.lower())
        cand_set.add(cname.lower())

    # فصل الوثائق التي تذكر مرشحاً عن الباقي
    with_cand, without_cand = [], []
    for doc in docs:
        text_low = doc.get("text", "").lower()
        if any(c in text_low for c in cand_set):
            with_cand.append(doc)
        else:
            without_cand.append(doc)

    # ابنِ القائمة النهائية: المرشحون أولاً، ثم الباقي كـ fallback
    filtered = with_cand[:final_k]
    if len(filtered) < final_k:
        # أضف من الوثائق التي لا تذكر مرشحاً لإكمال العدد
        needed = final_k - len(filtered)
        filtered += without_cand[:needed]

    for i, d in enumerate(filtered, 1):
        d["rank"] = i
    return filtered

# ══════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════

def do_retrieval_base(record, top_k, extra_bridge_terms=None):
    """Hybrid Scored Retrieval — الأساسي لكل التجارب."""
    dn  = record.get("query_drug_name","")
    sup = record.get("supports",[])
    exp = expand_query(dn)
    flat = exp["terms"] if exp["success"] else []
    wt   = get_weighted_terms(dn)
    if extra_bridge_terms:
        for term in extra_bridge_terms:
            if term and len(term) > 2:
                wt.append({"term": term, "weight": 5.0, "type": "bridge_entity"})
                flat.append(term)
    return retrieve_hybrid_scored(query=record.get("query",""), supports=sup,
                                   drug_name=dn, flat_terms=flat, weighted_terms=wt, top_k=top_k)

# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run_experiment(strategy="guided_retrieval", top_k=3, sample_size=None, verbose=True):
    """
    Pipeline موحّد لكل استراتيجيات Entity Bridging.
    غيّري EXP_STRATEGY في __main__ للتبديل بين التجارب.
    """
    if sample_size is None: sample_size = DIAG_SAMPLE_SIZE
    if strategy not in STRATEGY_PROMPTS:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_PROMPTS.keys())}")

    retriever_type = "hybrid_scored"
    needs_bridge   = strategy in NEEDS_BRIDGE
    baseline_em, baseline_label = 22.22, "B3-EXP21"

    exp_name  = f"eb_{strategy}_k{top_k}"
    pred_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_predictions.json")
    logs_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_logs.json")

    tmpl = STRATEGY_PROMPTS[strategy]

    # تحديد المجموعة للعرض
    group = "A (Prompt-Based)" if strategy in {"no_decomp","enriched","cot_enriched","forced_reasoning","bridge_pivoted"} else "B (Retrieval-Based)"

    print(f"\n{'='*65}")
    print(f"  Entity Bridging Pipeline — Phase 2")
    print(f"{'='*65}")
    print(f"  Strategy : {strategy}  [Group {group}]")
    print(f"  Model    : {OLLAMA_MODEL_NAME} | K={top_k}")
    print(f"  Baseline : {baseline_label}={baseline_em}%")
    print(f"  Output   : {pred_file}\n")

    pipeline_start = time.time()
    client = get_ollama_client()
    if not check_model_available(client): sys.exit(1)
    data = load_data(sample_size); total = len(data)
    print(f"  [OK] {total} questions loaded")

    bridge_cache = _load_bridge_cache() if needs_bridge else {}
    cache_hits = 0
    if needs_bridge: print(f"  [OK] Bridge cache: {len(bridge_cache)} entries")

    existing = {}
    if os.path.exists(pred_file):
        try:
            with open(pred_file, encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("success"): existing[r["question_id"]] = r
            if existing: print(f"  [INFO] Resuming — {len(existing)} done")
        except: pass

    predictions  = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct"))
    nm_count     = sum(1 for p in predictions if p.get("nm_hit"))
    print(f"\n--- Running ({total - len(existing)} remaining) ---\n")

    for i, record in enumerate(data):
        qid      = record["id"]
        if qid in existing: continue
        drug_name = record.get("query_drug_name","")
        drug_id   = record.get("query_drug_id","")
        ans_name  = record.get("answer_name","")

        # ── Step 1: Bridge Extraction ──────────────────────────────────
        bridge_info = ""
        if needs_bridge:
            ck = f"mechanism::{drug_name}"
            if ck in bridge_cache:
                bridge_info = bridge_cache[ck]; cache_hits += 1
            else:
                br = run_inference_bridge(client, BRIDGE_PROMPT.format(drug_name=drug_name, docs=drug_name), f"{qid}_br")
                if br["success"]:
                    bridge_info = _extract_bridge(br["raw_response"])
                    if bridge_info and len(bridge_info) > 3:
                        bridge_cache[ck] = bridge_info; _save_bridge_cache(bridge_cache)
            if not bridge_info: bridge_info = "unknown"

        # ── Step 2: Bridge terms للـ Retrieval ─────────────────────────
        bridge_terms = None
        if bridge_info and bridge_info != "unknown":
            clean = bridge_info.replace("inhibits","").replace("blocks","").replace("acts","").replace("via","")
            bridge_terms = [w for w in clean.split() if len(w) > 2]

        # ── Step 3: Retrieval (حسب الاستراتيجية) ───────────────────────
        ret_start = time.time()

        if strategy == "reranked":
            # B4-EXP5: Hybrid K=10 → Cross-Encoder → أفضل K=3
            raw_docs  = do_retrieval_base(record, RERANK_FETCH_K, extra_bridge_terms=bridge_terms)
            retrieved = rerank_documents(query=record.get("query",""), bridge_info=bridge_info,
                                          retrieved_docs=raw_docs, final_k=top_k)

        elif strategy == "dual_retrieval":
            # B4-EXP7: بحث 1 (الدواء) + بحث 2 (الآلية) + merge
            original_docs = do_retrieval_base(record, top_k, extra_bridge_terms=None)
            bridge_docs   = []
            if bridge_info and bridge_info != "unknown":
                bridge_query = f"{drug_name} {bridge_info}"
                bridge_docs  = retrieve_semantic(query=bridge_query,
                                                  supports=record.get("supports",[]),
                                                  drug_name=drug_name, top_k=top_k)
            retrieved = merge_and_deduplicate(original_docs, bridge_docs, final_top_k=top_k+2)

        elif strategy == "cand_filtered":
            # B4-EXP9: Guided K=10 → فلتر بناء على ذكر المرشحين → أفضل K=3
            # الفكرة: الوثائق التي تذكر المرشحين أكثر صلة من التي لا تذكرهم
            raw_docs   = do_retrieval_base(record, CAND_FILTER_FETCH_K, extra_bridge_terms=bridge_terms)
            cand_names_list = record.get("candidate_names", record["candidates"])
            retrieved  = filter_by_candidates(raw_docs, record["candidates"], cand_names_list, final_k=top_k)

        else:
            # كل الاستراتيجيات الأخرى: Hybrid Scored مع bridge terms (إن وُجدت)
            extra = bridge_terms if strategy in {"guided_retrieval","bridge_pivoted","cand_filtered"} else None
            retrieved = do_retrieval_base(record, top_k, extra_bridge_terms=extra)

        docs_text = _format_docs(retrieved)

        # ── Step 4: Prompt building ─────────────────────────────────────
        cands = "\n".join(f"- {n} ({c})" if n != c else f"- {c}"
                          for c, n in zip(record["candidates"], record.get("candidate_names", record["candidates"])))

        if strategy == "no_decomp":
            prompt = tmpl.format(drug_name=drug_name, drug_id=drug_id, docs=docs_text, candidates_text=cands)
        else:
            prompt = tmpl.format(drug_name=drug_name, drug_id=drug_id, bridge_info=bridge_info,
                                  docs=docs_text, candidates_text=cands)

        # ── Step 5: LLM Inference ───────────────────────────────────────
        inf_result = run_inference(client, prompt, qid)

        if inf_result["success"]:
            raw = _clean(inf_result["raw_response"])
            if raw: inf_result["raw_response"] = raw

        prediction = extract_drug_id(inf_result["raw_response"], record["candidates"]) if inf_result["success"] else ""
        is_correct = bool(prediction) and prediction.upper() == record["answer"].upper() and inf_result["success"]
        if is_correct: correct_count += 1
        nm_hit = bool(ans_name) and ans_name.lower() in inf_result.get("raw_response","").lower() and inf_result["success"]
        if nm_hit: nm_count += 1

        pred_record = {
            "question_id":    qid,
            "query_drug_name": drug_name, "query_drug_id": drug_id,
            "prediction":     prediction,  "answer": record["answer"], "answer_name": ans_name,
            "is_correct":     is_correct,  "nm_hit": nm_hit,
            "bridge_info":    bridge_info if needs_bridge else "",
            "raw_response":   inf_result.get("raw_response",""),
            "success":        inf_result.get("success", False),
            "error":          inf_result.get("error",""),
            "model":          OLLAMA_MODEL_NAME, "retriever": retriever_type, "top_k": top_k,
            "strategy":       strategy, "inference_time": inf_result.get("inference_time", 0),
        }
        predictions.append(pred_record); existing[qid] = pred_record; save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 5 == 0 or done == total):
            ans_now = sum(1 for p in predictions if p.get("success"))
            em = (correct_count / ans_now * 100) if ans_now else 0
            b_short = f" | B:'{bridge_info[:22]}'" if needs_bridge else ""
            print(f"  [{done:>3}/{total}] {'OK' if is_correct else '--'} {qid:<12} | EM: {em:.1f}%{b_short} | {(time.time()-pipeline_start)/60:.1f}m")

    metrics = calc_metrics(predictions)
    tt = time.time() - pipeline_start
    delta = metrics['strict_em'] - baseline_em

    print(f"\n{'='*65}")
    print(f"  RESULTS — Entity Bridging ({strategy})")
    print(f"{'='*65}")
    print(f"  Strict EM% : {metrics['strict_em']}%  |  Correct: {metrics['correct']}/{metrics['total']}")
    print(f"  Baseline   : {baseline_em}%  |  Delta: {'+' if delta>=0 else ''}{delta:.2f}%")
    if needs_bridge: print(f"  Cache hits : {cache_hits}")
    print(f"  Time       : {tt/60:.1f}m")
    print(f"  Output     : {pred_file}")
    print(f"{'='*65}\n")

    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump({"experiment": {"phase": "Phase 2 - Entity Bridging", "strategy": strategy,
                   "group": group, "retriever": retriever_type, "top_k": top_k,
                   "model": OLLAMA_MODEL_NAME, "baseline_em": baseline_em},
                   "results": metrics, "timing": {"total_min": round(tt/60,2)}},
                  f, indent=2, ensure_ascii=False)
    return metrics


# ══════════════════════════════════════════════════════════════
# ENTRY POINT — غيّري EXP_STRATEGY فقط للتبديل بين التجارب
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ┌─────────────────────────────────────────────────────────┐
    # │  المجموعة A — Prompt-Based                              │
    # ├─────────────────────────────────────────────────────────┤
    # │  no_decomp        Control — بدون bridge                 │
    # │  enriched         B4-EXP2 → 24.27%  (نُفِّذت)          │
    # │  cot_enriched     B4-EXP3 → 32.16%  (نُفِّذت)          │
    # │  forced_reasoning B4-EXP6 → 16.67%  (نُفِّذت)          │
    # │  bridge_pivoted   B5-EXP4 → ؟؟؟     (الجديدة)          │
    # ├─────────────────────────────────────────────────────────┤
    # │  المجموعة B — Retrieval-Based                           │
    # ├─────────────────────────────────────────────────────────┤
    # │  guided_retrieval B4-EXP4 → 33.33%  (نُفِّذت) ✓ أفضل  │
    # │  reranked         B4-EXP5 → 24.85%  (نُفِّذت)          │
    # │  dual_retrieval   B4-EXP7 → 17.25%  (نُفِّذت)          │
    # │  cand_filtered    B4-EXP9 → ؟؟؟     (الجديدة)          │
    # │  Guided K=10 → فلتر بالمرشحين → K=3                   │
    # └─────────────────────────────────────────────────────────┘

    # ← غيّري هذا السطر فقط
    EXP_STRATEGY = "guided_retrieval"   # B4-EXP9 — الجديدة
    EXP_TOP_K    = 3

    run_experiment(strategy=EXP_STRATEGY, top_k=EXP_TOP_K,
                   sample_size=DIAG_SAMPLE_SIZE, verbose=True)