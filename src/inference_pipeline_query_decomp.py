"""
src/inference_pipeline_query_decomp.py
========================================
BiomedicalMulti-Hop QA — Phase 3: Query Decomposition (True 2-Hop)

الموقف الأكاديمي:
  هذا الملف يطبّق "Query Decomposition" الحقيقي بالمعنى الأكاديمي للكلمة.
  الفرق الجوهري عن الملف السابق (inference_pipeline_entity_bridging.py):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Entity Bridging (Phase 2): الـ bridge يُستخدم لتحسين retrieval أو  │
  │  prompt واحد. لا يوجد انفصال حقيقي بين الـ hops.                   │
  │                                                                     │
  │  Query Decomposition (Phase 3): السؤال الأصلي يُقسَّم لـ 2 أسئلة  │
  │  فرعية مستقلة تمامًا، كل منها له retrieval + inference منفصل.      │
  └─────────────────────────────────────────────────────────────────────┘

  بنية MedHop:
    Drug A → [bridge_entity: protein/enzyme] → Drug B
    Hop 1: "What mechanism/enzyme does Drug A act on?" → bridge_entity
    Hop 2: "Which candidate drug is affected by [bridge_entity]?" → Drug B

  ما الفرق عن dual_retrieval (B4-EXP7) الذي أعطى 17.25%؟
    - dual_retrieval استخدم query = "{drug_name} {bridge_info}" ← خلط الـ hops
    - هنا Hop 2 يبحث بـ bridge_entity فقط، بدون drug_name ← انفصال حقيقي
    - plus: الـ prompt لـ Hop 2 مختلف تمامًا (يسأل عن bridge، لا عن Drug A)

المرجع الأكاديمي:
  - UETQuintet (BioCreative IX 2025, Rank 2): "Anchor-based Iteration"
    الـ answer من hop i يصبح "anchor" في hop i+1
  - IRCoT (2023): Interleaved Retrieval - كل خطوة تسترجع بناءً على السابقة
  - REAP (2026): Recursive Fact Substitution - المجهول يُحدَّث بالحقيقة المكتشفة

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  التجارب:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  QD-EXP1: sequential_2hop
  ─────────────────────────
  الفكرة: فصل كامل بين الـ hops
    Hop1_Retrieval: docs عن Drug A → استخرج bridge_entity
    Hop2_Retrieval: docs عن bridge_entity فقط (بدون Drug A)
    Hop2_Prompt: "Which candidate is affected by [bridge_entity]?"
  المرجع: UETQuintet Anchor-based Iteration

  QD-EXP2: ircot_subquestion
  ───────────────────────────
  الفكرة: توليد أسئلة فرعية صريحة (explicit sub-questions)
    Step 1: اللي يولّد: "Sub-Q1: What is Drug A's mechanism? Sub-Q2: ..."
    Step 2: يجاوب Sub-Q1 من drug docs
    Step 3: يستخدم الجواب كـ query لـ Sub-Q2
    Step 4: يدمج الجوابين للإجابة النهائية
  المرجع: IRCoT (2023), CLaC Systems (2025)

  QD-EXP3: candidate_narrowing
  ──────────────────────────────
  الفكرة: تصفية المرشحين قبل الإجابة النهائية
    Bridge extraction (مكلفة مسبقًا)
    LLM يختار top-3 candidates المتوافقين مع الـ bridge
    تشغيل guided_retrieval على المرشحين المُصفَّين فقط
  الميزة: يقلّص ضجيج المرشحين (avg 9 → 3) بدون extra retrieval
  المرجع: REAP Recursive Evaluation (2026)
"""

import json, os, sys, time, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE, OUTPUTS_DIR, DIAG_SAMPLE_SIZE, OLLAMA_MODEL_NAME,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MAX_TOKENS, LLM_NUM_CTX,
)
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.query_expander import expand_query
from src.retriever_hybrid_scored import retrieve_hybrid_scored
from src.query_expander_structured import get_weighted_terms
from src.prompt_builder import extract_drug_id
from src.retriever_semantic import retrieve_semantic

# ══════════════════════════════════════════════════════════════
# BRIDGE CACHE (shared with entity bridging pipeline)
# ══════════════════════════════════════════════════════════════
BRIDGE_CACHE_FILE = os.path.join(OUTPUTS_DIR, "bridge_cache.json")

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

# ══════════════════════════════════════════════════════════════
# HOP-1 PROMPT — استخراج الـ bridge (نفس كما في Phase 2)
# ══════════════════════════════════════════════════════════════
PROMPT_HOP1_BRIDGE = """\
You are a biomedical expert. What is the PRIMARY mechanism of action of {drug_name}?
Be concise: name the specific enzyme, receptor, or protein target.

Evidence:
{docs}

Answer in ONE short phrase (e.g., "inhibits CYP2D6", "blocks MAO-A", "agonist at mu-opioid receptor"):
MECHANISM:"""

# ══════════════════════════════════════════════════════════════
# QD-EXP1: SEQUENTIAL 2-HOP PROMPTS
# ══════════════════════════════════════════════════════════════

# Hop 2 prompt: يسأل عن الـ bridge entity فقط، بدون ذكر Drug A
PROMPT_QD1_HOP2 = """\
You are a biomedical expert specializing in drug mechanisms and interactions.

Supporting Evidence (retrieved for the mechanism "{bridge_entity}"):
{docs}

The biological mechanism involved is: {bridge_entity}

From the following candidate drugs, which ONE is DIRECTLY AFFECTED BY or INTERACTS WITH \
the mechanism "{bridge_entity}"?

A drug "interacts with" this mechanism if it:
- Is metabolized by the same enzyme
- Acts on the same receptor or protein target
- Inhibits or induces the same pathway

Candidates:
{candidates_text}

Answer with the DrugBank ID only (format: DBxxxxx):"""

# ══════════════════════════════════════════════════════════════
# QD-EXP2: IRCOT-STYLE SUB-QUESTION PROMPTS
# ══════════════════════════════════════════════════════════════

# Step 1: توليد الأسئلة الفرعية
PROMPT_QD2_GENERATE_SUBQS = """\
You are a biomedical reasoning assistant.

Given this drug interaction question:
"{question}"

Drug: {drug_name}
Candidates: {candidates_brief}

Decompose this into exactly 2 sub-questions that will help find the answer:

Sub-question 1 should ask about the mechanism/target of {drug_name}.
Sub-question 2 should ask which candidate drug is related to that mechanism.

Output format (answer ONLY with these two lines):
SQ1: [sub-question about {drug_name}'s mechanism]
SQ2: [sub-question about which candidate is affected by that mechanism]"""

# Step 2: إجابة Sub-Q2 (بعد ما Sub-Q1 أُجيب بالـ bridge)
PROMPT_QD2_HOP2 = """\
You are a biomedical expert.

Context from Hop 1: {drug_name} works by {bridge_entity}

Sub-question to answer: {sub_q2}

Supporting Evidence (retrieved for: {bridge_entity}):
{docs}

Candidates:
{candidates_text}

Based on the sub-question and the evidence, answer with the DrugBank ID only (format: DBxxxxx):"""

# ══════════════════════════════════════════════════════════════
# QD-EXP3: CANDIDATE NARROWING PROMPTS
# ══════════════════════════════════════════════════════════════

# Step 1: تصفية المرشحين بناءً على الـ bridge
PROMPT_QD3_NARROW = """\
You are a pharmacology expert.

The drug {drug_name} works by: {bridge_entity}

From these candidate drugs, select the 3 MOST LIKELY to interact with {drug_name} \
via the mechanism "{bridge_entity}".

A candidate is "likely" if it:
- Is metabolized by or inhibits the same enzyme
- Targets the same receptor
- Competes for the same pathway

Candidates:
{candidates_text}

List ONLY the DrugBank IDs of the top 3 candidates (one per line, format: DBxxxxx):"""

# Step 2: الإجابة النهائية على المرشحين المُصفَّين
PROMPT_QD3_FINAL = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_entity}

Choose the correct answer from these PRE-FILTERED candidates \
(already narrowed based on mechanism compatibility):
{candidates_text}

The interacting drug shares or is affected by the mechanism: {bridge_entity}
Answer with the DrugBank ID only (format: DBxxxxx):"""

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def _clean(text):
    if not text: return ""
    text = re.sub(r"<\|endoftext\|>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|im_start\|>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    # تنظيف thinking tokens
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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

def _format_docs(retrieved, max_chars=400):
    if not retrieved: return "No supporting evidence available."
    return "\n".join(
        f"[{r['rank']}] {r['text'][:max_chars]}{'...' if len(r['text'])>max_chars else ''}"
        for r in retrieved
    )

def _extract_candidate_ids(raw, candidates):
    """استخراج قائمة IDs من ناتج التصفية."""
    found = []
    for cid in candidates:
        if cid.upper() in raw.upper():
            found.append(cid)
    return found[:3] if found else candidates[:3]  # fallback لأول 3

def load_data(n=None):
    with open(MEDHOP_FILE, encoding="utf-8") as f: data = json.load(f)
    return data[:n] if n else data

def _model_short():
    return re.sub(r"[/:\\.]+", "_", OLLAMA_MODEL_NAME).strip("_")[-30:]

def save_preds(preds, path):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

def calc_metrics(preds):
    total = len(preds)
    failed = sum(1 for p in preds if not p.get("success"))
    answered = total - failed
    correct = sum(
        1 for p in preds
        if p.get("success") and p.get("prediction","").strip().upper() == p.get("answer","").strip().upper()
        and p.get("prediction","").strip()
    )
    return {
        "total": total, "answered": answered, "failed": failed, "correct": correct,
        "strict_em": round(correct/total*100, 2) if total else 0,
        "lenient_em": round(correct/answered*100, 2) if answered else 0,
    }

def _run_llm_short(client, prompt, qid, max_tokens=200):
    """استدعاء سريع للنموذج بعدد tokens محدود."""
    result = {"raw_response": "", "success": False, "error": ""}
    for attempt in range(2):
        try:
            resp = client.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P,
                    "top_k": LLM_TOP_K, "num_predict": max_tokens,
                    "num_ctx": LLM_NUM_CTX
                }
            )
            raw = resp["message"]["content"].strip()
            if not raw: raise ValueError("Empty response")
            result.update({"raw_response": raw, "success": True})
            return result
        except Exception as e:
            result["error"] = str(e)
            if attempt == 0: time.sleep(2)
    return result

# ══════════════════════════════════════════════════════════════
# HOP-2 RETRIEVAL — البحث بالـ bridge entity فقط (بدون drug name)
# الفرق الجوهري عن dual_retrieval السابق
# ══════════════════════════════════════════════════════════════

def _retrieve_hop2_bridge_only(record, bridge_entity, top_k=3):
    """
    Retrieval مخصص لـ Hop 2:
    - يبحث بـ bridge_entity فقط كـ query
    - لا يُضيف drug_name (هذا كان خطأ dual_retrieval)
    - يُطبّق الـ bridge كـ weighted term بوزن عالي
    """
    supports = record.get("supports", [])

    # استخراج الكلمات المفتاحية من الـ bridge (إزالة الأفعال)
    clean_bridge = re.sub(
        r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|targets?|acts?|via|of|the|a|an|and|or|by)\b",
        "", bridge_entity, flags=re.IGNORECASE
    )
    bridge_terms = [w for w in clean_bridge.split() if len(w) > 2]

    # weighted_terms مع وزن mechanism عالي
    weighted_terms = [
        {"term": t, "weight": 4.0, "type": "mechanism"}
        for t in bridge_terms if t
    ]

    return retrieve_hybrid_scored(
        query=bridge_entity,        # ← query هو الـ bridge فقط، ليس Drug A
        supports=supports,
        drug_name="",               # ← بدون drug_name في هذا الـ hop
        flat_terms=bridge_terms,
        weighted_terms=weighted_terms,
        top_k=top_k
    )

# ══════════════════════════════════════════════════════════════
# QD-EXP1: SEQUENTIAL 2-HOP
# ══════════════════════════════════════════════════════════════

def _do_guided_retrieval(record, bridge_entity, top_k):
    """
    Guided Retrieval — نفس استراتيجية B4-EXP4 (33.33%) بالضبط:
    Score = Hybrid(Drug_Name) + Weighted_Term_Count(Bridge_Entity)
    تُستخدم كـ Hop 2 في كل التجارب بدل bridge-only الذي ثبت ضعفه.
    """
    drug_name = record.get("query_drug_name", "")
    exp = expand_query(drug_name)
    flat = exp["terms"] if exp["success"] else []
    wt   = get_weighted_terms(drug_name)

    if bridge_entity and bridge_entity != "unknown mechanism":
        clean = re.sub(
            r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|by|the|a|an|and|or)\b",
            "", bridge_entity, flags=re.IGNORECASE
        )
        for t in clean.split():
            if len(t) > 2:
                wt.append({"term": t, "weight": 5.0, "type": "mechanism"})
                flat.append(t)

    return retrieve_hybrid_scored(
        query=record.get("query", ""), supports=record.get("supports", []),
        drug_name=drug_name, flat_terms=flat, weighted_terms=wt, top_k=top_k
    )


def run_qd1_sequential(client, record, bridge_cache, top_k=3):
    """
    QD-EXP1: Sequential 2-Hop — FIXED (v2)
    ─────────────────────────────────────────
    الإصلاح: Hop 2 يستخدم الآن Guided Retrieval (Drug A + Bridge)
    بدل bridge-only الذي أثبت أنه يخسر وثائق مهمة.

    الـ "decomposition" يبقى في الـ PROMPT (يسأل من منظور الـ bridge)
    لكن الاسترجاع يبقى مدمجاً مثل B4-EXP4.

    Hop 1: bridge extraction (من الـ cache — مجاني)
    Hop 2: Guided retrieval (Drug A + Bridge) → Bridge-focused prompt

    المرجع: UETQuintet Anchor-based Iteration (BioCreative IX 2025)
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # ── Hop 1: Bridge (من الـ cache — لا استدعاء LLM إضافي) ──
    ck = f"mechanism::{drug_name}"
    bridge_entity = bridge_cache.get(ck, "")

    if not bridge_entity:
        hop1_docs = retrieve_hybrid_scored(
            query=drug_name, supports=record.get("supports", []),
            drug_name=drug_name, flat_terms=[drug_name], top_k=3
        )
        hop1_result = _run_llm_short(client, PROMPT_HOP1_BRIDGE.format(
            drug_name=drug_name, docs=_format_docs(hop1_docs)), f"{qid}_h1", max_tokens=150)
        if hop1_result["success"]:
            bridge_entity = _extract_bridge(hop1_result["raw_response"])
            if bridge_entity and len(bridge_entity) > 3:
                bridge_cache[ck] = bridge_entity
                _save_bridge_cache(bridge_cache)

    if not bridge_entity:
        bridge_entity = "unknown mechanism"

    # ── Hop 2: Guided Retrieval (FIXED — Drug A + Bridge) ────
    hop2_docs = _do_guided_retrieval(record, bridge_entity, top_k)

    # ── Hop 2: Bridge-focused Prompt ─────────────────────────
    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    hop2_prompt = PROMPT_QD1_HOP2.format(
        bridge_entity=bridge_entity,
        docs=_format_docs(hop2_docs),
        candidates_text=cands_text
    )

    hop2_result = run_inference(client, hop2_prompt, qid)
    prediction = ""
    if hop2_result["success"]:
        raw = _clean(hop2_result["raw_response"])
        hop2_result["raw_response"] = raw
        prediction = extract_drug_id(raw, candidates)

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "raw_response": hop2_result.get("raw_response", ""),
        "success": hop2_result.get("success", False),
        "error": hop2_result.get("error", ""),
        "inference_time": hop2_result.get("inference_time", 0),
        "strategy": "QD-EXP1_sequential_2hop_v2"
    }


# ══════════════════════════════════════════════════════════════
# QD-EXP2: IRCOT-STYLE SUB-QUESTION
# ══════════════════════════════════════════════════════════════

def run_qd2_ircot(client, record, bridge_cache, top_k=3):
    """
    QD-EXP2: IRCoT-style Explicit Sub-questions
    ─────────────────────────────────────────────
    Step 1: Generate 2 explicit sub-questions from original question
    Step 2: Answer Sub-Q1 (mechanism) from Drug A docs
    Step 3: Use Sub-Q1 answer as anchor for Sub-Q2 retrieval
    Step 4: Answer Sub-Q2 (candidate) from bridge docs

    المرجع: IRCoT (2023), CLaC Systems (2025)
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # candidates_brief للعرض المختصر في Sub-Q generation
    cands_brief = ", ".join(cand_names[:4]) + ("..." if len(cand_names) > 4 else "")

    # ── Step 1: Bridge (Sub-Q1 answer) من الـ cache أو الاستخراج ──
    ck = f"mechanism::{drug_name}"
    bridge_entity = bridge_cache.get(ck, "")

    if not bridge_entity:
        hop1_docs = retrieve_hybrid_scored(
            query=drug_name, supports=record.get("supports", []),
            drug_name=drug_name, flat_terms=[drug_name], top_k=3
        )
        hop1_prompt = PROMPT_HOP1_BRIDGE.format(
            drug_name=drug_name, docs=_format_docs(hop1_docs)
        )
        r = _run_llm_short(client, hop1_prompt, f"{qid}_h1", max_tokens=150)
        if r["success"]:
            bridge_entity = _extract_bridge(r["raw_response"])
            if bridge_entity and len(bridge_entity) > 3:
                bridge_cache[ck] = bridge_entity
                _save_bridge_cache(bridge_cache)

    if not bridge_entity:
        bridge_entity = "unknown mechanism"

    # ── Step 2: توليد Sub-Q2 الصريح ──────────────────────────
    # (Sub-Q1 تمّت إجابته ضمنيًا بالـ bridge)
    sub_q2 = f"Which of the candidates is metabolized by, targets, or is otherwise affected by: {bridge_entity}?"

    # ── Step 3: Guided Retrieval (FIXED — Drug A + Bridge) ───
    hop2_docs = _do_guided_retrieval(record, bridge_entity, top_k=top_k)

    # ── Step 4: Inference بالـ IRCoT prompt ──────────────────
    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    hop2_prompt = PROMPT_QD2_HOP2.format(
        drug_name=drug_name,
        bridge_entity=bridge_entity,
        sub_q2=sub_q2,
        docs=_format_docs(hop2_docs),
        candidates_text=cands_text
    )

    hop2_result = run_inference(client, hop2_prompt, qid)
    prediction = ""
    if hop2_result["success"]:
        raw = _clean(hop2_result["raw_response"])
        hop2_result["raw_response"] = raw
        prediction = extract_drug_id(raw, candidates)

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "sub_q2": sub_q2,
        "raw_response": hop2_result.get("raw_response", ""),
        "success": hop2_result.get("success", False),
        "error": hop2_result.get("error", ""),
        "inference_time": hop2_result.get("inference_time", 0),
        "strategy": "QD-EXP2_ircot_subquestion_v2"
    }


# ══════════════════════════════════════════════════════════════
# QD-EXP3: CANDIDATE NARROWING — FIXED (v2)
# ══════════════════════════════════════════════════════════════

# الـ prompt الأفضل ثبوتًا — نفس B4-EXP4 (33.33%)
# المتغيّر {bridge_info} ← نفس اسم placeholder كما في entity_bridging
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


def run_qd3_candidate_narrowing(client, record, bridge_cache, top_k=3):
    """
    QD-EXP3: Candidate Narrowing — FIXED (v2)
    ───────────────────────────────────────────
    Step 1: Bridge من الـ cache (مجاني)
    Step 2: LLM يصفّي 9→3 مرشحين بناءً على الـ bridge
    Step 3: Guided Retrieval (Drug A + Bridge) — نفس B4-EXP4 (كان صحيحاً)
    Step 4: FIXED — يستخدم الآن PROMPT_COT_ENRICHED (نفس B4-EXP4)
             بدل PROMPT_QD3_FINAL الذي كان أضعف

    الفرق عن B4-EXP4: إضافة تصفية المرشحين كخطوة وسيطة
    المرجع: REAP Recursive Evaluation (2026), DMIS Lab Decision-Maker (2025)
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # ── Step 1: Bridge من الـ cache ───────────────────────────
    ck = f"mechanism::{drug_name}"
    bridge_entity = bridge_cache.get(ck, "")

    if not bridge_entity:
        hop1_docs = retrieve_hybrid_scored(
            query=drug_name, supports=record.get("supports", []),
            drug_name=drug_name, flat_terms=[drug_name], top_k=3
        )
        r = _run_llm_short(client, PROMPT_HOP1_BRIDGE.format(
            drug_name=drug_name, docs=_format_docs(hop1_docs)), f"{qid}_h1", max_tokens=150)
        if r["success"]:
            bridge_entity = _extract_bridge(r["raw_response"])
            if bridge_entity and len(bridge_entity) > 3:
                bridge_cache[ck] = bridge_entity
                _save_bridge_cache(bridge_cache)

    if not bridge_entity:
        bridge_entity = "unknown mechanism"

    # ── Step 2: تصفية المرشحين بناءً على الـ bridge ──────────
    cands_text_full = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    narrow_prompt = PROMPT_QD3_NARROW.format(
        drug_name=drug_name,
        bridge_entity=bridge_entity,
        candidates_text=cands_text_full
    )

    narrow_result = _run_llm_short(client, narrow_prompt, f"{qid}_narrow", max_tokens=100)

    narrowed_candidates = candidates  # fallback: كل المرشحين
    narrowed_names = cand_names
    if narrow_result["success"]:
        raw_narrow = _clean(narrow_result["raw_response"])
        filtered_ids = _extract_candidate_ids(raw_narrow, candidates)
        if len(filtered_ids) >= 2:
            narrowed_candidates = filtered_ids
            narrowed_names = [
                cand_names[candidates.index(c)] if c in candidates else c
                for c in narrowed_candidates
            ]

    # ── Step 3: Retrieval موجَّه بالـ bridge + drug name ─────
    exp = expand_query(drug_name)
    flat = exp["terms"] if exp["success"] else []

    # bridge terms بوزن عالي في الـ retrieval
    bridge_clean = re.sub(
        r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|by|the|a|an)\b",
        "", bridge_entity, flags=re.IGNORECASE
    )
    bridge_terms_wt = [
        {"term": t, "weight": 5.0, "type": "mechanism"}
        for t in bridge_clean.split() if len(t) > 2
    ]
    flat.extend([t["term"] for t in bridge_terms_wt])

    wt = get_weighted_terms(drug_name) + bridge_terms_wt

    retrieved = retrieve_hybrid_scored(
        query=record.get("query", ""), supports=record.get("supports", []),
        drug_name=drug_name, flat_terms=flat, weighted_terms=wt, top_k=top_k
    )

    # ── Step 4: Final Answer بـ PROMPT_COT_ENRICHED (FIXED) ──
    # نفس الـ prompt الأفضل من B4-EXP4 بدل PROMPT_QD3_FINAL الضعيف
    # الفرق: نعرض المرشحين المُصفَّين فقط للتقليل من الضجيج
    cands_narrowed_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(narrowed_candidates, narrowed_names)
    )

    final_prompt = PROMPT_COT_ENRICHED.format(
        docs=_format_docs(retrieved),
        drug_name=drug_name,
        drug_id=drug_id,
        bridge_info=bridge_entity,         # ← نفس placeholder اسم B4-EXP4
        candidates_text=cands_narrowed_text
    )

    final_result = run_inference(client, final_prompt, qid)
    prediction = ""
    if final_result["success"]:
        raw = _clean(final_result["raw_response"])
        final_result["raw_response"] = raw
        # نبحث في كل المرشحين (ليس فقط المُصفَّين) لتجنّب خسارة الإجابة الصحيحة
        prediction = extract_drug_id(raw, candidates)

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "original_candidates_count": len(candidates),
        "narrowed_candidates": narrowed_candidates,
        "narrowed_count": len(narrowed_candidates),
        "raw_response": final_result.get("raw_response", ""),
        "success": final_result.get("success", False),
        "error": final_result.get("error", ""),
        "inference_time": final_result.get("inference_time", 0),
        "strategy": "QD-EXP3_candidate_narrowing_v2"
    }


# ══════════════════════════════════════════════════════════════
# QD-EXP4a: GUIDED RETRIEVAL + CANDIDATE NARROWING (Combined)
# ══════════════════════════════════════════════════════════════
# الفكرة: دمج أفضل شيء من كل تجربة:
#   guided_retrieval (33.33%) → أفضل retrieval
#   candidate_narrowing → تصفية المرشحين
#   PROMPT_COT_ENRICHED → أفضل prompt مثبت
#
# الفرق عن QD-EXP3 (27.19%):
#   QD-EXP3 كان يبني الـ retrieval يدوياً → احتمال فروق طفيفة
#   هنا نستخدم _do_guided_retrieval بالضبط مثل B4-EXP4
#
# المرجع: REAP (2026), DMIS Lab Decision-Maker (2025)

PROMPT_NARROW_4a = """\
You are a pharmacology expert.

Drug: {drug_name} — mechanism: {bridge_entity}

From these candidates, select the 3 MOST LIKELY to interact with {drug_name} \
via this mechanism. A candidate interacts if it:
- Is metabolized by or inhibits the same enzyme
- Acts on the same receptor/target
- Competes in the same pathway

Candidates:
{candidates_text}

Output ONLY 3 DrugBank IDs, one per line (format DBxxxxx):"""


def run_qd4a(client, record, bridge_cache, top_k=3):
    """
    QD-EXP4a: Guided Retrieval + Candidate Narrowing
    ──────────────────────────────────────────────────
    Step 1: Bridge من الـ cache (مجاني)
    Step 2: LLM يصفّي 9→3 مرشحين بناءً على الـ bridge
    Step 3: _do_guided_retrieval بالضبط = B4-EXP4 retrieval
    Step 4: PROMPT_COT_ENRICHED على المرشحين المُصفَّين

    هذه هي النسخة "النظيفة" التي تضمن أن الـ retrieval مطابق تماماً لـ B4-EXP4.
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # Bridge من الـ cache — مجاني
    ck = f"mechanism::{drug_name}"
    bridge_entity = bridge_cache.get(ck, "unknown mechanism")

    # ── Step 1: Candidate Narrowing (استدعاء LLM قصير) ───────
    cands_text_full = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    narrow_r = _run_llm_short(
        client,
        PROMPT_NARROW_4a.format(
            drug_name=drug_name,
            bridge_entity=bridge_entity,
            candidates_text=cands_text_full
        ),
        f"{qid}_narrow", max_tokens=80
    )

    narrowed = candidates  # fallback: كل المرشحين
    narrowed_names = cand_names
    if narrow_r["success"]:
        raw_n = _clean(narrow_r["raw_response"])
        found = [c for c in candidates if c.upper() in raw_n.upper()]
        if len(found) >= 2:
            narrowed = found[:3]
            narrowed_names = [
                cand_names[candidates.index(c)] if c in candidates else c
                for c in narrowed
            ]

    # ── Step 2: Guided Retrieval = B4-EXP4 بالضبط ───────────
    retrieved = _do_guided_retrieval(record, bridge_entity, top_k)

    # ── Step 3: PROMPT_COT_ENRICHED على المرشحين المُصفَّين ──
    cands_narrowed_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(narrowed, narrowed_names)
    )

    final_prompt = PROMPT_COT_ENRICHED.format(
        docs=_format_docs(retrieved),
        drug_name=drug_name, drug_id=drug_id,
        bridge_info=bridge_entity,
        candidates_text=cands_narrowed_text
    )

    final_result = run_inference(client, final_prompt, qid)
    prediction = ""
    if final_result["success"]:
        raw = _clean(final_result["raw_response"])
        final_result["raw_response"] = raw
        prediction = extract_drug_id(raw, candidates)  # بحث في كل المرشحين

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "narrowed_candidates": narrowed,
        "narrowed_count": len(narrowed),
        "raw_response": final_result.get("raw_response", ""),
        "success": final_result.get("success", False),
        "error": final_result.get("error", ""),
        "inference_time": final_result.get("inference_time", 0),
        "strategy": "QD-EXP4a_guided_narrowing"
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# QD-EXP5: CANDIDATE-CONDITIONED RETRIEVAL (CCR)
# ══════════════════════════════════════════════════════════════
# التشخيص الجذري:
#   Bridge quality ليست المشكلة (71% specific في الصحيح، 78% في الخاطئ)
#   المشكلة: الوثائق المُسترجعة لا تذكر المرشح الصحيح صراحةً
#   الدليل: Gold Chain 76.9% (يعطي الوثائق التي تذكر الإجابة) مقابل 33.33%
#   MedHop paper: "providing docs guaranteed to be relevant greatly improves performance"
#
# الحل: لكل مرشح → استرجع وثائق تذكره هو + Drug A + Bridge معاً
#   بدل: retrieve(drug_A + bridge) → ask(all candidates)
#   نفعل: for each candidate: retrieve(drug_A + candidate + bridge) → score
#   ثم: أعطِ النموذج الوثائق الأكثر ملاءمة لكل مرشح
#
# المرجع:
#   - MedHop paper (Welbl 2018): Gold Chain experiment = oracle upper bound
#   - Orekhovich (IBMC, 43.9%): candidate-aware retrieval approach
#   - UETQuintet (83.8%): per-candidate scoring in ensemble

PROMPT_CCR_FINAL = """\
You are a biomedical expert specializing in drug interactions.

Drug A: {drug_name} ({drug_id})
Mechanism of {drug_name}: {bridge_entity}

Below is evidence retrieved specifically for each candidate drug:

{candidate_evidence}

Based on the evidence above, which candidate drug INTERACTS with {drug_name} \
via its mechanism "{bridge_entity}"?

The interacting drug is affected by this mechanism (metabolized by same enzyme, \
acts on same receptor, or competes in same pathway).

Answer with the DrugBank ID only (format: DBxxxxx):"""


def _retrieve_for_candidate(record, drug_name, candidate_id, candidate_name,
                             bridge_entity, top_k=2):
    """
    استرجاع وثائق مخصصة لمرشح واحد:
    query = drug_A + candidate_name + bridge_terms
    هدف: جلب وثائق تذكر الدواءين معاً عبر الـ bridge
    """
    supports = record.get("supports", [])

    # بناء query يشمل Drug A + Candidate + Bridge
    query = f"{drug_name} {candidate_name}"

    # weighted terms: bridge بوزن عالي + اسم المرشح بوزن عالي
    bridge_clean = re.sub(
        r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|by|the|a|an|and|or)\b",
        "", bridge_entity, flags=re.IGNORECASE
    )
    bridge_terms = [w for w in bridge_clean.split() if len(w) > 2]

    wt = get_weighted_terms(drug_name)
    # أضف الـ bridge بوزن عالي
    for t in bridge_terms:
        wt.append({"term": t, "weight": 4.0, "type": "mechanism"})
    # أضف اسم المرشح بوزن عالي جداً — هذا هو الفرق الجوهري
    cand_words = [w for w in candidate_name.split() if len(w) > 2]
    for w in cand_words:
        wt.append({"term": w, "weight": 6.0, "type": "mechanism"})

    flat = [drug_name] + bridge_terms + cand_words

    docs = retrieve_hybrid_scored(
        query=query, supports=supports,
        drug_name=drug_name,
        flat_terms=flat, weighted_terms=wt,
        top_k=top_k
    )
    return docs


def run_qd5_ccr(client, record, bridge_cache, top_k=3):
    """
    QD-EXP5: Candidate-Conditioned Retrieval (CCR)
    ────────────────────────────────────────────────
    التشخيص: المشكلة الجذرية = الوثائق لا تذكر المرشح الصحيح
    الحل: لكل مرشح نسترجع وثائق تذكره هو + Drug A معاً

    Algorithm:
      Step 1: Bridge من الـ cache (مجاني)
      Step 2: تصفية 9→3 مرشحين بالـ bridge (LLM call قصير)
      Step 3: لكل مرشح من الـ 3: retrieve(drug_A + candidate + bridge)
      Step 4: دمج الوثائق الخاصة بكل مرشح → عرضها للـ LLM مُنظمة

    المرجع: MedHop paper (Gold Chain), Orekhovich (IBMC), UETQuintet
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # ── Step 1: Bridge من الـ cache ──────────────────────────
    ck = f"mechanism::{drug_name}"
    bridge_entity = bridge_cache.get(ck, "unknown mechanism")

    # ── Step 2: تصفية 9→3 مرشحين ────────────────────────────
    cands_text_full = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    narrow_r = _run_llm_short(
        client,
        PROMPT_NARROW_4a.format(
            drug_name=drug_name,
            bridge_entity=bridge_entity,
            candidates_text=cands_text_full
        ),
        f"{qid}_narrow", max_tokens=80
    )

    shortlist_ids = candidates[:3]  # fallback
    shortlist_names = cand_names[:3]
    if narrow_r["success"]:
        raw_n = _clean(narrow_r["raw_response"])
        found = [c for c in candidates if c.upper() in raw_n.upper()]
        if len(found) >= 2:
            shortlist_ids = found[:3]
            shortlist_names = [
                cand_names[candidates.index(c)] if c in candidates else c
                for c in shortlist_ids
            ]

    # ── Step 3: Per-candidate retrieval ──────────────────────
    # لكل مرشح من الـ 3: استرجع وثائق تذكره مع Drug A
    docs_per_candidate = {}
    for cid, cname in zip(shortlist_ids, shortlist_names):
        cand_docs = _retrieve_for_candidate(
            record, drug_name, cid, cname,
            bridge_entity, top_k=2  # وثيقتان لكل مرشح
        )
        docs_per_candidate[cid] = (cname, cand_docs)

    # ── Step 4: Build candidate evidence block ───────────────
    # عرض الوثائق منظمة حسب المرشح
    evidence_blocks = []
    for cid, (cname, docs) in docs_per_candidate.items():
        if docs:
            top_doc = docs[0]["text"][:300] if docs else "No relevant evidence found."
            evidence_blocks.append(
                f"Candidate: {cname} ({cid})\n"
                f"Evidence: {top_doc}..."
            )
        else:
            evidence_blocks.append(f"Candidate: {cname} ({cid})\nEvidence: No relevant evidence found.")

    candidate_evidence = "\n\n".join(evidence_blocks)

    # أيضاً أضف المرشحين غير المُصفَّين في النهاية (للـ fallback)
    remaining = [
        f"- {n} ({c})" for c, n in zip(candidates, cand_names)
        if c not in shortlist_ids
    ]
    if remaining:
        candidate_evidence += f"\n\nOther candidates (less likely): {', '.join(shortlist_ids + [c for c in candidates if c not in shortlist_ids])}"

    final_prompt = PROMPT_CCR_FINAL.format(
        drug_name=drug_name, drug_id=drug_id,
        bridge_entity=bridge_entity,
        candidate_evidence=candidate_evidence
    )

    final_result = run_inference(client, final_prompt, qid)
    prediction = ""
    if final_result["success"]:
        raw = _clean(final_result["raw_response"])
        final_result["raw_response"] = raw
        prediction = extract_drug_id(raw, candidates)

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "shortlisted_candidates": shortlist_ids,
        "raw_response": final_result.get("raw_response", ""),
        "success": final_result.get("success", False),
        "error": final_result.get("error", ""),
        "inference_time": final_result.get("inference_time", 0),
        "strategy": "QD-EXP5_ccr"
    }


EXPERIMENT_MAP = {
    # ── التجارب المنفّذة ──────────────────────────────────────
    "QD-EXP1": run_qd1_sequential,          # 24.85% → FIXED v2
    "QD-EXP2": run_qd2_ircot,               # 13.16% → FIXED v2
    "QD-EXP3": run_qd3_candidate_narrowing, # 27.19% → FIXED v2
    "QD-EXP4a": run_qd4a,                   # ??? → Guided + Narrowing
    # ── التجربة الجديدة (تعالج السبب الجذري) ─────────────────
    "QD-EXP5": run_qd5_ccr,  # Candidate-Conditioned Retrieval
    # ملاحظة: QD-EXP4b (Answer Verification) → Stage 4
}

def run_experiment(strategy="QD-EXP1", top_k=3, sample_size=None, verbose=True):
    """
    Pipeline موحّد لتجارب Query Decomposition.

    الـ baseline للمقارنة: guided_retrieval (B4-EXP4) = 33.33%
    """
    if sample_size is None: sample_size = DIAG_SAMPLE_SIZE
    if strategy not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(EXPERIMENT_MAP.keys())}")

    baseline_em    = 33.33
    baseline_label = "B4-EXP4 (guided_retrieval)"

    exp_name  = f"qd_{strategy.lower()}_k{top_k}"
    pred_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_predictions.json")
    logs_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_logs.json")

    print(f"\n{'='*65}")
    print(f"  Query Decomposition Pipeline — Phase 3")
    print(f"{'='*65}")
    print(f"  Strategy : {strategy}")
    print(f"  Model    : {OLLAMA_MODEL_NAME} | K={top_k}")
    print(f"  Baseline : {baseline_label} = {baseline_em}%")
    print(f"  Output   : {pred_file}\n")

    pipeline_start = time.time()
    client = get_ollama_client()
    if not check_model_available(client): sys.exit(1)

    data = load_data(sample_size)
    total = len(data)
    print(f"  [OK] {total} questions loaded")

    bridge_cache = _load_bridge_cache()
    print(f"  [OK] Bridge cache: {len(bridge_cache)} entries (free Hop-1 answers)")

    existing = {}
    if os.path.exists(pred_file):
        try:
            with open(pred_file, encoding="utf-8") as f:
                for r in json.load(f):
                    if r.get("success"): existing[r["question_id"]] = r
            if existing: print(f"  [INFO] Resuming — {len(existing)} done")
        except: pass

    predictions   = list(existing.values())
    correct_count = sum(1 for p in predictions if p.get("is_correct"))
    print(f"\n--- Running ({total - len(existing)} remaining) ---\n")

    exp_fn = EXPERIMENT_MAP[strategy]

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing: continue

        drug_name = record.get("query_drug_name", "")
        ans_name  = record.get("answer_name", "")

        # ── تشغيل التجربة المختارة ─────────────────────────────
        result = exp_fn(client, record, bridge_cache, top_k=top_k)

        prediction = result.get("prediction", "")
        is_correct = (
            bool(prediction) and
            prediction.upper() == record["answer"].upper() and
            result.get("success", False)
        )
        if is_correct: correct_count += 1

        pred_record = {
            "question_id":      qid,
            "query_drug_name":  drug_name,
            "query_drug_id":    record.get("query_drug_id", ""),
            "prediction":       prediction,
            "answer":           record["answer"],
            "answer_name":      ans_name,
            "is_correct":       is_correct,
            "model":            OLLAMA_MODEL_NAME,
            "top_k":            top_k,
            **result  # يُضيف كل بيانات التجربة (bridge, sub_q2, etc.)
        }
        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 5 == 0 or done == total):
            ans_now = sum(1 for p in predictions if p.get("success"))
            em = (correct_count / ans_now * 100) if ans_now else 0
            bridge_short = result.get("bridge_entity", "")[:22]
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} | "
                  f"EM: {em:.1f}% | Bridge: '{bridge_short}' | "
                  f"{(time.time()-pipeline_start)/60:.1f}m")

    metrics = calc_metrics(predictions)
    tt = time.time() - pipeline_start
    delta = metrics['strict_em'] - baseline_em

    print(f"\n{'='*65}")
    print(f"  RESULTS — Query Decomposition ({strategy})")
    print(f"{'='*65}")
    print(f"  Strict EM% : {metrics['strict_em']}%  |  Correct: {metrics['correct']}/{metrics['total']}")
    print(f"  Baseline   : {baseline_em}%  |  Delta: {'+' if delta>=0 else ''}{delta:.2f}%")
    print(f"  Bridge cache used: {sum(1 for p in predictions if p.get('bridge_entity'))}/{metrics['total']}")
    print(f"  Time       : {tt/60:.1f}m")
    print(f"  Output     : {pred_file}")
    print(f"{'='*65}\n")

    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": {
                "phase": "Phase 3 - Query Decomposition",
                "strategy": strategy,
                "model": OLLAMA_MODEL_NAME,
                "top_k": top_k,
                "baseline_em": baseline_em,
                "baseline_label": baseline_label,
            },
            "results": metrics,
            "timing": {"total_min": round(tt/60, 2)}
        }, f, indent=2, ensure_ascii=False)

    return metrics


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ┌─────────────────────────────────────────────────────────┐
    # │  Stage 3 — Query Decomposition                          │
    # │  Baseline: B4-EXP4 (guided_retrieval) = 33.33%          │
    # ├─────────────────────────────────────────────────────────┤
    # │  QD-EXP1  Sequential 2-Hop     → 24.85%  (نُفِّذت)         │
    # │           FIXED v2: Hop2 = Guided Retrieval             │
    # │                                                         │
    # │  QD-EXP2  IRCoT Sub-question   → 13.16%  (نُفِّذت)         │
    # │           FIXED v2: Hop2 = Guided Retrieval             │
    # │                                                         │
    # │  QD-EXP3  Candidate Narrowing  → 27.19%  (نُفِّذت)         │
    # │           FIXED v2: Final Prompt = PROMPT_COT_ENRICHED  │
    # │                                                         │
    # │  QD-EXP4a Guided + Narrowing   →  28.36% (نُفِّذت)         │
    # │           _do_guided_retrieval + CoT + تصفية 9→3       │
    # │                                                         │
    # │  QD-EXP5  Candidate-Conditioned Retrieval → ??? (جديدة)│
    # │    لكل مرشح: retrieve(drug_A + candidate + bridge)     │
    # │    التشخيص: المشكلة = docs لا تذكر المرشح الصحيح      │
    # │    مرجع: MedHop paper Gold Chain + Orekhovich (43.9%)  │
    # └─────────────────────────────────────────────────────────┘

    EXP_STRATEGY = "QD-EXP5"   # ← غيّري هنا فقط
    EXP_TOP_K    = 3

    run_experiment(
        strategy=EXP_STRATEGY,
        top_k=EXP_TOP_K,
        sample_size=DIAG_SAMPLE_SIZE,
        verbose=True
    )