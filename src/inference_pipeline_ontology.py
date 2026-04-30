"""
src/inference_pipeline_ontology.py
=====================================
BiomedicalMulti-Hop QA — Stage 4: Ontology Verification & Correction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  هذه المرحلة تبني على أفضل نتيجة من Stage 2/3 (B4-EXP4 = 33.33%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

الفكرة الجوهرية:
  بدل external ontology (DrugBank API, MeSH terms) التي تحتاج اتصالاً خارجياً،
  نستخدم الـ SUPPORTS الموجودة داخل كل سؤال كـ "ontology محلية":
  - avg 36.4 وثيقة MEDLINE لكل سؤال
  - هذه الوثائق تحتوي على chain الدواء-بروتين-دواء
  - نُحقق من الإجابة بفحص: هل يوجد وثيقة تذكر Drug A + bridge + candidate معاً؟

لماذا ليس external ontology؟
  - DrugBank (free tier): محدود، يحتاج API key
  - MeSH/UMLS: يحتاج registration و license
  - Bridge cache (LLM-extracted): 25% منها ضبابية وليست موثوقة 100%
  - الـ SUPPORTS موجودة بالفعل في الداتا ← مجانية، موثوقة، مرجعية

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  التجارب:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OV-EXP1: Support-Based Candidate Reranking (ZERO LLM calls)
  ─────────────────────────────────────────────────────────────
  score(candidate) = #support_docs mentioning candidate_name
                   + bonus إذا تذكر bridge_terms أيضاً
  نختار المرشح ذو أعلى score مباشرةً بدون نموذج
  المرجع: MedHop paper Section 6.1 (Max-mention baseline, لكن مُحسَّن بالـ bridge)

  OV-EXP2: Guided Retrieval + Post-hoc Consistency Check
  ───────────────────────────────────────────────────────
  الخطوات:
    1. شغّل B4-EXP4 (guided_retrieval) → prediction₁
    2. تحقق: هل في support doc يذكر prediction₁ مع bridge_entity؟
       → إذا نعم (consistent): أبقِ prediction₁
       → إذا لا (inconsistent): اختر المرشح الأعلى support-score من OV-EXP1
  Cost: نفس B4-EXP4 + verification (text matching فقط، 0 LLM calls إضافية)
  المرجع: MedHop paper "Gold Chain" insight + TreeQA (2025) consistency check

  OV-EXP3: Bridge-Corrected Refinement Loop (مع LLM)
  ─────────────────────────────────────────────────────
  الخطوات:
    1. شغّل B4-EXP4 → prediction₁
    2. تحقق via support docs (مثل OV-EXP2)
    3. إذا inconsistent → استخرج bridge أدق من الوثائق المُسترجعة
       (بدل استخدام bridge_cache القديم قد يكون ضبابياً)
    4. شغّل guided_retrieval مرة ثانية بالـ bridge المُحسَّن → prediction₂
  Cost: 1 LLM call إضافي فقط للحالات غير المتسقة (~40% من الأسئلة)
  المرجع: Thought Rollback (2024), REAP Recursive Evaluation (2026)
"""

import json, os, sys, time, re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDHOP_FILE, OUTPUTS_DIR, DIAG_SAMPLE_SIZE, OLLAMA_MODEL_NAME,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_NUM_CTX,
)
from src.llm_runner import get_ollama_client, check_model_available, run_inference
from src.query_expander import expand_query
from src.retriever_hybrid_scored import retrieve_hybrid_scored
from src.query_expander_structured import get_weighted_terms
from src.prompt_builder import extract_drug_id

# ══════════════════════════════════════════════════════════════
# BRIDGE CACHE
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

def _clean_bridge(raw_bridge):
    """
    استخراج الجزء المفيد من الـ bridge الضبابي.
    يُحسَّن جودة bridge_cache دون استدعاء LLM.
    """
    if not raw_bridge or raw_bridge == "unknown mechanism":
        return raw_bridge

    # إذا كان قصيراً وواضحاً → ممتاز
    if len(raw_bridge) <= 80 and "Based on" not in raw_bridge:
        return raw_bridge

    # استخراج من النصوص الضبابية
    patterns = [
        r'\*\*([^*]{5,60})\*\*',           # **bold mechanism**
        r'(?:is|are)\s+([a-zA-Z\s\-]{5,50}(?:inhibitor|agonist|antagonist|blocker|receptor|enzyme|reductase))',
        r'mechanism[^:]*:\s*\*?\*?([^.\n*]{5,60})',
        r'MECHANISM:\s*(.{5,60})',
    ]
    for p in patterns:
        m = re.search(p, raw_bridge, re.IGNORECASE)
        if m:
            result = m.group(1).strip()
            if len(result) >= 5:
                return result[:80]

    # آخر حل: خذ أول جملة قصيرة
    sentences = raw_bridge.split('.')
    for s in sentences:
        s = s.strip()
        if 5 < len(s) < 80 and "Based on" not in s and "provided" not in s:
            return s

    return raw_bridge[:80]  # fallback

# ══════════════════════════════════════════════════════════════
# SUPPORT DOCUMENT SCORING — قلب Stage 4
# ══════════════════════════════════════════════════════════════

def score_candidates_by_supports(record, bridge_entity, candidates, cand_names):
    """
    يُسجّل كل مرشح بناءً على عدد وثائق الـ supports التي تذكره.
    هذه الوظيفة هي "الـ ontology المحلية" — بدون LLM، بدون API خارجي.

    score(candidate) =
        mention_count: #docs mentioning candidate_name
        + bridge_bonus: #docs mentioning BOTH candidate AND bridge_terms
        + drug_a_bonus: #docs mentioning BOTH candidate AND Drug A (stronger link)

    من ورقة MedHop: الوثائق التي تذكر الدواءين معاً عبر بروتين مشترك
    هي بالضبط الوثائق الأكثر صلة.
    """
    supports = record.get("supports", [])
    drug_name = record.get("query_drug_name", "").lower()

    # استخراج كلمات الـ bridge
    bridge_terms = []
    if bridge_entity and bridge_entity != "unknown mechanism":
        clean = re.sub(
            r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|by|the|a|an|and|or|of|via)\b",
            "", bridge_entity, flags=re.IGNORECASE
        )
        bridge_terms = [w.lower() for w in clean.split() if len(w) > 2]

    scores = {}
    for cid, cname in zip(candidates, cand_names):
        cname_lower = cname.lower()
        cid_lower = cid.lower()

        mention_count = 0
        bridge_bonus = 0
        drug_a_bonus = 0

        for doc in supports:
            doc_text = doc.lower() if isinstance(doc, str) else str(doc).lower()

            # هل تذكر المرشح؟
            mentions_cand = cname_lower in doc_text or cid_lower in doc_text
            # هل تذكر الـ bridge؟
            mentions_bridge = any(t in doc_text for t in bridge_terms) if bridge_terms else False
            # هل تذكر الدواء A أيضاً؟
            mentions_drug_a = drug_name in doc_text

            if mentions_cand:
                mention_count += 1
                if mentions_bridge:
                    bridge_bonus += 2    # تذكر المرشح والآلية معاً ← مهم جداً
                if mentions_drug_a:
                    drug_a_bonus += 1    # تذكر الدواءين معاً ← مهم

        scores[cid] = {
            "mention_count": mention_count,
            "bridge_bonus": bridge_bonus,
            "drug_a_bonus": drug_a_bonus,
            "total": mention_count + bridge_bonus + drug_a_bonus,
            "name": cname
        }

    return scores


def pick_best_by_supports(scores, candidates):
    """اختيار المرشح ذو أعلى support score."""
    if not scores:
        return candidates[0] if candidates else ""

    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            scores.get(c, {}).get("total", 0),
            scores.get(c, {}).get("bridge_bonus", 0),
            scores.get(c, {}).get("mention_count", 0)
        ),
        reverse=True
    )
    return sorted_cands[0]


def is_consistent(prediction, bridge_entity, record):
    """
    هل الإجابة المتوقعة متسقة مع الـ bridge في الـ supports؟
    يُعيد True إذا وُجد على الأقل وثيقة تذكر المرشح مع الـ bridge.
    """
    if not prediction: return False
    supports = record.get("supports", [])
    candidates = record.get("candidates", [])
    cand_names = record.get("candidate_names", candidates)

    # اجد اسم المرشح
    pred_name = ""
    for cid, cname in zip(candidates, cand_names):
        if cid.upper() == prediction.upper():
            pred_name = cname.lower()
            break

    if not pred_name: return False

    bridge_terms = []
    if bridge_entity and bridge_entity != "unknown mechanism":
        clean = re.sub(
            r"\b(inhibits?|blocks?|activates?|agonist|antagonist|binds?|by|the|a|an|and|or|of|via)\b",
            "", bridge_entity, flags=re.IGNORECASE
        )
        bridge_terms = [w.lower() for w in clean.split() if len(w) > 2]

    for doc in supports:
        doc_text = doc.lower() if isinstance(doc, str) else str(doc).lower()
        mentions_pred = pred_name in doc_text or prediction.lower() in doc_text
        mentions_bridge = any(t in doc_text for t in bridge_terms) if bridge_terms else True
        if mentions_pred and mentions_bridge:
            return True
    return False


# ══════════════════════════════════════════════════════════════
# PROMPTS للـ Bridge Refinement (OV-EXP3)
# ══════════════════════════════════════════════════════════════

PROMPT_REFINE_BRIDGE = """\
You are a biomedical expert. The drug {drug_name} interacts with another drug via a biological mechanism.

From the supporting evidence below, extract the PRIMARY mechanism or protein target of {drug_name}:

Evidence:
{docs}

Give ONE concise phrase (e.g., "inhibits CYP3A4", "agonist at GLP-1 receptor"):
MECHANISM:"""

PROMPT_OV3_FINAL = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug: {drug_name} ({drug_id})
Refined Mechanism: {bridge_entity}

The support documents showed that {drug_name} works via: {bridge_entity}

From these candidates, which ONE is AFFECTED BY or INTERACTS WITH {drug_name} via this mechanism?
{candidates_text}

Answer with DrugBank ID only (format: DBxxxxx):"""

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _clean_response(text):
    if not text: return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+\|>", "", text)
    return text.strip("* \n").strip()

def _format_docs(retrieved, max_chars=400):
    if not retrieved: return "No supporting evidence available."
    return "\n".join(
        f"[{r['rank']}] {r['text'][:max_chars]}..."
        for r in retrieved if r.get('text')
    )

def _run_short(client, prompt, qid, max_tokens=150):
    for attempt in range(2):
        try:
            resp = client.chat(
                model=OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P,
                         "top_k": LLM_TOP_K, "num_predict": max_tokens,
                         "num_ctx": LLM_NUM_CTX}
            )
            raw = resp["message"]["content"].strip()
            if raw: return {"raw_response": raw, "success": True}
        except Exception as e:
            if attempt == 0: time.sleep(2)
    return {"raw_response": "", "success": False}

def _do_guided_retrieval(record, bridge_entity, top_k=3):
    """نفس B4-EXP4 بالضبط — الأفضل في Stage 2."""
    drug_name = record.get("query_drug_name", "")
    exp = expand_query(drug_name)
    flat = exp["terms"] if exp["success"] else []
    wt = get_weighted_terms(drug_name)

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

# الـ CoT prompt نفسه من B4-EXP4 (الأفضل المُثبَت)
PROMPT_GUIDED = """\
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
    answered = sum(1 for p in preds if p.get("success"))
    correct = sum(
        1 for p in preds if p.get("success")
        and p.get("prediction", "").strip().upper() == p.get("answer", "").strip().upper()
        and p.get("prediction", "").strip()
    )
    return {
        "total": total, "answered": answered, "correct": correct,
        "strict_em": round(correct / total * 100, 2) if total else 0,
        "lenient_em": round(correct / answered * 100, 2) if answered else 0,
    }


# ══════════════════════════════════════════════════════════════
# OV-EXP1: SUPPORT-BASED RERANKING (ZERO LLM CALLS)
# ══════════════════════════════════════════════════════════════

def run_ov1_support_rerank(client, record, bridge_cache, top_k=3):
    """
    OV-EXP1: Support-Based Candidate Reranking
    ────────────────────────────────────────────
    لا LLM calls على الإطلاق. يسجّل كل مرشح بناءً على:
    - عدد الـ supports التي تذكره
    - عدد الـ supports التي تذكره مع الـ bridge entity

    هذا baseline ذكي لـ Stage 4:
    - أبسط من Max-mention (ورقة MedHop الأصلية)
    - لكن يستخدم الـ bridge knowledge التي بنيناها في Stage 2

    إذا حقق > 33.33% → يعني الـ supports تحتوي الإشارة الكافية
    وهذا يُعزز حجة "retrieval is the bottleneck"
    """
    drug_name = record.get("query_drug_name", "")
    qid = record["id"]
    candidates = record["candidates"]
    # إذا candidate_names موجودة استخدمها، وإلا استخدم الـ IDs
    # (الـ score_candidates_by_supports يبحث بالاسم والـ ID معاً فلا مشكلة)
    cand_names = record.get("candidate_names") or candidates

    # Bridge من الـ cache (مع تنظيف) — يدعم كلا الـ prefix القديم والجديد
    raw_bridge = (
        bridge_cache.get(f"mechanism::{drug_name}")
        or bridge_cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    bridge_entity = _clean_bridge(raw_bridge) if raw_bridge else "unknown mechanism"

    # تسجيل المرشحين بناءً على الـ supports
    scores = score_candidates_by_supports(record, bridge_entity, candidates, cand_names)
    prediction = pick_best_by_supports(scores, candidates)

    return {
        "prediction": prediction,
        "bridge_entity": bridge_entity,
        "scores": {k: v["total"] for k, v in scores.items()},
        "raw_response": f"Support-based: {scores.get(prediction, {}).get('name', '')}",
        "success": True,
        "error": "",
        "inference_time": 0,
        "strategy": "OV-EXP1_support_rerank"
    }


# ══════════════════════════════════════════════════════════════
# OV-EXP2: GUIDED + CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════

def run_ov2_guided_consistency(client, record, bridge_cache, top_k=3):
    """
    OV-EXP2: Guided Retrieval (B4-EXP4) + Post-hoc Consistency Check
    ───────────────────────────────────────────────────────────────────
    Step 1: شغّل B4-EXP4 (best Stage 2 result) → prediction₁
    Step 2: تحقق: هل في support doc يذكر prediction₁ مع bridge_entity؟
      → Consistent: أبقِ prediction₁
      → Inconsistent: بدّل للمرشح الأعلى support-score (من OV-EXP1)

    الفرق عن B4-EXP4: الـ consistency check مجاني (0 LLM calls إضافية)
    المرجع: MedHop "Gold Chain" insight + TreeQA (2025) invalidation
    """
    drug_name = record.get("query_drug_name", "")
    drug_id = record.get("query_drug_id", "")
    qid = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # Bridge (مُنظَّف) — يدعم كلا الـ prefix القديم والجديد
    raw_bridge = (
        bridge_cache.get(f"mechanism::{drug_name}")
        or bridge_cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    bridge_entity = _clean_bridge(raw_bridge) if raw_bridge else "unknown mechanism"

    # Step 1: Guided Retrieval + LLM (= B4-EXP4)
    retrieved = _do_guided_retrieval(record, bridge_entity, top_k)

    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    prompt = PROMPT_GUIDED.format(
        docs=_format_docs(retrieved),
        drug_name=drug_name, drug_id=drug_id,
        bridge_info=bridge_entity,
        candidates_text=cands_text
    )

    result = run_inference(client, prompt, qid)
    prediction1 = ""
    if result["success"]:
        raw = _clean_response(result["raw_response"])
        result["raw_response"] = raw
        prediction1 = extract_drug_id(raw, candidates)

    # Step 2: Consistency Check (0 LLM calls)
    # OV-EXP2 (الأصلي): is_consistent() — شرط صارم (يتطلب ذكر الجواب+bridge في نفس الوثيقة)
    consistent = is_consistent(prediction1, bridge_entity, record) if prediction1 else False
    final_prediction = prediction1
    corrected = False

    if not consistent or not prediction1:
        # اختر المرشح الأعلى support-score
        scores = score_candidates_by_supports(record, bridge_entity, candidates, cand_names)
        support_best = pick_best_by_supports(scores, candidates)

        # فقط نبدّل إذا الـ support candidate مختلف عن prediction₁
        if support_best and support_best != prediction1:
            final_prediction = support_best
            corrected = True

    return {
        "prediction": final_prediction,
        "prediction_stage2": prediction1,
        "consistent": consistent,
        "corrected": corrected,
        "bridge_entity": bridge_entity,
        "raw_response": result.get("raw_response", ""),
        "success": result.get("success", False),
        "error": result.get("error", ""),
        "inference_time": result.get("inference_time", 0),
        "strategy": "OV-EXP2_guided_consistency"
    }


# ══════════════════════════════════════════════════════════════
# OV-EXP4: CONSERVATIVE CONSISTENCY CHECK (الإصلاح الحقيقي)
# ══════════════════════════════════════════════════════════════

def run_ov4_conservative_check(client, record, bridge_cache, top_k=3):
    """
    OV-EXP4: Conservative Consistency Check
    ─────────────────────────────────────────
    الفرق الجوهري عن OV-EXP2:
      OV-EXP2: يُبدّل الإجابة إذا لم تظهر مع bridge في نفس وثيقة (صارم جداً)
               → 87.5% false correction rate → يُلغي إجابات B4 الصحيحة
      OV-EXP4: يُبدّل الإجابة فقط إذا support_score = 0
               أي الإجابة لا تُذكر في أي وثيقة على الإطلاق (غير موجودة)
               → يحمي إجابات B4 الصحيحة التي تظهر في الـ supports بدون bridge
               → يُصحح فقط الحالات التي النموذج فيها هلوس كلياً

    من التحليل: 65 سؤال كانت B4 صح لكن OV-EXP2 خرّبتها
    هذا التعديل يحمي معظمها.

    الخطوات:
      1. شغّل B4-EXP4 → prediction₁
      2. احسب support_score(prediction₁) = عدد الوثائق التي تذكره
      3. إذا score > 0: الإجابة موجودة في الـ supports → أبقِها (consistent)
         إذا score = 0: الإجابة غير موجودة في أي وثيقة → استبدل بالأفضل

    المرجع: تحليل OV-EXP2 failure analysis (182/208 false corrections)
    """
    drug_name = record.get("query_drug_name", "")
    drug_id   = record.get("query_drug_id", "")
    qid       = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names") or candidates

    # Bridge
    raw_bridge = (
        bridge_cache.get(f"mechanism::{drug_name}")
        or bridge_cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    bridge_entity = _clean_bridge(raw_bridge) if raw_bridge else "unknown mechanism"

    # Step 1: Guided Retrieval + LLM (= B4-EXP4 بالضبط)
    retrieved = _do_guided_retrieval(record, bridge_entity, top_k)

    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    prompt = PROMPT_GUIDED.format(
        docs=_format_docs(retrieved),
        drug_name=drug_name, drug_id=drug_id,
        bridge_info=bridge_entity,
        candidates_text=cands_text
    )

    result = run_inference(client, prompt, qid)
    prediction1 = ""
    if result["success"]:
        raw = _clean_response(result["raw_response"])
        result["raw_response"] = raw
        prediction1 = extract_drug_id(raw, candidates)

    # Step 2: Conservative Check — احسب support_score للإجابة
    scores = score_candidates_by_supports(record, bridge_entity, candidates, cand_names)
    pred1_score = scores.get(prediction1, {}).get("total", 0) if prediction1 else 0

    # consistent = الإجابة مذكورة في الـ supports (حتى ولو score=1)
    consistent   = (pred1_score > 0)
    final_prediction = prediction1
    corrected    = False

    if not consistent or not prediction1:
        # الإجابة غير موجودة في أي وثيقة أو فارغة → استبدل بالأفضل
        support_best = pick_best_by_supports(scores, candidates)
        if support_best and support_best != prediction1:
            final_prediction = support_best
            corrected = True

    return {
        "prediction":        final_prediction,
        "prediction_stage2": prediction1,
        "pred1_score":       pred1_score,
        "consistent":        consistent,
        "corrected":         corrected,
        "bridge_entity":     bridge_entity,
        "raw_response":      result.get("raw_response", ""),
        "success":           result.get("success", False),
        "error":             result.get("error", ""),
        "inference_time":    result.get("inference_time", 0),
        "strategy":          "OV-EXP4_conservative_check"
    }


# ══════════════════════════════════════════════════════════════
# OV-EXP3: BRIDGE REFINEMENT LOOP
# ══════════════════════════════════════════════════════════════

def run_ov3_bridge_refinement(client, record, bridge_cache, top_k=3):
    """
    OV-EXP3: Bridge Refinement Correction Loop
    ─────────────────────────────────────────────
    Step 1: شغّل B4-EXP4 → prediction₁
    Step 2: تحقق via supports
    Step 3: إذا inconsistent → استخرج bridge جديد من الوثائق المُسترجعة
            (أكثر دقة من bridge_cache لأنه مبني على retrieved docs الفعلية)
    Step 4: شغّل guided_retrieval بالـ bridge المُحسَّن → prediction₂

    الميزة: يُعالج 25% من الـ bridges الضبابية في الـ cache
    المرجع: Thought Rollback (2024), REAP Recursive Evaluation (2026)
    """
    drug_name = record.get("query_drug_name", "")
    drug_id = record.get("query_drug_id", "")
    qid = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names", candidates)

    # Bridge أولي (مُنظَّف) — يدعم كلا الـ prefix القديم والجديد
    ck = f"mechanism::{drug_name}"  # نكتب دائماً بالـ prefix الجديد
    raw_bridge = (
        bridge_cache.get(ck)
        or bridge_cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    bridge_entity = _clean_bridge(raw_bridge) if raw_bridge else "unknown mechanism"

    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    # ── Step 1: First attempt (= B4-EXP4) ───────────────────
    retrieved1 = _do_guided_retrieval(record, bridge_entity, top_k)
    prompt1 = PROMPT_GUIDED.format(
        docs=_format_docs(retrieved1),
        drug_name=drug_name, drug_id=drug_id,
        bridge_info=bridge_entity,
        candidates_text=cands_text
    )
    result1 = run_inference(client, prompt1, qid)
    prediction1 = ""
    if result1["success"]:
        raw = _clean_response(result1["raw_response"])
        result1["raw_response"] = raw
        prediction1 = extract_drug_id(raw, candidates)

    # ── Step 2: Consistency Check ────────────────────────────
    consistent = is_consistent(prediction1, bridge_entity, record) if prediction1 else False

    if consistent or not result1["success"]:
        # الإجابة الأولى متسقة → أبقِها
        return {
            "prediction": prediction1,
            "bridge_entity": bridge_entity,
            "bridge_refined": False,
            "attempts": 1,
            "raw_response": result1.get("raw_response", ""),
            "success": result1.get("success", False),
            "error": result1.get("error", ""),
            "inference_time": result1.get("inference_time", 0),
            "strategy": "OV-EXP3_bridge_refinement"
        }

    # ── Step 3: Bridge Refinement (استخرج bridge أفضل) ──────
    refine_result = _run_short(
        client,
        PROMPT_REFINE_BRIDGE.format(
            drug_name=drug_name,
            docs=_format_docs(retrieved1, max_chars=300)
        ),
        f"{qid}_refine",
        max_tokens=100
    )

    bridge_refined = bridge_entity  # fallback
    if refine_result["success"]:
        raw_bridge_new = _clean_response(refine_result["raw_response"])
        # استخرج من "MECHANISM: ..."
        m = re.search(r"MECHANISM:\s*(.+)", raw_bridge_new, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip().split("\n")[0][:100]
            if len(extracted) > 5:
                bridge_refined = extracted
                # حدّث الـ cache بالـ bridge المُحسَّن
                bridge_cache[ck] = bridge_refined
                _save_bridge_cache(bridge_cache)

    # ── Step 4: Second attempt بالـ bridge المُحسَّن ─────────
    retrieved2 = _do_guided_retrieval(record, bridge_refined, top_k)
    prompt2 = PROMPT_OV3_FINAL.format(
        docs=_format_docs(retrieved2),
        drug_name=drug_name, drug_id=drug_id,
        bridge_entity=bridge_refined,
        candidates_text=cands_text
    )
    result2 = run_inference(client, prompt2, f"{qid}_r2")
    prediction2 = prediction1  # fallback للإجابة الأولى

    if result2["success"]:
        raw2 = _clean_response(result2["raw_response"])
        result2["raw_response"] = raw2
        pred2 = extract_drug_id(raw2, candidates)
        if pred2:
            prediction2 = pred2

    return {
        "prediction": prediction2,
        "prediction_attempt1": prediction1,
        "bridge_entity": bridge_entity,
        "bridge_refined": bridge_refined,
        "was_inconsistent": True,
        "attempts": 2,
        "raw_response": result2.get("raw_response", result1.get("raw_response", "")),
        "success": result2.get("success", result1.get("success", False)),
        "error": result2.get("error", ""),
        "inference_time": result1.get("inference_time", 0) + result2.get("inference_time", 0),
        "strategy": "OV-EXP3_bridge_refinement"
    }


# ══════════════════════════════════════════════════════════════
# OV-EXP5: SUPPORT HINTS IN PROMPT (Proactive Ontology)
# ══════════════════════════════════════════════════════════════

# Prompt مُحسَّن يتضمن Support Hints داخله
PROMPT_OV5_WITH_HINTS = """\
You are a biomedical expert specializing in drug interactions.

Supporting Evidence:
{docs}

Drug in question: {drug_name} ({drug_id})
Known mechanism: {bridge_info}

Candidate mention frequency in supporting documents:
{support_hints}

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- {drug_name} has mechanism: {bridge_info}
- The "mention frequency" above shows how often each candidate appears in the supporting literature
- Look for a candidate that is AFFECTED BY this mechanism (metabolized by same enzyme, targets same receptor, shares same pathway)
- Use both your biomedical knowledge AND the mention frequency as evidence
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer

Answer:"""


def _build_support_hints(scores):
    """
    بناء نص الـ hints من نتائج score_candidates_by_supports.
    مثال:
      - Warfarin (DB00682): 5 mentions (+3 with mechanism)
      - Aspirin (DB00945): 2 mentions
      - Metformin (DB00331): not found in supporting documents
    """
    lines = []
    for cid, info in sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True):
        name = info.get("name", cid)
        mc   = info.get("mention_count", 0)
        bb   = info.get("bridge_bonus", 0)
        if mc == 0:
            lines.append(f"  - {name} ({cid}): not found in supporting documents")
        elif bb > 0:
            lines.append(f"  - {name} ({cid}): {mc} mention{'s' if mc != 1 else ''} (+{bb} with mechanism)")
        else:
            lines.append(f"  - {name} ({cid}): {mc} mention{'s' if mc != 1 else ''}")
    return "\n".join(lines)


def run_ov5_support_hints(client, record, bridge_cache, top_k=3):
    """
    OV-EXP5: Support Hints in Prompt — Proactive Ontology
    ───────────────────────────────────────────────────────
    الفرق الجوهري عن OV-EXP2/3/4 (كلها post-hoc):
      OV-EXP2/3/4: تُصحح الجواب بعد النموذج
      OV-EXP5:     تُضاف المعلومة داخل الـ prompt قبل أن يُجيب

    الخطوات:
      1. نفس retrieval B4-EXP4 (guided)
      2. احسب support_scores لكل مرشح (من الـ supports كاملاً)
      3. أنشئ "Support Hints" نصية: كم مرة يُذكر كل مرشح
      4. أضف الـ hints داخل الـ prompt
      5. النموذج نفسه يوازن بين معرفته البيولوجية + الـ hints

    المنطق: النموذج Qwen3.5-9B أفضل حكماً من أي تصحيح خارجي.
    بدل إجباره بعد الإجابة، نُغني مدخلاته.

    المرجع: RAG-style evidence augmentation — analogous to
             MedRAG (2024) و RRAG (Retrieval-Rerank-Answer-Generate)
    التوقع: 34–37% (التجربة الأكثر واعداً في Stage 4)
    """
    drug_name  = record.get("query_drug_name", "")
    drug_id    = record.get("query_drug_id", "")
    qid        = record["id"]
    candidates = record["candidates"]
    cand_names = record.get("candidate_names") or candidates

    # Bridge (مُنظَّف)
    raw_bridge = (
        bridge_cache.get(f"mechanism::{drug_name}")
        or bridge_cache.get(f"2hop_mechanism::{drug_name}")
        or ""
    )
    bridge_entity = _clean_bridge(raw_bridge) if raw_bridge else "unknown mechanism"

    # Step 1: Guided Retrieval (= B4-EXP4 بالضبط)
    retrieved = _do_guided_retrieval(record, bridge_entity, top_k)

    # Step 2: Support Scores (من كامل الـ supports، لا من الـ retrieved فقط)
    # نستخدم كامل الـ supports لأنها أكثر شمولاً من K=3 وثائق مسترجعة
    scores = score_candidates_by_supports(record, bridge_entity, candidates, cand_names)

    # Step 3: بناء Support Hints
    support_hints = _build_support_hints(scores)

    # Step 4: Prompt مع الـ hints
    cands_text = "\n".join(
        f"- {n} ({c})" if n != c else f"- {c}"
        for c, n in zip(candidates, cand_names)
    )

    prompt = PROMPT_OV5_WITH_HINTS.format(
        docs=_format_docs(retrieved),
        drug_name=drug_name, drug_id=drug_id,
        bridge_info=bridge_entity,
        support_hints=support_hints,
        candidates_text=cands_text
    )

    # Step 5: LLM Inference (نفس B4-EXP4)
    result = run_inference(client, prompt, qid)
    prediction = ""
    if result["success"]:
        raw = _clean_response(result["raw_response"])
        result["raw_response"] = raw
        prediction = extract_drug_id(raw, candidates)

    # أعلى مرشح بالـ supports (للتحليل فقط، لا للتصحيح)
    support_top = max(scores, key=lambda c: scores[c]["total"]) if scores else ""
    support_agreed = (prediction.upper() == support_top.upper()) if prediction and support_top else False

    return {
        "prediction":       prediction,
        "bridge_entity":    bridge_entity,
        "support_top":      support_top,
        "support_agreed":   support_agreed,
        "hint_scores":      {k: v["total"] for k, v in scores.items()},
        "raw_response":     result.get("raw_response", ""),
        "success":          result.get("success", False),
        "error":            result.get("error", ""),
        "inference_time":   result.get("inference_time", 0),
        "strategy":         "OV-EXP5_support_hints"
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

EXPERIMENT_MAP = {
    # ── تجارب بدون LLM calls إضافية ─────────────────────────
    "OV-EXP1": run_ov1_support_rerank,       # 0 LLM calls — support-based scoring         → 16.67%
    "OV-EXP2": run_ov2_guided_consistency,   # = B4-EXP4 + consistency correction (صارم)   → 20.47%
    # ── تجارب مع LLM call إضافي ──────────────────────────────
    "OV-EXP3": run_ov3_bridge_refinement,    # B4-EXP4 + bridge refinement loop             → 27.78%
    "OV-EXP4": run_ov4_conservative_check,   # B4-EXP4 + conservative check (score>0)       → 32.16%
    # ── OV-EXP5: تجربة جديدة — proactive بدل post-hoc ───────
    "OV-EXP5": run_ov5_support_hints,        # B4-EXP4 + Support Hints in Prompt            → ???
    # ملاحظة: OV-EXP5 هو التحسين الأكثر واعداً — يُضيف معلومة
    # الـ supports داخل الـ prompt نفسه بدل التصحيح البعدي
}


def run_experiment(strategy="OV-EXP2", top_k=3, sample_size=None, verbose=True):
    """
    Pipeline موحّد لتجارب Stage 4: Ontology Verification & Correction.

    Baseline للمقارنة: B4-EXP4 (guided_retrieval) = 33.33%
    الهدف: تجاوز الـ baseline بخطوة verification بسيطة

    ترتيب التنفيذ المنطقي:
    1. OV-EXP1 أولاً (0 LLM calls، أسرع) → لفهم قيمة الـ supports
    2. OV-EXP2 (يبني على OV-EXP1 + Stage 2) → الأكثر واعداً
    3. OV-EXP3 (أبطأ، للحالات الصعبة فقط)
    """
    if sample_size is None: sample_size = DIAG_SAMPLE_SIZE
    assert strategy in EXPERIMENT_MAP, f"Unknown strategy: {strategy}"

    baseline_em    = 33.33
    baseline_label = "B4-EXP4 (guided_retrieval)"

    exp_name  = f"ov_{strategy.lower()}_k{top_k}"
    pred_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_predictions.json")
    logs_file = os.path.join(OUTPUTS_DIR, f"{exp_name}_{_model_short()}_logs.json")

    print(f"\n{'='*65}")
    print(f"  Stage 4 — Ontology Verification & Correction")
    print(f"{'='*65}")
    print(f"  Strategy : {strategy}")
    print(f"  Model    : {OLLAMA_MODEL_NAME} | K={top_k}")
    print(f"  Baseline : {baseline_label} = {baseline_em}%")
    print(f"  Output   : {pred_file}\n")

    pipeline_start = time.time()

    # OV-EXP1 لا يحتاج Ollama
    if strategy == "OV-EXP1":
        client = None
    else:
        client = get_ollama_client()
        if not check_model_available(client): sys.exit(1)

    data = load_data(sample_size)
    total = len(data)
    bridge_cache = _load_bridge_cache()
    print(f"  [OK] {total} questions loaded")
    print(f"  [OK] Bridge cache: {len(bridge_cache)} entries")

    # إحصاء جودة الـ bridge
    clean_count = sum(
        1 for v in bridge_cache.values()
        if v and "Based on" not in v and "provided" not in v and len(v) <= 80
    )
    print(f"  [OK] Clean bridges: {clean_count}/{len(bridge_cache)} ({clean_count/len(bridge_cache)*100:.0f}%)")

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
    exp_fn = EXPERIMENT_MAP[strategy]

    print(f"\n--- Running ({total - len(existing)} remaining) ---\n")

    for i, record in enumerate(data):
        qid = record["id"]
        if qid in existing: continue

        drug_name = record.get("query_drug_name", "")
        ans_name  = record.get("answer_name", "")

        result = exp_fn(client, record, bridge_cache, top_k=top_k)

        prediction = result.get("prediction", "")
        is_correct = (
            bool(prediction)
            and prediction.upper() == record["answer"].upper()
            and result.get("success", False)
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
            **result
        }
        predictions.append(pred_record)
        existing[qid] = pred_record
        save_preds(predictions, pred_file)

        done = i + 1
        if verbose and (done % 10 == 0 or done == total):
            ans_now = sum(1 for p in predictions if p.get("success"))
            em = (correct_count / ans_now * 100) if ans_now else 0
            corrected_info = ""
            if strategy == "OV-EXP2" and result.get("corrected"):
                corrected_info = " [CORRECTED]"
            print(f"  [{done:>3}/{total}] {'✓' if is_correct else '✗'} {qid:<12} | "
                  f"EM: {em:.1f}%{corrected_info} | {(time.time()-pipeline_start)/60:.1f}m")

    metrics = calc_metrics(predictions)
    tt = time.time() - pipeline_start
    delta = metrics['strict_em'] - baseline_em

    # إحصاءات إضافية حسب التجربة
    extra_stats = ""
    if strategy in ("OV-EXP2", "OV-EXP4"):
        corrections = sum(1 for p in predictions if p.get("corrected"))
        consistent  = sum(1 for p in predictions if p.get("consistent"))
        label = "score>0" if strategy == "OV-EXP4" else "is_consistent()"
        extra_stats = (
            f"\n  Consistent ({label}): {consistent}/{metrics['total']}"
            f"\n  Corrections : {corrections}/{metrics['total']}"
        )
        if strategy == "OV-EXP4":
            zero_score = sum(
                1 for p in predictions
                if p.get("pred1_score", -1) == 0 and p.get("success", False)
            )
            extra_stats += f"\n  Score=0 cases (triggered): {zero_score}/{metrics['total']}"
    elif strategy == "OV-EXP3":
        refined = sum(1 for p in predictions if p.get("bridge_refined") and p["bridge_refined"] != p.get("bridge_entity"))
        extra_stats = f"\n  Bridges refined: {refined}/{metrics['total']}"
    elif strategy == "OV-EXP5":
        agreed = sum(1 for p in predictions if p.get("support_agreed"))
        extra_stats = f"\n  LLM agreed with Support-Top: {agreed}/{metrics['total']} ({agreed/metrics['total']*100:.1f}%)"

    print(f"\n{'='*65}")
    print(f"  RESULTS — Stage 4 ({strategy})")
    print(f"{'='*65}")
    print(f"  Strict EM% : {metrics['strict_em']}%  |  Correct: {metrics['correct']}/{metrics['total']}")
    print(f"  Baseline   : {baseline_em}%  |  Delta: {'+' if delta>=0 else ''}{delta:.2f}%{extra_stats}")
    print(f"  Time       : {tt/60:.1f}m")
    print(f"  Output     : {pred_file}")
    print(f"{'='*65}\n")

    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": {
                "stage": "Stage 4 - Ontology Verification",
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
    # │  Stage 4 — Ontology Verification & Correction           │
    # │  Baseline: B4-EXP4 (guided_retrieval) = 33.33%         │
    # ├─────────────────────────────────────────────────────────┤
    # │  OV-EXP1  Support Reranking   → 16.67% (0 LLM calls)  │
    # │  OV-EXP2  Guided+Consistency  → 20.47% (صارم جداً)    │
    # │  OV-EXP3  Bridge Refinement   → 27.78% (1 extra call)  │
    # │  OV-EXP4  Conservative Check  → 32.16% (score>0)       │
    # │  OV-EXP5  Support Hints       → ???    (الجديدة) ★     │
    # │           hints داخل الـ prompt — proactive بدل        │
    # │           post-hoc — الأكثر واعداً                      │
    # └─────────────────────────────────────────────────────────┘

    EXP_STRATEGY = "OV-EXP5"   # ← غيّري هنا: OV-EXP1|OV-EXP2|OV-EXP3|OV-EXP4|OV-EXP5
    EXP_TOP_K    = 3

    run_experiment(
        strategy=EXP_STRATEGY,
        top_k=EXP_TOP_K,
        sample_size=DIAG_SAMPLE_SIZE,
        verbose=True
    )