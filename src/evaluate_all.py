"""
src/evaluate_all.py
====================
يحسب المعايير الصحيحة لنظامنا (BioCreative IX MedHop Track):

المعيار 1: Strict EM% (Exact Match على كامل الداتا)
  البسط  = عدد الإجابات الصحيحة (تطابق DrugBank ID)
  المقام = إجمالي الأسئلة (بما فيها الفاشلة)
  → هذا هو المعيار الرسمي المستخدم في جدول BioCreative IX

المعيار 2: Lenient EM% (Exact Match على الأسئلة المجابة فقط)
  البسط  = نفس البسط
  المقام = الأسئلة التي أجاب عليها النموذج فعلاً (success=True)
  → يقيس جودة النموذج بمعزل عن مشاكل الاتصال/الانتهاء

المعيار 3: Concept-level Accuracy (متى يكون ممكناً)
  يُقيس ما إذا كان النموذج يُنتج الدواء الصحيح مفهومياً،
  حتى لو كتب اسمه بدلاً من الـ ID.

  ─── لماذا نحتاج Concept-level في بعض الأنظمة؟ ───
  الأنظمة التي تُنتج أسماءً نصية (مثل "Aspirin") تحتاج Concept-level
  لأن "Aspirin" و"Acetylsalicylic acid" هما نفس الدواء.
  
  ─── هل نظامنا يحتاجه؟ ───
  نظامنا يطلب من النموذج إنتاج DrugBank ID مباشرة (مثل "DB00065").
  إذا أنتج النموذج "DB00065" → Strict EM = 1 (صحيح).
  إذا أنتج النموذج "Aspirin"  → Strict EM = 0، لكن Concept يمكن تطبيقه.
  
  ─── كيف نحسبه؟ ───
  باستخدام DrugBank vocabulary CSV:
    كل ID له اسم واحد رسمي (Common name) وقد يكون له synonyms.
    نتحقق: هل الاسم الذي أنتجه النموذج يُرجع لنفس الـ ID كالإجابة الصحيحة؟
  
  ملاحظة مهمة من DrugBank:
    "كل دواء له ID واحد فريد — لا يوجد دوائان مختلفان بنفس الـ ID"
    "كل Synonyms لنفس الدواء تحمل نفس الـ ID"
    → إذاً: Concept accuracy = نطابق الـ names ضد DrugBank vocab لنعود للـ ID

  ─── متى يُطبَّق في ملفات نتائجنا؟ ───
  يُطبَّق فقط عندما prediction لا تحتوي "DB" pattern
  (أي عندما النموذج كتب اسماً بدلاً من ID).

ملاحظة عن معيار NM (Name Match):
  NM هو "اسم الدواء الصحيح موجود في رد النموذج الخام" — ليس معياراً بحثياً.
  لا يوجد في الدراسات المرجعية معيار بهذا الاسم.
  يُستخدم فقط كـ diagnostic indicator داخلي للتشخيص.
  لا يُستشهد به في الأبحاث.

Error Analysis:
  يُصدر تقريراً تفصيلياً عن أنواع الأخطاء:
  1. Retrieval failure: الإجابة الصحيحة غير موجودة في الوثائق المسترجعة
  2. Extraction error: النموذج حصل على المعلومات لكن أخطأ في استخراج الـ ID
  3. Reasoning error: النموذج استرجع الوثائق الصحيحة لكن استنتج خطأ
  4. Format error: النموذج أنتج تنسيقاً خاطئاً (اسم بدلاً من ID)
"""

import json
import os
import sys
import csv
import re
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR, MEDHOP_FILE, DRUGBANK_VOCAB


# ─────────────────────────────────────────────
# LOAD DRUGBANK VOCAB (للـ Concept-level)
# ─────────────────────────────────────────────

def load_drugbank_vocab_for_eval() -> dict:
    """
    يُحمّل DrugBank vocabulary لاستخدامه في الـ Concept-level evaluation.
    يُعيد: { "aspirin": "DB00945", "acetylsalicylic acid": "DB00945", ... }
    (key = lowercase name, value = DrugBank ID)
    """
    name_to_id = {}

    if not os.path.exists(DRUGBANK_VOCAB):
        return name_to_id

    try:
        with open(DRUGBANK_VOCAB, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                drug_id   = row.get("DrugBank ID", "").strip()
                drug_name = row.get("Common name", "").strip()
                synonyms  = row.get("Synonyms", "").strip()

                if drug_id and drug_name:
                    name_to_id[drug_name.lower()] = drug_id

                # إضافة كل synonyms إن وُجدت في الـ CSV
                if drug_id and synonyms:
                    for syn in synonyms.split("|"):
                        syn = syn.strip()
                        if syn:
                            name_to_id[syn.lower()] = drug_id
    except Exception as e:
        print(f"  [WARN] Could not load DrugBank vocab for Concept-level eval: {e}")

    return name_to_id


_DRUGBANK_NAME_TO_ID = None   # lazy load


def get_concept_id(prediction: str, name_to_id: dict) -> str:
    """
    يحاول تحويل نص (اسم دواء) إلى DrugBank ID.
    يُعيد الـ ID إذا وُجد في الـ vocab، وإلا يُعيد سلسلة فارغة.
    """
    if not prediction:
        return ""
    # إذا كان الـ prediction هو DB ID بالفعل → يُعيده مباشرة
    if re.match(r"DB\d{5}", prediction.strip(), re.IGNORECASE):
        return prediction.strip().upper()
    # البحث في الـ vocab
    pred_lower = prediction.lower().strip()
    return name_to_id.get(pred_lower, "")


# ─────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds: list, name_to_id: dict = None) -> dict:
    """
    يحسب المعايير:

    Strict EM%   = correct_strict / total        ← المعيار الرسمي
    Lenient EM%  = correct_strict / answered     ← جودة النموذج

    Concept-level Accuracy (تشخيصي فقط):
      يُطبَّق فقط عندما النموذج يكتب اسم دواء بدلاً من DB ID.
      مثال: النموذج أنتج "Acetaminophen" بدلاً من "DB00328"
             → Strict EM = 0
             → Concept يبحث في DrugBank vocab: "acetaminophen" → "DB00328" = الإجابة الصحيحة
             → Concept = 1

      ─── ملاحظة مهمة ───
      نظامنا يطلب من النموذج إنتاج DB ID مباشرة.
      لذلك Concept ≈ Strict EM في معظم الحالات.
      الفرق (concept_extra) يُظهر فقط حالات Format Error
      التي كانت صحيحة مفهومياً لكن بتنسيق خاطئ.
      هذا مؤشر تشخيصي وليس معيار تقييم مستقل.
    """
    total    = len(preds)
    failed   = sum(1 for p in preds if not p.get("success", False))
    answered = total - failed

    correct_strict  = 0
    correct_concept = 0
    concept_extra   = 0  # حالات Concept صحيح لكن Strict خطأ

    for p in preds:
        if not p.get("success", False):
            continue
        pred = p.get("prediction", "").strip()
        ans  = p.get("answer", "").strip()

        if not pred:
            continue

        is_strict_correct = (pred.upper() == ans.upper())

        # Strict EM: تطابق حرفي للـ ID
        if is_strict_correct:
            correct_strict += 1
            correct_concept += 1  # strict correct → concept correct بالتبعية

        # Concept-level: فقط للحالات التي Strict فشل فيها
        # نحاول تحويل اسم الدواء إلى DB ID عبر DrugBank vocab
        elif name_to_id:
            concept_id = get_concept_id(pred, name_to_id)
            if concept_id and concept_id.upper() == ans.upper():
                correct_concept += 1
                concept_extra += 1  # هذه حالة Format Error صحيحة مفهومياً

    strict_em   = round(correct_strict  / total    * 100, 2) if total    > 0 else 0.0
    lenient_em  = round(correct_strict  / answered * 100, 2) if answered > 0 else 0.0
    concept_acc = round(correct_concept / total    * 100, 2) if (total > 0 and name_to_id) else None

    result = {
        "total":          total,
        "answered":       answered,
        "failed":         failed,
        "correct":        correct_strict,
        "strict_em":      strict_em,
        "lenient_em":     lenient_em,
    }
    if concept_acc is not None:
        result["concept_acc"]     = concept_acc
        result["concept_correct"] = correct_concept
        result["concept_extra"]   = concept_extra  # عدد الحالات الإضافية فوق Strict EM

    return result


# ─────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────

def analyze_errors(preds: list, name_to_id: dict = None,
                   medhop_candidates: dict = None) -> dict:
    """
    تحليل مفصّل لأنواع الأخطاء.

    إصلاح: كثير من ملفات النتائج لا تحفظ حقل candidates مباشرة.
    نُستخدم medhop_candidates (من medhop.json) كـ fallback لضمان دقة
    تصنيف hallucination vs wrong_candidate.

    أنواع الأخطاء:
    1. format_error:    النموذج أنتج اسماً بدلاً من DrugBank ID
    2. wrong_candidate: النموذج اختار مرشحاً خاطئاً من القائمة
    3. hallucination:   النموذج أنتج ID غير موجود في قائمة المرشحين
    4. empty_output:    النموذج لم ينتج شيئاً
    5. api_failure:     فشل الاتصال بالنموذج
    """
    analysis = {
        "total":          len(preds),
        "correct":        0,
        "errors": {
            "format_error":    {"count": 0, "examples": []},
            "wrong_candidate": {"count": 0, "examples": []},
            "hallucination":   {"count": 0, "examples": []},
            "empty_output":    {"count": 0, "examples": []},
            "api_failure":     {"count": 0, "examples": []},
        },
        "wrong_predictions_freq": Counter(),   # أكثر الإجابات الخاطئة تكراراً
        "correct_answers_freq":   Counter(),   # توزيع الإجابات الصحيحة
    }

    for p in preds:
        answer    = p.get("answer", "").strip().upper()
        pred      = p.get("prediction", "").strip()
        pred_up   = pred.upper()
        success   = p.get("success", False)
        qid       = p.get("question_id", "")

        # candidates: من السجل مباشرة، أو من medhop.json كـ fallback
        raw_cands = p.get("candidates") or (medhop_candidates or {}).get(qid, [])
        candidates = [c.upper() for c in raw_cands]

        # 1. API failure
        if not success:
            analysis["errors"]["api_failure"]["count"] += 1
            if len(analysis["errors"]["api_failure"]["examples"]) < 3:
                analysis["errors"]["api_failure"]["examples"].append({
                    "id":    p.get("question_id", "?"),
                    "drug":  p.get("query_drug_name", "?"),
                    "error": p.get("error", "unknown"),
                })
            continue

        # 2. Correct
        if pred_up == answer:
            analysis["correct"] += 1
            analysis["correct_answers_freq"][answer] += 1
            continue

        # 3. Empty output
        if not pred:
            analysis["errors"]["empty_output"]["count"] += 1
            if len(analysis["errors"]["empty_output"]["examples"]) < 3:
                analysis["errors"]["empty_output"]["examples"].append({
                    "id":   p.get("question_id", "?"),
                    "drug": p.get("query_drug_name", "?"),
                    "ans":  answer,
                })
            continue

        # 4. Format error: لا يبدو DrugBank ID
        is_db_format = bool(re.match(r"DB\d{5}", pred, re.IGNORECASE))
        if not is_db_format:
            analysis["errors"]["format_error"]["count"] += 1
            if len(analysis["errors"]["format_error"]["examples"]) < 5:
                analysis["errors"]["format_error"]["examples"].append({
                    "id":         p.get("question_id", "?"),
                    "drug":       p.get("query_drug_name", "?"),
                    "prediction": pred[:50],
                    "expected":   answer,
                })
            analysis["wrong_predictions_freq"][pred[:30]] += 1
            continue

        # 5. Hallucination: ID صحيح الشكل لكن غير في المرشحين
        if candidates and pred_up not in candidates:
            analysis["errors"]["hallucination"]["count"] += 1
            if len(analysis["errors"]["hallucination"]["examples"]) < 5:
                analysis["errors"]["hallucination"]["examples"].append({
                    "id":         p.get("question_id", "?"),
                    "drug":       p.get("query_drug_name", "?"),
                    "prediction": pred,
                    "expected":   answer,
                    "n_cands":    len(candidates),
                })
            analysis["wrong_predictions_freq"][pred_up] += 1
            continue

        # 6. Wrong candidate: اختار مرشحاً خاطئاً (الخطأ الأكثر شيوعاً)
        analysis["errors"]["wrong_candidate"]["count"] += 1
        if len(analysis["errors"]["wrong_candidate"]["examples"]) < 10:
            analysis["errors"]["wrong_candidate"]["examples"].append({
                "id":         p.get("question_id", "?"),
                "drug":       p.get("query_drug_name", "?"),
                "prediction": pred,
                "expected":   answer,
                "answer_name": p.get("answer_name", "?"),
            })
        analysis["wrong_predictions_freq"][pred_up] += 1

    return analysis


def format_error_report(analysis: dict, exp_name: str = "") -> str:
    """يُنسّق تقرير Error Analysis كنص قابل للطباعة والحفظ."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  ERROR ANALYSIS REPORT — {exp_name}")
    lines.append("=" * 70)

    total   = analysis["total"]
    correct = analysis["correct"]
    errors  = analysis["errors"]

    lines.append(f"  Total questions : {total}")
    lines.append(f"  Correct (EM=1)  : {correct} ({correct/total*100:.1f}%)")
    lines.append(f"  Wrong/Failed    : {total - correct} ({(total-correct)/total*100:.1f}%)")
    lines.append("")
    lines.append("  ── Error Breakdown ──────────────────────────────────")

    error_types = [
        ("wrong_candidate", "Wrong Candidate (chose wrong from list)"),
        ("format_error",    "Format Error    (name instead of DB ID)"),
        ("hallucination",   "Hallucination   (DB ID not in candidates)"),
        ("empty_output",    "Empty Output    (no answer produced)"),
        ("api_failure",     "API/Model Failure"),
    ]

    for key, label in error_types:
        cnt = errors[key]["count"]
        pct = cnt / total * 100 if total > 0 else 0
        lines.append(f"  {label:<50}: {cnt:>4}  ({pct:.1f}%)")

    # أمثلة على Wrong Candidate
    wc_examples = errors["wrong_candidate"]["examples"]
    if wc_examples:
        lines.append("")
        lines.append("  ── Wrong Candidate Examples (up to 5) ───────────────")
        for ex in wc_examples[:5]:
            lines.append(
                f"  Q: {ex.get('id','?'):<15} Drug: {ex.get('drug','?'):<20} "
                f"Pred: {ex.get('prediction','?'):<10} "
                f"Expected: {ex.get('expected','?'):<10} ({ex.get('answer_name','?')})"
            )

    # أمثلة على Format Error
    fe_examples = errors["format_error"]["examples"]
    if fe_examples:
        lines.append("")
        lines.append("  ── Format Error Examples (up to 5) ──────────────────")
        for ex in fe_examples[:5]:
            lines.append(
                f"  Q: {ex.get('id','?'):<15} Drug: {ex.get('drug','?'):<20} "
                f"Pred: '{ex.get('prediction','?')[:30]}'  Expected: {ex.get('expected','?')}"
            )

    # أكثر الإجابات الخاطئة تكراراً
    wf = analysis.get("wrong_predictions_freq", Counter())
    if wf:
        lines.append("")
        lines.append("  ── Most Common Wrong Predictions (top 5) ────────────")
        for pred, cnt in wf.most_common(5):
            lines.append(f"  '{pred}' — {cnt} times")

    lines.append("")
    lines.append("  ── Key Observations ──────────────────────────────────")
    wc_pct = errors["wrong_candidate"]["count"] / total * 100 if total > 0 else 0
    fe_pct = errors["format_error"]["count"]    / total * 100 if total > 0 else 0
    ha_pct = errors["hallucination"]["count"]   / total * 100 if total > 0 else 0

    if wc_pct > 40:
        lines.append("  -> Primary error: Wrong Candidate selection from list")
        lines.append("     Likely cause: Retrieved docs lack sufficient evidence (retrieval gap)")
    if fe_pct > 10:
        lines.append("  -> High Format Error rate — model outputs names instead of DB IDs")
        lines.append("     Fix: Reinforce DBxxxxx format instruction in prompt")
    if ha_pct > 5:
        lines.append("  -> Model generates IDs not in candidate list — extraction issue")

    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────

def extract_meta(preds: list, fname: str) -> dict:
    if not preds:
        return {}
    p0 = preds[0]
    model     = (p0.get("model") or p0.get("api_model") or p0.get("llm_model") or "unknown")
    retriever = (p0.get("retriever") or _guess_retriever(fname))
    prompt    = (p0.get("prompt_type") or _guess_prompt(fname))
    k         = p0.get("top_k", _guess_k(fname))
    # Flag oracle experiments: gold_chain retriever = oracle (cheats by giving docs with correct answer)
    is_oracle = "gold_chain" in str(retriever).lower() or "gold_chain" in fname.lower()
    is_ensemble = "ensemble" in fname.lower()
    return {
        "model":     str(model)[:35],
        "retriever": str(retriever),
        "prompt":    str(prompt),
        "k":         str(k),
        "is_oracle":   is_oracle,
        "is_ensemble": is_ensemble,
    }


def _guess_retriever(fname: str) -> str:
    f = fname.lower()
    # Oracle / special retrievers
    if "ensemble"        in f: return "Ensemble-Vote"
    if "gold_chain"      in f: return "gold_chain"
    if "query_drug"      in f: return "query_drug"
    if "weighted_struct" in f: return "weighted_struct"
    if "decomp"          in f: return "Decomp"
    if "combined"        in f: return "Combined"
    if "expanded"        in f: return "Expanded"
    if "semantic"        in f: return "MedCPT"
    if "bm25"            in f: return "BM25"
    if "hybrid"          in f: return "Hybrid"
    if "stage4"          in f: return "Bridge"
    # Phase/stage prefixes
    if f.startswith("phase1_") or f.startswith("phase2_"): return "Hybrid"
    if f.startswith("eb_"):    return "Bridge-Hybrid"   # Entity Bridging
    if f.startswith("ov_"):    return "Int.Supports"    # Ontology Verification
    if f.startswith("qd_"):    return "Hybrid"          # Query Decomposition
    if f.startswith("decomp_"): return "Decomp"
    if f.startswith("diagnosis_"): return "Expanded"
    # Baselines
    if "baseline1" in f: return "None"
    if "baseline2" in f: return "BM25"
    if "baseline3" in f: return "MedCPT"
    return "?"


def _guess_prompt(fname: str) -> str:
    f = fname.lower()
    if "fewshot"         in f: return "fewshot"
    if "direct"          in f: return "direct"
    if "ensemble"        in f: return "majority_vote"
    if "guided_retrieval"in f: return "guided"
    if "cot_enriched"    in f: return "cot_enriched"
    if "cand_filtered"   in f: return "cand_filtered"
    if "bridge_pivoted"  in f: return "bridge_pivot"
    if "forced_reasoning"in f: return "forced"
    if "enriched"        in f: return "enriched"
    if "reranked"        in f: return "reranked"
    if "dual_retrieval"  in f: return "dual_ret"
    # OV experiments
    if "ov-exp1"  in f: return "support_rerank"
    if "ov-exp2"  in f: return "consistency"
    if "ov-exp3"  in f: return "bridge_refine"
    if "ov-exp4"  in f: return "conservative"
    if "ov-exp5"  in f: return "supp_hints"
    # QD experiments
    if "qd-exp1"  in f: return "qd_hop2only"
    if "qd-exp2"  in f: return "qd_ircot"
    if "qd-exp3"  in f: return "qd_candfilter"
    if "qd-exp4"  in f: return "qd_guided"
    if "qd-exp5"  in f: return "qd_per_cand"
    if "decomp"          in f: return "decomp"
    if "cot"             in f: return "CoT"
    return "zero-shot"

def _guess_k(fname: str) -> str:
    m = re.search(r"_k(\d+)[_.]", fname)
    return m.group(1) if m else "?"


# ─────────────────────────────────────────────
# REFERENCE PAPERS
# ─────────────────────────────────────────────

PAPER_RESULTS = [
    ("DMIS Lab (Rank 1, BioCreative IX)",    87.3, "GPT-4o + MedCPT + Web Search"),
    ("UETQuintet (Rank 2, BioCreative IX)",  83.8, "GPT-4o-mini + Wikipedia"),
    ("NHSRAG (PreceptorAI)",                 73.4, "MedReason-8B + Wikipedia (spaCy NER)"),
    ("Fluxion (Advanced Prompting)",         68.1, "Gemini 2.0/2.5 Flash — No RAG"),
    ("CLaC (Agentic SmolAgents)",            67.6, "Qwen2.5-Coder-32B + Wikipedia + PubMed"),
    ("Orekhovich (local Ollama)",            43.9, "Llama3-Med42-8B + BM25S + MedEmbed"),
    ("lasigeBioTM",                          28.3, "Mistral-7B + Mondo Ontology"),
    ("DeepRAG (BioHop)",                     20.7, "DeepSeek R1 + Wikipedia + DPO"),
    ("CaresAI (fine-tuned LLaMA-3 8B)",     18.6, "LoRA fine-tuning on BioASQ/MedQuAD"),
]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 110)
    print("  EVALUATION — Strict EM% + Lenient EM% + Concept Accuracy + Error Analysis")
    print("=" * 110)

    # Load DrugBank vocab for Concept-level eval
    print("  Loading DrugBank vocabulary for Concept-level evaluation...")
    name_to_id = load_drugbank_vocab_for_eval()
    if name_to_id:
        print(f"  [OK] {len(name_to_id):,} drug name→ID mappings loaded")
    else:
        print("  [WARN] DrugBank vocab not found — Concept-level will be skipped")

    # ── Load medhop.json for candidates fallback ──────────────────────────────
    # كثير من ملفات النتائج لا تحفظ حقل candidates مباشرة
    # نُحمّل medhop.json كمرجع لمعرفة قائمة المرشحين لكل سؤال
    medhop_candidates = {}  # { question_id: [candidates...] }
    if os.path.exists(MEDHOP_FILE):
        try:
            with open(MEDHOP_FILE, encoding="utf-8") as f:
                medhop_data = json.load(f)
            for record in medhop_data:
                medhop_candidates[record["id"]] = record.get("candidates", [])
            print(f"  [OK] medhop.json loaded — {len(medhop_candidates)} question candidates")
        except Exception as e:
            print(f"  [WARN] Could not load medhop.json: {e}")
    else:
        print("  [WARN] medhop.json not found — hallucination detection may be imprecise")

    # Load all prediction files
    pred_files = sorted([
        f for f in os.listdir(OUTPUTS_DIR)
        if f.endswith("_predictions.json")
    ])

    rows = []
    all_error_reports = []

    for fname in pred_files:
        path = os.path.join(OUTPUTS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                preds = json.load(f)
        except Exception:
            continue
        if not preds:
            continue

        metrics = compute_metrics(preds, name_to_id if name_to_id else None)
        meta    = extract_meta(preds, fname)

        # ── Skip experiments that are too small to be meaningful ──────────────
        # Threshold: must have at least 50 questions (≥14% of full dataset)
        # This excludes 1-3 question tests, partial runs, and diagnostic probes
        # Full dataset = 342 questions; 50 is a reasonable minimum for any reported result
        if metrics["total"] < 50:
            continue

        # Skip incomplete runs (< 50% of full dataset) UNLESS they were intentional
        # 50-question runs are intentional API tests (baseline1_api, diagnosis_*)
        # Files named "diagnosis_" are always intentional small runs
        is_diagnosis = "diagnosis_" in fname
        # ★ FIX: أزلنا الشرط المُكرّر "metrics['total'] < 50"
        # الملفات بعدد < 50 سبق استثناؤها في السطر 557 أعلاه
        # هنا نستثني فقط الملفات الجزئية (50 <= total < 342) التي ليست تشخيصية
        is_partial   = metrics["total"] < 342
        if is_partial and not is_diagnosis:
            continue

        # Skip if no correct answers at all AND very few answered
        if metrics["correct"] == 0 and metrics["answered"] < 5:
            continue

        row = {
            "file":      fname,
            "model":     meta.get("model", "?"),
            "retriever": meta.get("retriever", "?"),
            "prompt":    meta.get("prompt", "?"),
            "k":         meta.get("k", "?"),
            "is_oracle":   meta.get("is_oracle",   False),
            "is_ensemble": meta.get("is_ensemble", False),
            **metrics,
        }
        rows.append(row)

        # Error analysis — تمرير medhop_candidates كـ fallback للـ candidates
        error_analysis = analyze_errors(
            preds,
            name_to_id if name_to_id else None,
            medhop_candidates
        )
        exp_short = fname.replace("_predictions.json", "")[:50]
        report    = format_error_report(error_analysis, exp_short)
        all_error_reports.append((fname, report))

    # Sort by Strict EM%
    rows.sort(key=lambda x: x["strict_em"], reverse=True)

    lines = []

    # ── Reference Papers ──
    lines.append("=" * 110)
    lines.append("  ── Reference Papers (BioCreative IX MedHop Track 2025) ──")
    lines.append(f"  {'System':<48} {'Strict EM%':>10}  Notes")
    lines.append("  " + "─" * 105)
    for name, em, notes in PAPER_RESULTS:
        lines.append(f"  {name:<48} {em:>9.1f}%  {notes}")

    # ── Our Experiments ──
    lines.append("")
    lines.append("  ── Our Experiments ──")
    has_concept = any("concept_acc" in r for r in rows)

    header = (
        f"  {'Strict EM%':>10}  {'Lenient EM%':>11}  "
        f"{'n':>5}  {'fail':>4}  "
        f"{'Model':<35}  {'Retriever':<12}  {'Prompt':<10}  {'K':>3}"
        f"  {'Note'}"
    )
    if has_concept:
        header += f"  {'Concept%':>9}"
    lines.append(header)
    lines.append("  " + "─" * 110)

    for r in rows:
        # Build note: oracle flag takes priority, then partial run flag
        if r.get("is_ensemble"):
            note = "[ENSEMBLE — majority vote of 3 best experiments — ★ BEST RESULT]"
        elif r.get("is_oracle"):
            note = "[ORACLE — upper bound only, not a production result]"
        elif r["total"] < 342:
            note = f"[{r['total']}q — partial/test run, NOT comparable to full 342q]"
        else:
            note = ""

        line = (
            f"  {r['strict_em']:>9.1f}%  {r['lenient_em']:>10.1f}%  "
            f"{r['answered']:>5}  {r['failed']:>4}  "
            f"{r['model']:<35}  {str(r['retriever'])[:12]:<12}  "
            f"{str(r['prompt'])[:10]:<10}  {r['k']:>3}"
            f"  {note}"
        )
        if has_concept and "concept_acc" in r:
            line += f"  {r['concept_acc']:>8.1f}%"
        lines.append(line)

    lines.append("")
    lines.append("=" * 110)

    if rows:
        # Best result: full 342q runs, excluding oracle experiments
        full_rows      = [r for r in rows if r["total"] == 342]
        oracle_rows    = [r for r in full_rows if r.get("is_oracle")]
        production_rows = [r for r in full_rows if not r.get("is_oracle")]
        partial_rows   = [r for r in rows if r["total"] < 342]

        if oracle_rows:
            best_oracle = max(oracle_rows, key=lambda x: x["strict_em"])
            lines.append(f"  Oracle upper bound (gold_chain):         {best_oracle['strict_em']:.1f}%"
                         f"  [ORACLE — not comparable to papers, included for reference only]")

        if production_rows:
            best_prod    = max(production_rows, key=lambda x: x["strict_em"])
            best_strict  = best_prod["strict_em"]
            best_lenient = max(r["lenient_em"] for r in production_rows)
            lines.append(f"  Best Strict EM%  — production (342q):    {best_strict:.1f}%"
                         f"  [{best_prod['model']} / {best_prod['retriever']} / {best_prod['prompt']}]")
            lines.append(f"  Best Lenient EM% — production (342q):    {best_lenient:.1f}%")
            lines.append(f"  Gap to oracle:                            "
                         f"{best_oracle['strict_em'] - best_strict:.1f}pp" if oracle_rows else "")

        if partial_rows:
            lines.append("")
            lines.append(f"  NOTE: {len(partial_rows)} partial/test run(s) shown above are NOT comparable to full 342q results.")
            lines.append(f"  A 100% on 3 questions ≠ 100% on 342 questions.")

    lines.append("")
    lines.append("  ── Metric Definitions ──────────────────────────────────────────────")
    lines.append("  Strict EM%  : correct / total        — official BioCreative IX metric")
    lines.append("  Lenient EM% : correct / answered     — excludes API/connection failures")
    lines.append("  Concept%    : Strict EM + Format Errors that were conceptually correct (diagnostic only)")
    lines.append("                In our system Concept ≈ Strict EM (model outputs DB IDs directly)")
    lines.append("  NM (Name Match): NOT a research metric — internal diagnostic indicator only")

    # Concept-level extra summary
    total_concept_extra = sum(r.get("concept_extra", 0) for r in rows)
    if total_concept_extra > 0:
        lines.append("")
        lines.append("  ── Concept-level additional info ──")
        lines.append(f"  Total cases where Concept > Strict EM: {total_concept_extra}")
        lines.append("  (Model wrote drug name instead of DB ID but was conceptually correct)")

    lines.append("=" * 110)

    table_text = "\n".join(lines)
    print(table_text)

    # ── Save main results ──
    txt_path  = os.path.join(OUTPUTS_DIR, "evaluation_results.txt")
    json_path = os.path.join(OUTPUTS_DIR, "evaluation_results.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_text + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "our_experiments":  rows,
            "paper_references": [
                {"system": n, "strict_em": e, "notes": nt}
                for n, e, nt in PAPER_RESULTS
            ],
        }, f, indent=2, ensure_ascii=False)

    # ── Save Error Analysis ──
    error_report_path = os.path.join(OUTPUTS_DIR, "error_analysis_all.txt")
    with open(error_report_path, "w", encoding="utf-8") as f:
        f.write("ERROR ANALYSIS — ALL EXPERIMENTS\n")
        f.write("=" * 70 + "\n\n")
        for fname, report in all_error_reports:
            f.write(f"\nFile: {fname}\n")
            f.write(report + "\n\n")

    print(f"\n  Saved → {txt_path}")
    print(f"  Saved → {json_path}")
    print(f"  Saved → {error_report_path}")
    print()

    # Print Error Analysis for best PRODUCTION run (non-oracle, full 342q)
    if rows:
        full_rows       = [r for r in rows if r["total"] == 342]
        production_rows = [r for r in full_rows if not r.get("is_oracle")]
        if production_rows:
            best_row  = production_rows[0]   # already sorted by strict_em
            best_file = os.path.join(OUTPUTS_DIR, best_row["file"])
            print(f"\n  Error Analysis for best production experiment: {best_row['file']}")
            try:
                with open(best_file, encoding="utf-8") as f:
                    best_preds = json.load(f)
                analysis = analyze_errors(best_preds, name_to_id if name_to_id else None, medhop_candidates)
                print(format_error_report(analysis, best_row["file"]))
            except Exception:
                pass


if __name__ == "__main__":
    main()