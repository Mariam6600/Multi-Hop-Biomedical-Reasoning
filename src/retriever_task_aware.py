"""
src/retriever_task_aware.py
============================
Biomedical Multi-Hop QA — Task-Aware Multi-Signal Retriever

المصدر:
  - NHSRAG   → Pairwise co-occurrence scoring
  - TreeQA   → Hypothesis support (direct statement detection)
  - Orekhovich → RRF-style signal fusion

نظام التقييم:
  score = 0.35 × cooccurrence  +  (Drug_A + Candidate في نفس الجملة)
          0.25 × interaction    +  (كلمات تفاعل دوائي موجودة)
          0.20 × hypothesis     +  (جملة مباشرة عن التفاعل)
          0.20 × bridge         ← (مصطلحات الآلية/الجسر)

فلاتر (بلا أرقام اعتباطية):
  ─────────────────────────────────────────────────────────
  Option 1 — Signal-Based :
    MUST: Drug_A موجود في الوثيقة
    PLUS: أي مرشح موجود  OR  أي مصطلح bridge موجود
    → بلا threshold اعتباطية

  Option 2 — Relative Threshold:
    threshold = best_score × ratio (عادةً ratio=0.5)

  Option 3 — Minimum Signals Count:
    الوثيقة تحتاج >= 2 إشارات من أصل 3

  B4-EXP10 — Task-Aware Hybrid (الاقتراح الكامل):
    score >= 0.3 (threshold الاعتباطية) + Score-Gap filter
  ─────────────────────────────────────────────────────────

Usage:
    from src.retriever_task_aware import retrieve_task_aware

    # B4-EXP10: Wikipedia + Internal + Task-Aware
    docs = retrieve_task_aware(
        supports_internal=supports,
        supports_wiki=wiki_chunks,
        drug_a=drug_name,
        bridge_terms=["inhibits MAO-A", "monoamine oxidase"],
        candidates=["DB04871", "DB04844"],
        candidate_names=["Tetrabenazine", "Selegiline"],
        filter_mode="signal",  # "signal" | "relative" | "count" | "threshold_0.3"
        max_k=5,
    )
"""

import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────
# SENTENCE SPLITTER
# ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> list:
    """يقسّم النص إلى جمل بطريقة بسيطة."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────
# INTERACTION VOCABULARY
# ──────────────────────────────────────────────────────

INTERACTION_TERMS = [
    'interact', 'interaction', 'inhibit', 'inhibitor',
    'contraindicated', 'avoid', 'concurrent', 'combination',
    'potentiate', 'enhance', 'reduce', 'affect', 'risk',
    'metabolism', 'metabolize', 'substrate', 'inducer', 'enzyme',
    'cytochrome', 'cyp', 'plasma', 'level', 'toxicity', 'adverse',
]


# ──────────────────────────────────────────────────────
# CORE SCORER — Supervisor's Revised Scoring System
# ──────────────────────────────────────────────────────

def compute_task_aware_score(
    doc_text: str,
    drug_a: str,
    bridge_terms: list,
    candidate_names: list,
) -> dict:
    """
    Multi-signal scoring (مُستوحى من NHSRAG + TreeQA + Orekhovich).

    Args:
        doc_text:        نص الوثيقة
        drug_a:          اسم الدواء في السؤال
        bridge_terms:    مصطلحات الآلية/الجسر (e.g., ["inhibits MAO-A"])
        candidate_names: أسماء المرشحين (e.g., ["Tetrabenazine", "Selegiline"])

    Returns:
        dict with {score, signals: {cooccurrence, interaction, hypothesis, bridge}, details}
    """
    text      = doc_text.lower()
    drug_low  = drug_a.lower().strip() if drug_a else ""
    sentences = _split_sentences(text)

    # ── SIGNAL 1: Drug_A Presence (شرط لازم) ──
    if drug_low and drug_low not in text:
        return {
            "score":        0.0,
            "has_drug_a":   False,
            "signals":      {"cooccurrence": 0.0, "interaction": 0.0, "hypothesis": 0.0, "bridge": 0.0},
            "details":      "Drug_A not found — score=0",
        }

    has_drug_a = bool(drug_low and drug_low in text)

    # ── SIGNAL 2: Co-occurrence Score (NHSRAG-inspired) ──
    cooccurrence_scores = []
    for cand in candidate_names:
        cand_low = cand.lower().strip() if cand else ""
        if not cand_low or cand_low not in text:
            cooccurrence_scores.append(0.0)
            continue
        # نفس الجملة = إشارة قوية جداً
        same_sentence = False
        for sent in sentences:
            if drug_low in sent and cand_low in sent:
                same_sentence = True
                break
        if same_sentence:
            cooccurrence_scores.append(1.0)
        else:
            cooccurrence_scores.append(0.5)   # نفس الوثيقة لكن جمل مختلفة

    cooccurrence_score = max(cooccurrence_scores) if cooccurrence_scores else 0.0

    # ── SIGNAL 3: Interaction Language ──
    interaction_hits = sum(1 for t in INTERACTION_TERMS if t in text)
    interaction_score = min(1.0, interaction_hits / 3)

    # ── SIGNAL 4: Bridge/Mechanism Terms ──
    bridge_hits = 0
    for bt in bridge_terms:
        bt_clean = bt.lower().strip() if bt else ""
        if bt_clean and bt_clean in text:
            bridge_hits += 1
        else:
            # جرّب كل كلمة من مصطلح الجسر
            for word in bt_clean.split():
                if len(word) > 4 and word in text:
                    bridge_hits += 0.5
                    break
    bridge_score = min(1.0, bridge_hits / 2)

    # ── SIGNAL 5: Hypothesis Support (TreeQA-inspired) ──
    hypothesis_score = 0.0
    for cand in candidate_names:
        cand_low = cand.lower().strip() if cand else ""
        if not cand_low:
            continue
        patterns = [
            f"{drug_low} interact {cand_low}",
            f"{drug_low} and {cand_low}",
            f"{cand_low} interact {drug_low}",
            f"{cand_low} and {drug_low}",
            f"{drug_low} with {cand_low}",
        ]
        # نستخدم نص مبسّط
        simple_text = text.replace("s ", " ").replace("ion ", " ").replace("ed ", " ")
        for pat in patterns:
            if pat in simple_text:
                hypothesis_score = 1.0
                break
        if hypothesis_score > 0:
            break

    # ── COMBINE (RRF-style weighting) ──
    final_score = (
        0.35 * cooccurrence_score  +
        0.25 * interaction_score   +
        0.20 * hypothesis_score    +
        0.20 * bridge_score
    )

    return {
        "score":      round(final_score, 4),
        "has_drug_a": has_drug_a,
        "signals": {
            "cooccurrence": round(cooccurrence_score, 4),
            "interaction":  round(interaction_score,  4),
            "hypothesis":   round(hypothesis_score,   4),
            "bridge":       round(bridge_score,       4),
        },
        "details": f"cooc={cooccurrence_score:.2f} int={interaction_score:.2f} hyp={hypothesis_score:.2f} br={bridge_score:.2f}",
    }


# ──────────────────────────────────────────────────────
# SIGNAL-BASED FILTER (Option 1 — موصى به من المشرفة)
# ──────────────────────────────────────────────────────

def signal_based_filter(
    docs: list,
    drug_a: str,
    candidate_names: list,
    bridge_terms: list,
) -> list:
    """
    Image 2 — "REVISED FILTERING: Signal-Based (No Magic Numbers)"

    MUST HAVE:   Drug_A in document
    PLUS ONE OF: Any candidate drug mentioned  OR  Bridge/mechanism term mentioned

    Logic: Drug_A AND (Candidate OR Bridge)

    Args:
        docs:            list of {text, score, rank, ...}
        drug_a:          اسم الدواء
        candidate_names: أسماء المرشحين
        bridge_terms:    مصطلحات الآلية

    Returns: filtered list with same structure + added 'has_candidate', 'has_bridge'
    """
    drug_low = drug_a.lower().strip() if drug_a else ""

    filtered = []
    for doc in docs:
        text = (doc.get("text", "") if isinstance(doc, dict) else str(doc)).lower()

        # ── MUST: Drug_A present ──
        if drug_low and drug_low not in text:
            continue

        # ── PLUS: Candidate OR Bridge ──
        has_candidate = any(
            c.lower().strip() in text for c in candidate_names if c
        )
        has_bridge = any(
            any(word in text for word in bt.lower().split() if len(word) > 4)
            for bt in bridge_terms if bt
        )

        if has_candidate or has_bridge:
            item = dict(doc) if isinstance(doc, dict) else {"text": str(doc), "score": 0.0, "rank": 0}
            item["has_candidate"] = has_candidate
            item["has_bridge"]    = has_bridge
            filtered.append(item)

    return filtered


# ──────────────────────────────────────────────────────
# RELATIVE THRESHOLD FILTER (Option 2)
# ──────────────────────────────────────────────────────

def relative_threshold_filter(scored_docs: list, ratio: float = 0.5) -> list:
    """
    Option 2: threshold = best_score × ratio
    يتكيّف مع توزيع الـ scores الفعلي.

    Args:
        scored_docs: list of dicts with 'score' key, مُرتّبة تنازلياً
        ratio:       نسبة الـ best score (0.5 = احتفظ بـ docs >= 50% من الأفضل)

    Returns: filtered list
    """
    if not scored_docs:
        return []
    best_score = scored_docs[0]["score"]
    if best_score == 0.0:
        return scored_docs[:3]  # fallback
    threshold = best_score * ratio
    return [d for d in scored_docs if d["score"] >= threshold]


# ──────────────────────────────────────────────────────
# SIGNALS COUNT FILTER (Option 3)
# ──────────────────────────────────────────────────────

def count_signals_filter(
    docs: list,
    drug_a: str,
    candidate_names: list,
    bridge_terms: list,
    min_signals: int = 2,
) -> list:
    """
    Option 3: الوثيقة تحتاج >= min_signals إشارات من أصل 3.
    بديهي: "الوثيقة يجب أن تُطابق >= 2 معايير"

    Args:
        min_signals: الحد الأدنى للإشارات (2 من أصل 3)
    """
    drug_low = drug_a.lower().strip() if drug_a else ""

    result = []
    for doc in docs:
        text = (doc.get("text", "") if isinstance(doc, dict) else str(doc)).lower()
        signals = 0
        if drug_low and drug_low in text:
            signals += 1
        if any(c.lower().strip() in text for c in candidate_names if c):
            signals += 1
        if any(any(w in text for w in bt.lower().split() if len(w) > 4) for bt in bridge_terms if bt):
            signals += 1
        if signals >= min_signals:
            item = dict(doc) if isinstance(doc, dict) else {"text": str(doc), "score": 0.0}
            item["signal_count"] = signals
            result.append(item)
    return result


# ──────────────────────────────────────────────────────
# SCORE GAP FILTER (من الكود الكامل المقترح)
# ──────────────────────────────────────────────────────

def score_gap_filter(sorted_docs: list, gap_ratio: float = 0.6, max_k: int = 5, min_k: int = 2) -> list:
    """
    الـ adaptive step: نتوقف عند انقطاع الـ score بشكل كبير.

    إذا score[i] < score[i-1] × gap_ratio → نتوقف هنا.
    Safety cap: max_k وثائق.

    ★ FIX: min_k=2 — multi-hop يحتاج وثيقتين على الأقل دائماً
    بدون هذا الحد، الفلتر يمكن أن يرجع وثيقة واحدة فقط (k=1)
    مما يمنع الـ LLM من رؤية مسار الاستدلال الكامل.

    Args:
        sorted_docs: list of dicts with 'score', مُرتّبة تنازلياً
        gap_ratio:   نسبة الانقطاع (0.6 = نتوقف إذا انخفض الـ score > 40%)
        max_k:       الحد الأقصى للعودة
        min_k:       الحد الأدنى للعودة (2 = multi-hop requirement)

    Returns: selected docs
    """
    if not sorted_docs:
        return []

    selected = [sorted_docs[0]]
    for i in range(1, len(sorted_docs)):
        prev_score = sorted_docs[i - 1].get("boosted_score", sorted_docs[i - 1]["score"])
        curr_score = sorted_docs[i].get("boosted_score", sorted_docs[i]["score"])
        if prev_score > 0 and curr_score < prev_score * gap_ratio:
            break
        selected.append(sorted_docs[i])
        if len(selected) >= max_k:
            break

    # ★ FIX: Enforce MIN_K=2 (multi-hop needs at least 2 docs)
    if len(selected) < min_k and len(sorted_docs) >= min_k:
        selected = sorted_docs[:min_k]

    return selected


# ──────────────────────────────────────────────────────
# MAIN FUNCTION — retrieve_task_aware
# (B4-EXP10 Complete Pipeline)
# ──────────────────────────────────────────────────────

def retrieve_task_aware(
    supports_internal: list,
    supports_wiki: list,
    drug_a: str,
    bridge_terms: list,
    candidates: list,
    candidate_names: list,
    filter_mode: str = "signal",
    threshold: float = 0.3,
    relative_ratio: float = 0.5,
    min_signals: int = 2,
    gap_ratio: float = 0.6,
    max_k: int = 5,
    fetch_k_internal: int = 10,
    fetch_k_wiki: int = 3,
) -> list:
    """
    B4-EXP10: Wikipedia + Task-Aware Adaptive Retrieval.

    الخطوات (من صورة المشرفة):
      1. Retrieve: internal top fetch_k_internal (Hybrid Bridge-Boosted)
                 + wiki top fetch_k_wiki
      2. Score ALL docs with task-aware scoring
      3. Filter by filter_mode
      4. Score-gap filter → adaptive selection
      5. Return final docs (max max_k)

    Args:
        supports_internal: internal supports من medhop.json
        supports_wiki:     chunks من ويكيبيديا (قائمة strings)
        drug_a:            اسم الدواء
        bridge_terms:      مصطلحات الآلية/الجسر
        candidates:        DrugBank IDs للمرشحين
        candidate_names:   أسماء المرشحين (للـ scoring)
        filter_mode:       "signal" | "relative" | "count" | "threshold"
        threshold:         الحد الأدنى للـ score (للـ threshold mode فقط)
        relative_ratio:    نسبة relative threshold
        min_signals:       الحد الأدنى للإشارات (للـ count mode)
        gap_ratio:         نسبة الانقطاع للـ score gap filter
        max_k:             الحد الأقصى للوثائق النهائية
        fetch_k_internal:  عدد الوثائق الداخلية قبل الـ scoring
        fetch_k_wiki:      عدد chunks ويكيبيديا

    Returns:
        list of dicts: {text, score, rank, source, task_score, signals}
    """
    from src.retriever_hybrid_scored import retrieve_hybrid_scored
    from src.query_expander import expand_query
    from src.query_expander_structured import get_weighted_terms

    # ── STEP 1A: Internal retrieval (Hybrid Bridge-Boosted) ──
    exp_result     = expand_query(drug_a)
    flat_terms     = exp_result.get("terms", []) if exp_result.get("success") else []
    weighted_terms = get_weighted_terms(drug_a) or []

    if bridge_terms:
        bridge_wt = [{"term": bt, "weight": 5.0, "category": "mechanism"} for bt in bridge_terms if bt]
        weighted_terms = bridge_wt + weighted_terms

    # نجلب fetch_k_internal وثيقة بالـ hybrid scorer
    internal_retrieved = retrieve_hybrid_scored(
        query=drug_a,
        supports=supports_internal,
        drug_name=drug_a,
        flat_terms=flat_terms,
        weighted_terms=weighted_terms,
        top_k=fetch_k_internal,
    )

    # ── STEP 1B: Wikipedia docs (أخذ أول fetch_k_wiki) ──
    wiki_retrieved = []
    for chunk in supports_wiki[:fetch_k_wiki]:
        wiki_retrieved.append({
            "text":         chunk,
            "score":        0.5,   # placeholder score
            "rank":         len(wiki_retrieved) + 1,
            "medcpt_score": 0.0,
            "term_score":   0.0,
            "source":       "wikipedia",
        })

    # ── STEP 2: Score ALL docs with task-aware scoring ──
    all_docs_raw = (
        [{"source": "internal", **d} for d in internal_retrieved] +
        wiki_retrieved
    )

    scored_docs = []
    for doc in all_docs_raw:
        text = doc.get("text", "")
        if not text:
            continue
        ts = compute_task_aware_score(
            doc_text=text,
            drug_a=drug_a,
            bridge_terms=bridge_terms,
            candidate_names=candidate_names,
        )
        doc["task_score"]  = ts["score"]
        doc["has_drug_a"]  = ts["has_drug_a"]
        doc["signals"]     = ts["signals"]
        doc["score"]       = ts["score"]   # ← override للـ sorting
        scored_docs.append(doc)

    # مرتّبة تنازلياً
    scored_docs.sort(key=lambda d: d["score"], reverse=True)

    # ── STEP 3: Filter ──
    if filter_mode == "signal":
        filtered = signal_based_filter(scored_docs, drug_a, candidate_names, bridge_terms)
        if not filtered:
            filtered = scored_docs[:3]   # Fallback
    elif filter_mode == "relative":
        filtered = relative_threshold_filter(scored_docs, ratio=relative_ratio)
        if not filtered:
            filtered = scored_docs[:3]
    elif filter_mode == "count":
        filtered = count_signals_filter(scored_docs, drug_a, candidate_names, bridge_terms, min_signals)
        if not filtered:
            filtered = scored_docs[:3]
    else:  # "threshold" (arbitrary 0.3 كما اقترحت المشرفة)
        filtered = [d for d in scored_docs if d["score"] >= threshold]
        if not filtered:
            filtered = scored_docs[:3]   # Fallback

    # ── STEP 4: Boost filtered docs ──
    # نضيف boost على الـ task_score مثل الكود المقترح
    for doc in filtered:
        boost = 0.0
        has_cand = doc.get("has_candidate", any(
            c.lower() in doc.get("text", "").lower() for c in candidate_names if c
        ))
        has_br = doc.get("has_bridge", False)
        if has_cand:
            boost += 0.3
        if has_br:
            boost += 0.2
        doc["score"] = round(doc["task_score"] + boost, 4)

    filtered.sort(key=lambda d: d["score"], reverse=True)

    # ── STEP 5: Score-gap filter → Adaptive selection ──
    selected = score_gap_filter(filtered, gap_ratio=gap_ratio, max_k=max_k)

    # ── Re-rank (rank 1..N) ──
    for idx, doc in enumerate(selected, 1):
        doc["rank"] = idx

    return selected


# ──────────────────────────────────────────────────────
# ADAPTIVE K — Signal-Based (تطبيق على B4-EXP4 مباشرة)
# ──────────────────────────────────────────────────────

def retrieve_adaptive_signal(
    query: str,
    supports: list,
    drug_a: str,
    bridge_terms: list,
    candidate_names: list,
    flat_terms: list = None,
    weighted_terms: list = None,
    fetch_k: int = 10,
    max_k: int = 5,
    gap_ratio: float = 0.6,
    min_k: int = 2,
) -> tuple:
    """
    Adaptive K بـ signal-based filter مباشرةً على B4-EXP4.

    الفرق عن B4-EXP4:
      - نجلب fetch_k=10 بالـ hybrid_scored
      - نفلتر بـ Drug_A + (Candidate OR Bridge)
      - نُضيف boost ونطبّق score_gap filter
      - k النهائي يتحدد تلقائياً (adaptive)

    ★ FIX: min_k يُفرض في 3 مستويات لضمان عدم ظهور k < min_k:
      1. بعد signal_based_filter → إذا النتيجة < min_k نستخدم fallback
      2. داخل score_gap_filter → min_k parameter
      3. فحص نهائي → if len(selected) < min_k

    Returns:
        (selected_docs, k_used, filter_stats)
    """
    from src.retriever_hybrid_scored import retrieve_hybrid_scored

    all_retrieved = retrieve_hybrid_scored(
        query=query,
        supports=supports,
        drug_name=drug_a,
        flat_terms=flat_terms or [],
        weighted_terms=weighted_terms or [],
        top_k=fetch_k,
    )

    if not all_retrieved:
        return [], 0, {}

    # Signal-based filter
    filtered = signal_based_filter(all_retrieved, drug_a, candidate_names, bridge_terms)

    # ★ BUG FIX: تغيير الشرط من "if not filtered" إلى "if len(filtered) < min_k"
    # السبب: signal_based_filter قد يُرجع وثيقة واحدة فقط.
    # الشرط القديم "if not filtered" كان يتحقق فقط من الحالة الفارغة (0 وثائق),
    # لكن حالة وثيقة واحدة كانت تمر بدون fallback → k=1.
    # الحل: إذا عدد الوثائق المُفلترة < min_k (2)، نلجأ للـ fallback.
    if len(filtered) < min_k:
        if len(filtered) == 0:
            # الحالة القديمة: لا شيء مُفلتر → جرب bridge filter فقط
            filtered = [
                d for d in all_retrieved
                if any(any(w in d["text"].lower() for w in bt.lower().split() if len(w) > 4)
                       for bt in bridge_terms if bt)
            ]
            if len(filtered) < min_k:
                filtered = all_retrieved[:min_k]   # Hard fallback — نأخذ min_k على الأقل
        else:
            # ★ الحالة الجديدة: filtered فيها وثيقة/وثائق لكن أقل من min_k
            # نُكمّل من all_retrieved بإضافة الوثائق التالية التي ليست في filtered
            filtered_ids = set(id(d) for d in filtered)
            remaining = [d for d in all_retrieved if id(d) not in filtered_ids]
            needed = min_k - len(filtered)
            filtered = filtered + remaining[:needed]

    # Boost
    for doc in filtered:
        boost = 0.3 if doc.get("has_candidate", False) else 0.0
        boost += 0.2 if doc.get("has_bridge",    False) else 0.0
        doc["boosted_score"] = round(doc["score"] + boost, 4)

    filtered.sort(key=lambda d: d.get("boosted_score", d["score"]), reverse=True)

    # Score gap filter (adaptive k) — with min_k enforcement
    selected = score_gap_filter(filtered, gap_ratio=gap_ratio, max_k=max_k, min_k=min_k)

    # ★ FIX: Final safety net — ensure k >= min_k regardless of all filters
    if len(selected) < min_k and len(filtered) >= min_k:
        selected = filtered[:min_k]
    elif len(selected) < min_k:
        # Even filtered < min_k (edge case) — return whatever we have
        pass

    # ★ FIX: Round k to nearest allowed value {2, 3, 5} if max_k=5
    # score_gap_filter قد يُرجع k=4 (إذا gap_ratio لم يقطع عند 3)
    # نُقرّب لـ {2, 3, 5} للاحتفاظ بدلالة التوزيع
    allowed_ks = [2, 3, 5] if max_k == 5 else list(range(min_k, max_k + 1))
    if len(selected) not in allowed_ks:
        # أقرب قيمة مسموحة
        closest = min(allowed_ks, key=lambda x: abs(x - len(selected)))
        if closest > len(selected):
            selected = filtered[:closest]  # لا نُقلّل تحت min_k
        else:
            selected = selected[:closest]  # نأخذ الأعلى scored

    for idx, doc in enumerate(selected, 1):
        doc["rank"] = idx

    filter_stats = {
        "fetched":   len(all_retrieved),
        "filtered":  len(filtered),
        "selected":  len(selected),
        "k_used":    len(selected),
    }

    return selected, len(selected), filter_stats


if __name__ == "__main__":
    # اختبار سريع
    sample_docs = [
        "Moclobemide is a reversible inhibitor of MAO-A and interacts with Tetrabenazine.",
        "Selegiline is a selective MAO-B inhibitor used in Parkinson's disease.",
        "This document about biology doesn't mention any drugs.",
        "Moclobemide and serotonin syndrome risk when combined with other drugs.",
    ]
    result = retrieve_task_aware(
        supports_internal=sample_docs,
        supports_wiki=["Moclobemide is a drug. It inhibits MAO-A."],
        drug_a="Moclobemide",
        bridge_terms=["inhibits MAO-A", "monoamine oxidase"],
        candidates=["DB04844", "DB04871"],
        candidate_names=["Tetrabenazine", "Selegiline"],
        filter_mode="signal",
        max_k=5,
    )
    print(f"Retrieved {len(result)} docs:")
    for d in result:
        print(f"  [{d['rank']}] score={d['score']:.3f} ({d.get('source','?')}) "
              f"signals={d.get('signals',{})} | {d['text'][:80]}")
