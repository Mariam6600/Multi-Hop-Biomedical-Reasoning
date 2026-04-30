"""
src/retriever_combined.py
==========================
Biomedical Multi-Hop QA — Combined Retriever (التجربة الجديدة المقترحة)

الفكرة:
  score = term_count_score + 2 × (query_drug_name in doc)

لماذا هذا أفضل من كل منهما منفصلاً؟
  - query_drug: 16.4% — لأن الدواء لا يظهر بالاسم في كل النصوص
  - expanded (term_count): 19.6% — لأنه يمسك المصطلحات البيولوجية الأشمل
  - الجمع: المصطلحات تمسك السياق البيولوجي + bonus للنصوص التي تذكر الدواء مباشرة

صيغة الـ score النهائية:
  score = Σ term_found_in_doc (كل مصطلح فريد = 1 نقطة)
        + 2.0 × (اسم الدواء في الوثيقة)    ← drug_name bonus (مثل retriever_expanded)
        + QUERY_DRUG_BONUS × (اسم الدواء في الوثيقة بشكل صريح)
          -- هذا هو الفرق الجوهري عن retriever_expanded --
          -- في retriever_expanded الـ bonus = 2.0 فقط --
          -- هنا QUERY_DRUG_BONUS = 2.0 إضافية = مجموع 4.0 --

ملاحظة مهمة:
  في retriever_expanded.py الـ drug_name bonus موجود بالفعل (score += 2.0)
  التجربة الجديدة تختبر إضافة bonus مضاعف (QUERY_DRUG_BONUS إضافي) بالضبط
  كما اقترح المشرف: score = term_count_score + 2 × (query_drug_name in doc)
  → أي: نُضاعف الأهمية النسبية لوجود اسم الدواء
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────
# الثوابت
# ─────────────────────────────────────────────

# Bonus إضافي لوجود الدواء (فوق الـ 2.0 الموجودة في retriever_expanded)
# المجموع = 2.0 (base drug bonus) + EXTRA_DRUG_BONUS
# النتيجة المقترحة: docs تحتوي الدواء تحصل على 4.0 بدلاً من 2.0
EXTRA_DRUG_BONUS = 2.0


def retrieve_combined(
    query: str,
    supports: list,
    drug_name: str,
    drug_id: str,
    expanded_terms: list,
    top_k: int = 5,
    extra_drug_bonus: float = EXTRA_DRUG_BONUS,
) -> list:
    """
    Combined retrieval: Term-Count + مضاعفة bonus دواء السؤال.

    هذا هو التعديل الجديد المقترح:
      قبل: score = term_count + 2.0 × drug_name_in_doc
      بعد: score = term_count + (2.0 + extra_drug_bonus) × drug_name_in_doc

    الهدف: تفضيل الوثائق التي تذكر دواء السؤال صراحةً،
            مع الاستفادة من المصطلحات البيولوجية المولّدة.

    Args:
        query:            نص السؤال الأصلي
        supports:         قائمة نصوص الوثائق الداعمة
        drug_name:        اسم الدواء في السؤال (مثلاً "Moclobemide")
        drug_id:          DrugBank ID للدواء في السؤال (مثلاً "DB01171")
        expanded_terms:   المصطلحات البيولوجية المولّدة بالـ LLM
        top_k:            عدد الوثائق المسترجعة
        extra_drug_bonus: الـ bonus الإضافي لوجود الدواء (افتراضي: 2.0)

    Returns:
        قائمة من dicts: [{text, score, rank, matched_terms, has_drug_name, has_drug_id}, ...]
    """
    if not supports:
        return []

    # تنظيف المصطلحات وإزالة التكرار
    terms_lower = list(dict.fromkeys(
        t.lower().strip() for t in expanded_terms if t and t.strip()
    ))

    drug_name_l = drug_name.lower().strip() if drug_name else ""
    drug_id_l   = drug_id.lower().strip()   if drug_id   else ""

    scored = []
    for doc_text in supports:
        if not doc_text or not doc_text.strip():
            continue

        doc_lower = doc_text.lower()

        # ── Component 1: Term count ──
        matched = [t for t in terms_lower if t in doc_lower]
        score = float(len(matched))

        # ── Component 2: Drug name bonus (base = 2.0 كما في retriever_expanded) ──
        has_name = bool(drug_name_l and drug_name_l in doc_lower)
        has_id   = bool(drug_id_l   and drug_id_l   in doc_lower)
        has_drug = has_name or has_id

        # Base bonus (كما في retriever_expanded)
        if has_drug:
            score += 2.0

        # ── Component 3: Extra drug bonus (الجديد) ──
        # هذا هو التعديل المقترح: مضاعفة الأهمية
        if has_drug:
            score += extra_drug_bonus

        scored.append({
            "text":          doc_text,
            "score":         score,
            "matched_terms": matched,
            "has_drug_name": has_name,
            "has_drug_id":   has_id,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    results = []
    for rank, item in enumerate(scored[:top_k], 1):
        results.append({
            "text":          item["text"],
            "score":         item["score"],
            "rank":          rank,
            "matched_terms": item["matched_terms"],
            "has_drug_name": item["has_drug_name"],
            "has_drug_id":   item["has_drug_id"],
        })

    return results