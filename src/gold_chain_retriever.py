"""
src/gold_chain_retriever.py
============================
يحتوي على نوعين من الـ retrieval:

──────────────────────────────────────────────────
retrieve_gold_chain()  ← "الغش" — للتحليل النظري فقط
──────────────────────────────────────────────────
  تصفية الوثائق بناءً على اسم/ID الدواء الصحيح (الإجابة).
  يتطلب معرفة الإجابة مسبقاً → لا يُستخدم في الإنتاج.
  الهدف: قياس الـ UPPER BOUND.
  النتائج: 76.9% عند k=3، 66.96% عند k=5.

──────────────────────────────────────────────────
retrieve_query_drug()  ← التجربة الجديدة — قابلة للاستخدام الحقيقي
──────────────────────────────────────────────────
  تصفية الوثائق بناءً على اسم/ID الدواء الموجود في السؤال (query drug).
  لا تحتاج معرفة الإجابة → يمكن استخدامها في الإنتاج.
  
  الفكرة: وثائق تذكر الدواء الأصلي هي الأكثر صلة بالسؤال.
  مثال:
    السؤال عن: Moclobemide (DB01171)
    → نُرجع الوثائق التي تذكر "Moclobemide" أو "DB01171"
    → هذه الوثائق ستحتوي على آلية عمله وتفاعلاته المحتملة
    → تُساعد النموذج على معرفة الدواء الذي يتفاعل معه

  الفرق عن retrieve_gold_chain:
    gold_chain:  يبحث عن اسم الإجابة (Tetrabenazine) → يحتاج معرفة الجواب
    query_drug:  يبحث عن اسم السؤال (Moclobemide)   → لا يحتاج معرفة الجواب

Usage في inference_pipeline5.py:
    EXP_RETRIEVER = "gold_chain"    ← التجربة النظرية (upper bound)
    EXP_RETRIEVER = "query_drug"    ← التجربة الجديدة الواقعية
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────
# الدالة 1: Gold Chain — تصفية بالإجابة الصحيحة (للتحليل النظري فقط)
# ─────────────────────────────────────────────

def retrieve_gold_chain(
    query: str,
    supports: list,
    drug_name: str,
    answer_id: str,
    answer_name: str = "",
    top_k: int = 5,
) -> list:
    """
    ترتّب الوثائق حسب احتوائها على اسم/ID الدواء الصحيح (الإجابة).
    تتطلب معرفة الإجابة → للتحليل النظري فقط.

    Scoring:
        3.0 = تحتوي على الاسم + ID معاً
        2.0 = تحتوي على الاسم فقط
        1.0 = تحتوي على الـ ID فقط
        0.0 = لا تحتوي على أي منهما
    """
    if not supports:
        return []

    answer_id_l   = answer_id.lower().strip()   if answer_id   else ""
    answer_name_l = answer_name.lower().strip() if answer_name else ""

    scored = []
    for doc in supports:
        if not doc or not doc.strip():
            continue
        doc_l = doc.lower()

        has_name = bool(answer_name_l and answer_name_l in doc_l)
        has_id   = bool(answer_id_l   and answer_id_l   in doc_l)

        if has_name and has_id:
            score = 3.0
        elif has_name:
            score = 2.0
        elif has_id:
            score = 1.0
        else:
            score = 0.0

        scored.append({"text": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "text":          item["text"],
            "score":         item["score"],
            "rank":          rank,
            "is_gold":       item["score"] > 0,
            "matched_terms": ["[GOLD_ANSWER]"] if item["score"] > 0 else [],
        }
        for rank, item in enumerate(scored[:top_k], 1)
    ]


# ─────────────────────────────────────────────
# الدالة 2: Query Drug — تصفية باسم الدواء في السؤال (واقعية)
# ─────────────────────────────────────────────

def retrieve_query_drug(
    query: str,
    supports: list,
    drug_name: str,
    drug_id: str = "",
    top_k: int = 5,
) -> list:
    """
    ترتّب الوثائق حسب احتوائها على اسم/ID الدواء الموجود في السؤال.
    لا تحتاج معرفة الإجابة → قابلة للاستخدام الحقيقي.

    Args:
        drug_name: اسم الدواء في السؤال (query drug) — مثلاً "Moclobemide"
        drug_id:   DrugBank ID للدواء في السؤال — مثلاً "DB01171"

    Scoring:
        3.0 = تحتوي على الاسم + ID معاً
        2.0 = تحتوي على الاسم فقط
        1.0 = تحتوي على الـ ID فقط
        0.0 = لا تحتوي على أي منهما
    """
    if not supports:
        return []

    drug_name_l = drug_name.lower().strip() if drug_name else ""
    drug_id_l   = drug_id.lower().strip()   if drug_id   else ""

    scored = []
    for doc in supports:
        if not doc or not doc.strip():
            continue
        doc_l = doc.lower()

        has_name = bool(drug_name_l and drug_name_l in doc_l)
        has_id   = bool(drug_id_l   and drug_id_l   in doc_l)

        if has_name and has_id:
            score = 3.0
        elif has_name:
            score = 2.0
        elif has_id:
            score = 1.0
        else:
            score = 0.0

        scored.append({"text": doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    return [
        {
            "text":          item["text"],
            "score":         item["score"],
            "rank":          rank,
            "is_gold":       item["score"] > 0,
            "matched_terms": ["[QUERY_DRUG]"] if item["score"] > 0 else [],
        }
        for rank, item in enumerate(scored[:top_k], 1)
    ]


# ─────────────────────────────────────────────
# Helper: قياس التغطية
# ─────────────────────────────────────────────

def check_gold_coverage(supports: list, answer_id: str, answer_name: str = "") -> dict:
    """كم وثيقة تحتوي على الإجابة الصحيحة — للتحليل."""
    answer_id_l   = answer_id.lower().strip()   if answer_id   else ""
    answer_name_l = answer_name.lower().strip() if answer_name else ""

    found_by_name = sum(1 for d in supports if answer_name_l and answer_name_l in d.lower())
    found_by_id   = sum(1 for d in supports if answer_id_l   and answer_id_l   in d.lower())

    return {
        "total_docs":    len(supports),
        "found_by_name": found_by_name,
        "found_by_id":   found_by_id,
        "any_found":     found_by_name > 0 or found_by_id > 0,
    }


def check_query_drug_coverage(supports: list, drug_name: str, drug_id: str = "") -> dict:
    """كم وثيقة تحتوي على الدواء في السؤال."""
    drug_name_l = drug_name.lower().strip() if drug_name else ""
    drug_id_l   = drug_id.lower().strip()   if drug_id   else ""

    found_by_name = sum(1 for d in supports if drug_name_l and drug_name_l in d.lower())
    found_by_id   = sum(1 for d in supports if drug_id_l   and drug_id_l   in d.lower())

    return {
        "total_docs":    len(supports),
        "found_by_name": found_by_name,
        "found_by_id":   found_by_id,
        "any_found":     found_by_name > 0 or found_by_id > 0,
    }


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import MEDHOP_FILE

    if not os.path.exists(MEDHOP_FILE):
        print(f"medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)

    with open(MEDHOP_FILE) as f:
        data = json.load(f)

    sample = data[:20]

    gold_cov  = 0   # وثائق تحتوي الإجابة
    query_cov = 0   # وثائق تحتوي دواء السؤال

    print("\n  Coverage Comparison — Gold Answer vs Query Drug (first 20 questions)")
    print("  " + "─" * 65)
    print(f"  {'ID':<14} {'Gold(answer)':<16} {'Query drug':<16} Query Drug Name")
    print("  " + "─" * 65)

    for record in sample:
        supports   = record.get("supports", [])
        drug_name  = record.get("query_drug_name", "")
        drug_id    = record.get("query_drug_id", "")
        answer_id  = record.get("answer", "")
        answer_nm  = record.get("answer_name", "")

        gc = check_gold_coverage(supports, answer_id, answer_nm)
        qc = check_query_drug_coverage(supports, drug_name, drug_id)

        if gc["any_found"]: gold_cov += 1
        if qc["any_found"]: query_cov += 1

        print(
            f"  {record['id']:<14} "
            f"{'✓' if gc['any_found'] else '✗'} ({gc['found_by_name']}/{gc['total_docs']})"
            f"          "
            f"{'✓' if qc['any_found'] else '✗'} ({qc['found_by_name']}/{qc['total_docs']})"
            f"          {drug_name}"
        )

    print()
    print(f"  Gold answer  coverage: {gold_cov}/20  ({gold_cov/20*100:.0f}%)")
    print(f"  Query drug   coverage: {query_cov}/20  ({query_cov/20*100:.0f}%)")
    print()
    print("  Interpretation:")
    print("  - Gold coverage ~ 68-74% → 26-32% of questions have NO relevant docs for answer")
    print("  - Query drug coverage shows how well supports cover the query drug itself")
    print("  - High query coverage + lower EM → multi-hop reasoning gap (not retrieval gap)")