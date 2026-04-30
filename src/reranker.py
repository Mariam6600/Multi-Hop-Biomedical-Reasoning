"""
src/reranker.py
================
Biomedical Multi-Hop QA — Cross-Encoder Re-Ranker
B5-EXP1: Guided Retrieval + Re-Ranking

الفكرة:
  بعد أن يجلب Hybrid Scored Retriever أفضل Top-N وثيقة (N=10),
  يقوم الـ Cross-Encoder بقراءة كل وثيقة مع السؤال والآلية معاً
  ويعطيها نقطة صلة دقيقة، ثم نأخذ أفضل K=3 فقط.

الفرق عن الـ Bi-Encoder (MedCPT):
  - MedCPT (Bi-Encoder): يرمّز السؤال والوثيقة بشكل منفصل → سريع لكن أقل دقة
  - Cross-Encoder: يقرأ السؤال + الوثيقة معاً في نفس الـ pass → أبطأ لكن أدق بكثير

النموذج المستخدم:
  cross-encoder/ms-marco-MiniLM-L-6-v2
  - صغير وسريع (6 layers)
  - يعمل محلياً بدون API
  - مدرّب على قراءة جملتين معاً وإعطاء نقطة صلة

Strategy:
  1. Hybrid Scored يجلب Top-10 (بدلاً من Top-3)
  2. Cross-Encoder يرتّبها من جديد بناءً على (query + mechanism + doc)
  3. نأخذ أفضل 3 فقط للـ LLM
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────
# GLOBAL MODEL INSTANCE (lazy loading)
# ─────────────────────────────────────────────

_reranker_model = None
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker():
    """
    Load Cross-Encoder model (cached globally).
    Downloads automatically on first use (~80 MB).
    """
    global _reranker_model

    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"  [INFO] Loading Cross-Encoder re-ranker: {RERANKER_MODEL_NAME}")
            # negligence GPU to avoid compatibility issues, force CPU
            _reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cpu')            
            print(f"  [OK]   Cross-Encoder loaded successfully!")
        except Exception as e:
            print(f"  [WARN] Could not load Cross-Encoder: {e}")
            print(f"  [WARN] Install with: pip install sentence-transformers --break-system-packages")
            _reranker_model = None

    return _reranker_model


def rerank_documents(
    query: str,
    bridge_info: str,
    retrieved_docs: list,
    final_k: int = 3,
) -> list:
    """
    Re-rank retrieved documents using Cross-Encoder.

    Args:
        query:         السؤال الأصلي
        bridge_info:   آلية عمل الدواء (من bridge cache)، مثل "inhibits CYP3A4"
        retrieved_docs: قائمة الوثائق من do_retrieval() — [{text, score, rank, ...}]
        final_k:       عدد الوثائق النهائية بعد إعادة الترتيب

    Returns:
        قائمة بأفضل final_k وثيقة بعد Re-Ranking، بنفس تنسيق المدخل
        مع إضافة حقل rerank_score لكل وثيقة
    """
    if not retrieved_docs:
        return []

    model = get_reranker()

    # إذا فشل تحميل النموذج → نرجع أفضل K من الاسترجاع الأصلي
    if model is None:
        return retrieved_docs[:final_k]

    # بناء query مُثرى بالآلية للـ Cross-Encoder
    # هذا هو جوهر التحسين: نعطي Cross-Encoder السؤال + الآلية معاً
    if bridge_info and bridge_info not in ("unknown", ""):
        enriched_query = f"{query} | Mechanism: {bridge_info}"
    else:
        enriched_query = query

    # تحضير pairs لـ Cross-Encoder: (query, document)
    pairs = [(enriched_query, doc["text"]) for doc in retrieved_docs]

    try:
        scores = model.predict(pairs)
    except Exception as e:
        print(f"  [WARN] Re-ranker failed: {e} — using original ranking")
        return retrieved_docs[:final_k]

    # دمج النقاط مع الوثائق
    scored_docs = []
    for doc, score in zip(retrieved_docs, scores):
        new_doc = dict(doc)
        new_doc["rerank_score"] = float(score)
        new_doc["original_score"] = doc.get("score", 0.0)
        scored_docs.append(new_doc)

    # ترتيب تنازلي حسب نقطة Re-Ranker
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    # تحديث rank
    for i, doc in enumerate(scored_docs[:final_k], 1):
        doc["rank"] = i

    return scored_docs[:final_k]


def check_reranker_available() -> bool:
    """Check if re-ranker can be loaded without actually loading it."""
    try:
        from sentence_transformers import CrossEncoder
        return True
    except ImportError:
        return False