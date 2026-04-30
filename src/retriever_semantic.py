"""
src/retriever_semantic.py
=========================
Biomedical Multi-Hop QA Project — Baseline 3

Semantic retrieval using MedCPT (PubMed-trained embeddings).

MedCPT is a specialized model trained on 255M PubMed queries.
It understands biomedical relationships better than BM25.

What this module does:
  1. Load MedCPT model (from local path or Hugging Face)
  2. Encode queries and documents into semantic vectors
  3. Retrieve top-K most similar documents using cosine similarity
  4. Return formatted results compatible with inference pipeline

Usage:
    py -3.10 src/retriever_semantic.py
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MEDCPT_QUERY_ENCODER,
    MEDCPT_TOP_K,
    MEDCPT_BATCH_SIZE,
    MODELS_DIR,
)

# ─────────────────────────────────────────────
# GLOBAL MODEL INSTANCE (lazy loading)
# ─────────────────────────────────────────────

_model = None


def get_model():
    """
    Load MedCPT model (cached globally).
    Supports both local path and Hugging Face model name.
    """
    global _model
    
    if _model is None:
        print("  [INFO] Loading MedCPT model...")
        
        # Check if using local path
        if os.path.exists(MEDCPT_QUERY_ENCODER):
            print(f"         Loading from LOCAL path: {MEDCPT_QUERY_ENCODER}")
        else:
            print(f"         Loading from Hugging Face: {MEDCPT_QUERY_ENCODER}")
        
        from sentence_transformers import SentenceTransformer
        
        # تم التعديل هنا: إضافة device='cpu' لإجبار الموديل على العمل على المعالج
        # هذا يحل مشكلة عدم توافق كرت الشاشة RTX 5050
        _model = SentenceTransformer(MEDCPT_QUERY_ENCODER, device='cpu')
        
        print("  [OK]   MedCPT model loaded successfully!")
    
    return _model


# ─────────────────────────────────────────────
# CORE RETRIEVAL FUNCTION
# ─────────────────────────────────────────────

def retrieve_semantic(
    query: str,
    supports: list,
    drug_name: str = "",
    top_k: int = None
) -> list:
    """
    Retrieve top-K most relevant supports using semantic similarity.
    
    Args:
        query:         The question text (e.g., "interacts_with DB01171?")
        supports:      List of supporting texts to search through
        drug_name:     Drug name to enhance the query (optional)
        top_k:         Number of results to return (default from settings)
    
    Returns:
        List of dicts: [{"text": ..., "score": ..., "rank": ...}, ...]
    """
    if not supports:
        return []
    
    if top_k is None:
        top_k = MEDCPT_TOP_K
    
    start_time = time.time()
    
    # Enhance query with drug name
    if drug_name:
        enhanced_query = f"{drug_name} {query}"
    else:
        enhanced_query = query
    
    # Get model
    model = get_model()
    
    # Encode query
    query_embedding = model.encode([enhanced_query], show_progress_bar=False)
    query_embedding = np.array(query_embedding)
    
    # Encode all supports
    # Handle empty or None texts
    valid_supports = []
    valid_indices = []
    for i, text in enumerate(supports):
        if text and isinstance(text, str) and text.strip():
            valid_supports.append(text.strip())
            valid_indices.append(i)
    
    if not valid_supports:
        return []
    
    # Encode in batches to avoid memory issues
    doc_embeddings = model.encode(
        valid_supports, 
        batch_size=MEDCPT_BATCH_SIZE,
        show_progress_bar=False
    )
    doc_embeddings = np.array(doc_embeddings)
    
    # Calculate cosine similarity
    # Normalize vectors for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Cosine similarity: dot product of normalized vectors
    similarities = np.dot(query_norm, doc_norm.T).flatten()
    
    # Get top-K indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, 1):
        original_idx = valid_indices[idx]
        results.append({
            "text": supports[original_idx],
            "score": float(similarities[idx]),
            "rank": rank,
            "original_index": original_idx,
        })
    
    retrieval_time = time.time() - start_time
    
    return results


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count."""
    return len(text) // 4


def check_answer_in_retrieved(retrieved: list, answer_names: list) -> bool:
    """
    Check if any answer name appears in retrieved texts.
    """
    if not retrieved or not answer_names:
        return False
    
    combined_text = " ".join([r["text"].lower() for r in retrieved])
    
    for name in answer_names:
        if name and name.lower() in combined_text:
            return True
    
    return False


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  MedCPT Semantic Retrieval — Test")
    print("=" * 60)
    
    # Check model path
    print()
    print(f"  Model path setting: {MEDCPT_QUERY_ENCODER}")
    if os.path.exists(MEDCPT_QUERY_ENCODER):
        print(f"  Status: LOCAL model found ✅")
    else:
        print(f"  Status: Will download from Hugging Face")
    print()
    
    # Load test data
    import json
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "medhop.json"
    )
    
    try:
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [OK]   Loaded {len(data)} questions")
    except Exception as e:
        print(f"  [FAIL] Could not load data: {e}")
        return
    
    print()
    print(f"  Top-K: {MEDCPT_TOP_K}")
    print()
    
    # Test on first 3 questions
    print("-" * 60)
    print("  Testing Semantic Retrieval on 3 Questions")
    print("-" * 60)
    
    for i, record in enumerate(data[:3]):
        print()
        print(f"  Question {i+1}: {record['id']}")
        print(f"    Query:  {record['query']}")
        print(f"    Drug:   {record.get('query_drug_name')} ({record.get('query_drug_id')})")
        print(f"    Answer: {record['answer']} ({record.get('answer_name', '')})")
        
        supports = record.get("supports", [])
        print(f"    Supports: {len(supports)} texts available")
        
        if not supports:
            print("    [SKIP] No supports available")
            continue
        
        # Retrieve
        query = record.get("query", "")
        drug_name = record.get("query_drug_name", "")
        
        retrieved = retrieve_semantic(
            query=query,
            supports=supports,
            drug_name=drug_name,
            top_k=MEDCPT_TOP_K
        )
        
        print()
        print(f"    Top-{len(retrieved)} Retrieved:")
        for r in retrieved:
            print(f"      Rank {r['rank']}: score={r['score']:.4f}")
            text_preview = r['text'][:80].replace('\n', ' ')
            print(f"               {text_preview}...")
        
        # Check if answer found
        answer_name = record.get("answer_name", "")
        if answer_name:
            found = check_answer_in_retrieved(retrieved, [answer_name])
            print()
            print(f"      Answer drug in retrieved: {'YES' if found else 'NO'}")
    
    print()
    print("=" * 60)
    print("  retriever_semantic.py working correctly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()