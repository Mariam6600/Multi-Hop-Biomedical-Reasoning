"""
src/query_expander.py
======================
Biomedical Multi-Hop QA — Semantic Query Expansion (FIXED VERSION)

إصلاحات في هذا الإصدار:
  1. حذف <think>...</think> من ردود Qwen3 (سبب تكرار المصطلحات)
  2. إزالة التكرار من قائمة المصطلحات (deduplication)
  3. تحديد عدد المصطلحات بـ N كحد أقصى بعد التنظيف
  4. حفظ نسخة احتياطية قبل الكتابة
"""

import time
import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    OLLAMA_MODEL_NAME,
    EXPANSION_N_TERMS,
    OUTPUTS_DIR,
)

# ─────────────────────────────────────────────
# CACHE SETUP
# ─────────────────────────────────────────────

CACHE_FILE = os.path.join(OUTPUTS_DIR, "expansion_cache.json")

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

_CACHE = _load_cache()

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

EXPANSION_PROMPT = """You are a biomedical expert. Given a drug name, list exactly {n} biomedical \
terms and keywords that are closely related to this drug.

Include terms from these categories:
- Mechanism of action (enzymes inhibited/activated, receptors targeted)
- Biological pathways affected
- Drug class and subclass
- Known drug interactions
- Metabolites

Drug: {drug_name}

Rules:
- Output ONLY the terms, one per line, no numbering, no explanations
- Each term must be 1-4 words maximum
- Focus on pharmacologically relevant terms
- Do not include the drug name itself

Terms:"""


# ─────────────────────────────────────────────
# CORE FUNCTION (SMART CACHE)
# ─────────────────────────────────────────────

def expand_query(drug_name: str, n_terms: int = None) -> dict:
    """
    1. Check OLD Cloud Cache (qwen3.6-plus-preview) — reuse previous work.
    2. Check NEW Local Cache.
    3. Generate using Local Ollama if not found.
    """
    if n_terms is None:
        n_terms = EXPANSION_N_TERMS

    old_cloud_key = f"qwen/qwen3.6-plus-preview:free::{drug_name}::{n_terms}"
    new_local_key = f"local::{drug_name}::{n_terms}"

    # Check OLD Cloud Cache
    if old_cloud_key in _CACHE:
        cached = _CACHE[old_cloud_key]
        # FIX: deduplicate even cached results on return
        cached["terms"] = _deduplicate(cached.get("terms", []), n_terms)
        cached["from_cache"] = True
        return cached

    # Check NEW Local Cache
    if new_local_key in _CACHE:
        cached = _CACHE[new_local_key]
        # FIX: deduplicate even cached results on return
        cached["terms"] = _deduplicate(cached.get("terms", []), n_terms)
        cached["from_cache"] = True
        return cached

    # Generate New Terms Locally
    result = {
        "drug_name":  drug_name,
        "terms":      [],
        "raw":        "",
        "success":    False,
        "time":       0.0,
        "error":      "",
        "from_cache": False,
    }

    try:
        import ollama
        client = ollama.Client()

        prompt = EXPANSION_PROMPT.format(drug_name=drug_name, n=n_terms)

        start = time.time()

        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_ctx": 2048}
        )

        elapsed = time.time() - start

        raw = response["message"]["content"].strip()

        # FIX 1: Strip thinking blocks (Qwen3.5 outputs <think>...</think>)
        # Without this, terms appear twice: once in think block, once in response
        raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw_clean = re.sub(r"<\|thinking\|>.*?<\|/thinking\|>", "", raw_clean, flags=re.DOTALL).strip()
        if not raw_clean:
            raw_clean = raw  # fallback if stripping removed everything

        terms = _parse_terms(raw_clean, n_terms)

        result.update({
            "terms":   terms,
            "raw":     raw_clean,
            "success": len(terms) > 0,
            "time":    round(elapsed, 3),
        })

        # Save to cache (use a COPY to avoid shared reference issues)
        _CACHE[new_local_key] = dict(result)
        _CACHE[new_local_key]["terms"] = list(terms)  # FIX 2: copy the list
        _save_cache(_CACHE)

    except Exception as e:
        result["error"] = str(e)
        print(f"  [WARN] Local expansion failed for {drug_name}: {e}")

    return result


def _deduplicate(terms: list, max_n: int) -> list:
    """Remove duplicates preserving order, limit to max_n unique terms."""
    seen = set()
    unique = []
    for t in terms:
        t_norm = t.lower().strip()
        if t_norm and t_norm not in seen:
            seen.add(t_norm)
            unique.append(t)
        if len(unique) >= max_n:
            break
    return unique


def _parse_terms(raw: str, max_n: int = 30) -> list:
    """
    Parse LLM output into a clean list of terms.
    Handles both newline-separated and comma-separated output.
    Deduplicates and limits to max_n terms.
    """
    # Try newline split first
    lines = raw.strip().splitlines()

    # If very few lines but content has commas → try comma split
    if len(lines) <= 3 and "," in raw:
        lines = re.split(r"[,\n]", raw)

    terms = []
    for line in lines:
        # Remove numbering, bullets, etc.
        line = re.sub(r"^[\s\-\*\•\d\.\)]+", "", line).strip()
        # Remove trailing punctuation
        line = re.sub(r"[,;]+$", "", line).strip()
        if line and 1 < len(line) < 80:
            terms.append(line)

    # FIX 3: Deduplicate and limit
    return _deduplicate(terms, max_n)


def build_expansion_query(drug_name: str, terms: list) -> str:
    return drug_name + " " + " ".join(terms)