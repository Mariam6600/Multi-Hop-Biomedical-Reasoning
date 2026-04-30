"""
src/query_expander_weighted.py
================================
يحوّل الـ flat terms الموجودة في الكاش إلى weighted_terms
بناءً على قواعد بسيطة — بدون أي API call جديد.

لماذا هذا الحل ذكي؟
  - الكاش عنده 66 دواء جاهز من Qwen3.6 (أفضل نموذج API)
  - بدل ما نطلب من Qwen3.5-9B يعيد توليدها (أبطأ وأضعف)
  - نصنّف الـ terms الموجودة تلقائياً حسب الكلمات الدالة
  - نتيجة: نفس جودة الـ terms + أوزان مفيدة + سرعة فورية

منطق التصنيف:
  mechanism:  inhibitor, inhibition, antagonist, agonist, blocker, reuptake...
  protein:    CYP, enzyme, receptor, transporter, kinase, protein, synthase...
  pathway:    pathway, signaling, cascade, metabolism, synthesis...
  drug_class: SSRI, MAOI, antidepressant, anticoagulant, class, inhibitor class...
  disease:    disorder, syndrome, disease, depression, anxiety, cancer...

Usage:
    from src.query_expander_weighted import get_weighted_terms
    weighted = get_weighted_terms(drug_name)
    # weighted = [{"term": str, "weight": float, "category": str}, ...]
"""

import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OUTPUTS_DIR

# ─────────────────────────────────────────────
# Category weights (per supervisor's recommendation)
# ─────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    "mechanism":  3.0,   # enzyme inhibition, receptor binding — most important for DDI
    "protein":    2.5,   # CYP enzymes, transporters, receptors
    "pathway":    2.0,   # biological pathways
    "drug_class": 1.5,   # SSRI, MAOI, anticoagulant...
    "disease":    1.0,   # conditions treated — least important for DDI
    "unknown":    1.0,
}

# ─────────────────────────────────────────────
# Classification rules (keyword-based)
# ─────────────────────────────────────────────

MECHANISM_KEYWORDS = [
    "inhibitor", "inhibition", "inhibit", "antagonist", "antagonism",
    "agonist", "blocker", "reuptake", "uptake", "binding", "reversible",
    "selective", "competitive", "irreversible", "mode of action",
    "mechanism", "potentiation", "activation", "suppression",
]

PROTEIN_KEYWORDS = [
    "cyp", "enzyme", "receptor", "transporter", "kinase", "protein",
    "synthase", "oxidase", "reductase", "dehydrogenase", "transferase",
    "pgp", "p-glycoprotein", "mao", "cox", "hmg", "ace",
    "sert", "dat", "net",  # neurotransmitter transporters
]

PATHWAY_KEYWORDS = [
    "pathway", "signaling", "cascade", "metabolism", "metabolic",
    "synthesis", "biosynthesis", "degradation", "clearance",
    "elimination", "hepatic", "renal", "first-pass",
]

DRUG_CLASS_KEYWORDS = [
    "ssri", "snri", "maoi", "rima", "tricyclic", "antidepressant",
    "anticoagulant", "antiplatelet", "antibiotic", "antifungal",
    "antiviral", "antipsychotic", "benzodiazepine", "statin",
    "beta-blocker", "ace inhibitor", "antihistamine", "opioid",
    "analgesic", "nsaid", "proton pump",
    "drug class", "pharmacological class", "therapeutic class",
]

DISEASE_KEYWORDS = [
    "disorder", "syndrome", "disease", "depression", "anxiety",
    "cancer", "tumor", "infection", "hypertension", "diabetes",
    "pain", "inflammation", "psychosis", "epilepsy", "schizophrenia",
    "condition", "disorder", "dysfunction",
]


def classify_term(term: str) -> tuple:
    """
    Classify a term into a category based on keywords.
    Returns (category, weight)
    """
    t_lower = term.lower()

    if any(kw in t_lower for kw in MECHANISM_KEYWORDS):
        return "mechanism", CATEGORY_WEIGHTS["mechanism"]
    if any(kw in t_lower for kw in PROTEIN_KEYWORDS):
        return "protein", CATEGORY_WEIGHTS["protein"]
    if any(kw in t_lower for kw in PATHWAY_KEYWORDS):
        return "pathway", CATEGORY_WEIGHTS["pathway"]
    if any(kw in t_lower for kw in DRUG_CLASS_KEYWORDS):
        return "drug_class", CATEGORY_WEIGHTS["drug_class"]
    if any(kw in t_lower for kw in DISEASE_KEYWORDS):
        return "disease", CATEGORY_WEIGHTS["disease"]

    return "unknown", CATEGORY_WEIGHTS["unknown"]


def get_weighted_terms(drug_name: str, cache_file: str = None) -> list:
    """
    Convert flat terms from cache to weighted terms.
    No API call needed — reads from existing expansion_cache.json.

    Args:
        drug_name:  e.g. "Moclobemide"
        cache_file: path to flat cache (default: outputs/expansion_cache.json)

    Returns:
        [{"term": str, "weight": float, "category": str}, ...]
    """
    if cache_file is None:
        cache_file = os.path.join(OUTPUTS_DIR, "expansion_cache.json")

    # Load cache
    if not os.path.exists(cache_file):
        return []

    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return []

    # Try different key formats
    flat_terms = None
    for n_terms in [30, 20, 25, 15]:
        key = f"local::{drug_name}::{n_terms}"
        if key in cache:
            flat_terms = cache[key].get("terms", [])
            break
        # Also try Qwen3.6 cloud key
        key = f"qwen/qwen3.6-plus-preview:free::{drug_name}::{n_terms}"
        if key in cache:
            flat_terms = cache[key].get("terms", [])
            break

    if not flat_terms:
        return []

    # Classify each term
    weighted = []
    seen = set()
    for term in flat_terms:
        t_clean = term.strip()
        if not t_clean or t_clean.lower() in seen:
            continue
        seen.add(t_clean.lower())
        category, weight = classify_term(t_clean)
        weighted.append({
            "term":     t_clean,
            "weight":   weight,
            "category": category,
        })

    return weighted


def get_weighted_terms_stats(drug_name: str) -> dict:
    """Print breakdown of weighted terms by category — useful for debugging."""
    weighted = get_weighted_terms(drug_name)
    stats = {}
    for wt in weighted:
        cat = wt["category"]
        stats.setdefault(cat, []).append(wt["term"])

    return {
        "drug": drug_name,
        "total": len(weighted),
        "by_category": {cat: {"count": len(terms), "terms": terms}
                        for cat, terms in stats.items()},
        "weighted_terms": weighted,
    }


def expand_query_structured(drug_name: str, cache_file: str = None) -> dict:
    """
    Wrapper function required by inference_pipeline5.py.
    It calls get_weighted_terms and formats the output dict.
    """
    weighted_terms = get_weighted_terms(drug_name, cache_file)
    
    return {
        "drug_name": drug_name,
        "weighted_terms": weighted_terms,
        "terms": [t["term"] for t in weighted_terms],  # قائمة مسطحة للتوافقية
        "success": len(weighted_terms) > 0,
        "from_cache": True,  # لأننا نقرأ من الكاش
    }

if __name__ == "__main__":
    # Test with Moclobemide
    print("\nTest: get_weighted_terms('Moclobemide')")
    wt = get_weighted_terms("Moclobemide")
    print(f"  Total: {len(wt)} terms")

    # Show by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for w in wt:
        by_cat[w["category"]].append((w["term"], w["weight"]))

    for cat, items in sorted(by_cat.items(), key=lambda x: -CATEGORY_WEIGHTS.get(x[0], 1.0)):
        print(f"\n  {cat} (weight={CATEGORY_WEIGHTS.get(cat, 1.0)}):")
        for term, weight in items[:4]:
            print(f"    [{weight}] {term}")