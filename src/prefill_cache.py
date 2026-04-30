"""
src/prefill_cache.py  —  Fixed version
FIXES:
  1. Uses max_tokens_override=600 to get all 30 terms without truncation
  2. Adds /no_think directive to prompt → disables Qwen3 thinking mode
  3. Better term parsing: strips numbering, bullets, empty lines
  4. Validates terms: skips lines that are clearly not drug terms
  5. Shows term count per drug so you can verify quality
  6. Skips drugs already in cache (resumes if interrupted)

Usage:
  1. settings.py → ACTIVE_PROVIDER = "groq"
  2. py -3.10 src/prefill_cache.py
  3. py -3.10 src/inference_pipeline5.py   ← now uses cached terms
"""

import json, os, sys, time, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MEDHOP_FILE, OUTPUTS_DIR, EXPANSION_N_TERMS
from src.llm_runner_api import get_api_client, run_inference

CACHE_FILE = os.path.join(OUTPUTS_DIR, "expansion_cache.json")

# ─────────────────────────────────────────────
# PROMPT — uses /no_think to disable Qwen3 thinking
# This gives clean output and saves ~400 tokens per call
# ─────────────────────────────────────────────
EXPANSION_PROMPT = """/no_think
You are a biomedical expert. For the drug below, list exactly {n} biomedical terms.

Include:
- Mechanism of action (enzymes inhibited, receptors targeted)
- Biological pathways affected
- Drug class and subclass
- Drug interaction mechanisms
- Metabolites and related compounds

Drug: {drug_name}

Output ONLY the terms, one per line. No numbers, no bullets, no explanations.

Terms:"""


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_all_drugs() -> list:
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)
    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)
    drugs = list(dict.fromkeys(r.get("query_drug_name", "") for r in data))
    return [d for d in drugs if d]


def parse_terms(raw: str) -> list:
    """
    Parse LLM output into clean term list.
    Removes: numbering, bullets, empty lines, junk lines
    """
    if not raw:
        return []

    terms = []
    for line in raw.splitlines():
        # Strip common prefixes: "1.", "-", "*", "•", "  "
        line = re.sub(r"^[\s\-\*\•\d\.\:\)]+", "", line).strip()
        line = re.sub(r"\*+$", "", line).strip()   # trailing **

        # Skip empty
        if not line:
            continue

        # Skip clearly bad lines (junk from thinking or prompt leakage)
        lower = line.lower()
        skip_patterns = [
            len(line) > 80,                         # too long → paragraph not term
            line.startswith("<"),                    # HTML/XML tag
            "you are" in lower,                     # prompt leak
            "i cannot" in lower,                    # refusal
            "i need to" in lower,                   # thinking leak
            "let me" in lower,                      # thinking leak
            "okay" == lower[:4],                    # thinking leak
            line.count(" ") > 6,                    # too many words → sentence
        ]
        if any(skip_patterns):
            continue

        terms.append(line)

    return terms


def main():
    print("\n" + "=" * 60)
    print("  Cache Prefill — All Drugs")
    print("=" * 60)
    print()

    cache    = load_cache()
    all_drugs = get_all_drugs()

    print(f"  Cache has {len(cache)} existing entries")
    print(f"  Dataset has {len(all_drugs)} unique drug names")

    # Find missing drugs
    missing = []
    for drug in all_drugs:
        local_key = f"local::{drug}::{EXPANSION_N_TERMS}"
        cloud_key = f"qwen/qwen3.6-plus-preview:free::{drug}::{EXPANSION_N_TERMS}"
        if local_key not in cache and cloud_key not in cache:
            missing.append(drug)

    print(f"  Missing from cache: {len(missing)} drugs")

    if not missing:
        print("\n  All drugs already in cache!")
        return

    print("\n--- Connecting to API ---")
    client = get_api_client()
    print()
    print(f"  Will process {len(missing)} drugs")
    print(f"  Estimated time: ~{len(missing) * 2 / 60:.1f} minutes\n")

    success_count = 0
    failed_drugs  = []
    start         = time.time()

    for i, drug in enumerate(missing):
        cache_key = f"local::{drug}::{EXPANSION_N_TERMS}"
        prompt    = EXPANSION_PROMPT.format(drug_name=drug, n=EXPANSION_N_TERMS)

        # Use max_tokens_override=600 to get all 30 terms
        # (LLM_MAX_TOKENS=200 is too small and causes truncation)
        result = run_inference(client, prompt, question_id=drug,
                               max_tokens_override=600)

        if result["success"] and result["raw_response"]:
            terms = parse_terms(result["raw_response"])

            # Warn if too few terms
            if len(terms) < 10:
                print(f"  [WARN] Only {len(terms)} terms for {drug} — check quality")

            cache[cache_key] = {
                "drug_name":  drug,
                "terms":      terms,
                "raw":        result["raw_response"],
                "success":    True,
                "time":       result["inference_time"],
                "error":      "",
                "from_cache": False,
            }
            save_cache(cache)
            success_count += 1

            elapsed   = time.time() - start
            avg       = elapsed / (i + 1)
            remaining = avg * (len(missing) - i - 1)
            print(f"  [{i+1:>3}/{len(missing)}] OK  {drug:<30} "
                  f"({len(terms):>2} terms) | ETA: {remaining/60:.1f}min")
        else:
            failed_drugs.append(drug)
            print(f"  [{i+1:>3}/{len(missing)}] FAIL {drug:<30} "
                  f"Error: {result['error'][:50]}")

    total_time = time.time() - start
    print()
    print("=" * 60)
    print(f"  Done: {success_count}/{len(missing)} drugs")
    print(f"  Failed: {len(failed_drugs)}")
    print(f"  Time: {total_time/60:.1f} minutes")
    print(f"  Cache size: {len(cache)} entries")
    if failed_drugs:
        print(f"  Failed: {failed_drugs[:10]}")
    print("=" * 60)
    print(f"\n  Saved → {CACHE_FILE}")
    print(f"  Next: py -3.10 src/inference_pipeline5.py\n")


if __name__ == "__main__":
    main()