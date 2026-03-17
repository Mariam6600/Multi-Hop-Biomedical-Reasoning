"""
src/prompt_builder.py
======================
Biomedical Multi-Hop QA Project — Baseline 1

Builds the prompt that will be sent to BioMistral for each question.

What this module does:
  1. Takes one normalized record from medhop.json
  2. Replaces Drug IDs with real drug names
  3. Formats the prompt using the template from settings.py
  4. Returns a clean prompt string ready to send to the LLM

Prompt Design (Baseline 1):
  - NO supporting documents (added in Stage 2)
  - NO chain-of-thought (added in Stage 3)
  - Candidates list shown with names + IDs
  - Model asked to return Drug ID only (easy to evaluate)

Usage:
    py -3.10 src/prompt_builder.py
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MEDHOP_FILE

# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are a biomedical expert specializing in drug interactions.

Task: Identify which drug interacts with the given drug.

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer
- Do not add any extra text

Answer:"""


# ─────────────────────────────────────────────
# CORE FUNCTION — BUILD PROMPT
# ─────────────────────────────────────────────

def build_prompt(record: dict) -> str:
    """
    Build a prompt string from one MedHop record.

    Args:
        record: One dict from medhop.json containing:
                id, query, query_drug_id, query_drug_name,
                candidates, candidate_names, answer, answer_name

    Returns:
        Formatted prompt string ready to send to BioMistral
    """
    drug_id   = record.get("query_drug_id", "")
    drug_name = record.get("query_drug_name", drug_id)

    candidates     = record.get("candidates", [])
    candidate_names = record.get("candidate_names", candidates)

    # Build candidates list: "- Marimastat (DB00786)"
    candidates_lines = []
    for cid, cname in zip(candidates, candidate_names):
        if cname and cname != cid:
            candidates_lines.append(f"- {cname} ({cid})")
        else:
            candidates_lines.append(f"- {cid}")

    candidates_text = "\n".join(candidates_lines)

    prompt = PROMPT_TEMPLATE.format(
        drug_name       = drug_name,
        drug_id         = drug_id,
        candidates_text = candidates_text,
    )

    return prompt


# ─────────────────────────────────────────────
# BATCH FUNCTION — BUILD ALL PROMPTS
# ─────────────────────────────────────────────

def build_all_prompts(data: list) -> list:
    """
    Build prompts for all records in the dataset.

    Args:
        data: List of normalized records from medhop.json

    Returns:
        List of dicts: { id, prompt, answer, answer_name }
    """
    prompts = []
    for record in data:
        prompt = build_prompt(record)
        prompts.append({
            "id":          record["id"],
            "prompt":      prompt,
            "answer":      record["answer"],
            "answer_name": record.get("answer_name", record["answer"]),
        })
    return prompts


# ─────────────────────────────────────────────
# HELPER — EXTRACT DRUG ID FROM MODEL RESPONSE
# ─────────────────────────────────────────────

def extract_drug_id(response: str, candidates: list) -> str:
    """
    Extract a valid DrugBank ID from the model's raw response.

    Strategy:
      1. Look for DBxxxxx pattern in the response
      2. Check if it's in the candidates list
      3. If not found, return the raw response stripped

    Args:
        response:   Raw text response from BioMistral
        candidates: List of valid Drug IDs for this question

    Returns:
        Extracted Drug ID string (e.g. "DB04844")
    """
    import re

    response_clean = response.strip()

    # Strategy 1: find DBxxxxx pattern
    matches = re.findall(r"DB\d{5}", response_clean)

    if matches:
        # Prefer a match that is in the candidates list
        for match in matches:
            if match in candidates:
                return match
        # Return first match even if not in candidates
        return matches[0]

    # Strategy 2: check if the whole response is a candidate
    response_upper = response_clean.upper()
    for candidate in candidates:
        if candidate.upper() in response_upper:
            return candidate

    # Strategy 3: return cleaned response as-is
    return response_clean[:20]  # cap length for safety


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Prompt Builder — Test")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        print("  Run src/load_dataset.py first.")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  [OK]   Loaded {len(data)} records from medhop.json")

    # Test on first 3 records
    print("\n--- Sample Prompts ---\n")

    for i, record in enumerate(data[:3]):
        prompt = build_prompt(record)

        print(f"  Record #{i+1} — {record['id']}")
        print(f"  Correct Answer: {record['answer']} ({record.get('answer_name', '')})")
        print()
        print("  " + "─" * 56)
        # Print prompt with indentation
        for line in prompt.split("\n"):
            print(f"  {line}")
        print("  " + "─" * 56)

        # Test extract_drug_id
        fake_responses = [
            f"The answer is {record['answer']}.",
            record['answer'],
            f"Based on the interaction, {record['answer']} is correct.",
            "I think DB99999 is the answer.",  # wrong ID
        ]

        print(f"\n  extract_drug_id() tests:")
        for resp in fake_responses:
            extracted = extract_drug_id(resp, record["candidates"])
            match = "✅" if extracted == record["answer"] else "❌"
            print(f"    {match}  input: '{resp[:50]}' → '{extracted}'")

        print()

    # Build all prompts
    print("--- Building All Prompts ---")
    all_prompts = build_all_prompts(data)
    print(f"  [OK]   Built {len(all_prompts)} prompts")

    # Verify structure
    sample = all_prompts[0]
    assert "id"          in sample, "Missing id"
    assert "prompt"      in sample, "Missing prompt"
    assert "answer"      in sample, "Missing answer"
    assert "answer_name" in sample, "Missing answer_name"
    print("  [OK]   Prompt structure verified")

    print()
    print("=" * 60)
    print("  ✅  prompt_builder.py working correctly.")
    print("  Next step: src/llm_runner.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
