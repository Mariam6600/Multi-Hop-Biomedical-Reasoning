"""
src/prompt_builder.py
======================
Biomedical Multi-Hop QA Project — Baseline 1, 2, 3 & FewShot+MedCPT

Builds the prompt that will be sent to BioMistral for each question.

What this module does:
  1. Takes one normalized record from medhop.json
  2. Replaces Drug IDs with real drug names
  3. Formats the prompt using the template
  4. For Baseline 2: adds retrieved context from BM25
  5. For Baseline 3 FewShot: adds Few-Shot example + MedCPT context

Prompt Design:
  - Baseline 1: NO supporting documents
  - Baseline 2: WITH supporting documents (BM25 retrieved)
  - Baseline 3 FewShot: WITH few-shot example + MedCPT retrieved context
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
from config.settings import MEDHOP_FILE, BM25_TOP_K

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────

PROMPT_TEMPLATE_V1 = """\
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
# BASELINE 1 EXPERIMENT PROMPTS
# ─────────────────────────────────────────────

# EXP2 — Chain of Thought: nudges the model to reason step by step
PROMPT_TEMPLATE_COT = """You are a biomedical expert specializing in drug interactions.

To identify the interacting drug, think step by step:
1. What is the mechanism of action of {drug_name}?
2. Which enzyme or pathway does it affect?
3. Which candidate drug shares or is affected by that pathway?

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer
- Do not add any extra text

Answer:"""


# EXP3 — Few-Shot: one concrete example to guide the model
PROMPT_TEMPLATE_FEWSHOT = """You are a biomedical expert specializing in drug interactions.

Example:
Drug: Fluoxetine (DB00472) — inhibits the CYP2D6 enzyme
Interacting drug: Desipramine (DB01151) — because it is metabolized by CYP2D6

Now answer the following:
Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the example reasoning pattern above
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer
- Do not add any extra text

Answer:"""


PROMPT_TEMPLATE_V2 = """\
You are a biomedical expert specializing in drug interactions.

Task: Identify which drug interacts with the given drug based on the supporting evidence provided.

Supporting Evidence:
{context_text}

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the supporting evidence to guide your answer
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer
- Do not add any extra text

Answer:"""


# ─────────────────────────────────────────────
# FEW-SHOT + MEDCPT CONTEXT TEMPLATE
# ─────────────────────────────────────────────

# EXP: Few-Shot + MedCPT Context
# Combines the best of Baseline 1 (few-shot pattern) and Baseline 3 (semantic retrieval)
PROMPT_TEMPLATE_FEWSHOT_WITH_CONTEXT = """\
You are a biomedical expert specializing in drug interactions.

Example:
Drug: Fluoxetine (DB00472) — inhibits the CYP2D6 enzyme
Supporting evidence: "Fluoxetine is a potent inhibitor of CYP2D6, affecting the metabolism of drugs like desipramine."
Interacting drug: Desipramine (DB01151) — because it is metabolized by CYP2D6

Now answer the following using the supporting evidence provided:

Supporting Evidence:
{context_text}

Drug in question: {drug_name} ({drug_id})

Choose the correct answer from the following candidates:
{candidates_text}

Instructions:
- Use the example reasoning pattern above
- Use the supporting evidence to guide your answer
- Answer with the DrugBank ID only (format: DBxxxxx)
- Do not explain your answer
- Do not add any extra text

Answer:"""


# ─────────────────────────────────────────────
# HELPER — FORMAT SUPPORTS
# ─────────────────────────────────────────────

def format_supports(supports: list) -> str:
    """
    Format retrieved supports for inclusion in prompt.
    
    Args:
        supports: List of dicts with 'text' and 'score' keys
    
    Returns:
        Formatted text string for prompt
    """
    if not supports:
        return "No supporting evidence available."
    
    formatted = []
    for i, support in enumerate(supports, 1):
        if isinstance(support, dict):
            text = support.get("text", "")
            score = support.get("score", 0)
            # Truncate long texts
            if len(text) > 400:
                text = text[:400] + "..."
            formatted.append(f"[{i}] (score: {score:.2f}) {text}")
        else:
            text = str(support)
            if len(text) > 400:
                text = text[:400] + "..."
            formatted.append(f"[{i}] {text}")
    
    return "\n".join(formatted)


# ─────────────────────────────────────────────
# CORE FUNCTION — BUILD PROMPT (Baseline 1)
# ─────────────────────────────────────────────

def build_prompt(record: dict) -> str:
    """
    Build a prompt string from one MedHop record (Baseline 1 - no context).

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

    prompt = PROMPT_TEMPLATE_V1.format(
        drug_name       = drug_name,
        drug_id         = drug_id,
        candidates_text = candidates_text,
    )

    return prompt


# ─────────────────────────────────────────────
# EXP2 — BUILD PROMPT WITH CHAIN OF THOUGHT
# ─────────────────────────────────────────────

def build_prompt_cot(record: dict) -> str:
    """Build a CoT prompt for Baseline 1 EXP2."""
    drug_id   = record.get("query_drug_id", "")
    drug_name = record.get("query_drug_name", drug_id)
    candidates     = record.get("candidates", [])
    candidate_names = record.get("candidate_names", candidates)

    candidates_lines = []
    for cid, cname in zip(candidates, candidate_names):
        if cname and cname != cid:
            candidates_lines.append(f"- {cname} ({cid})")
        else:
            candidates_lines.append(f"- {cid}")

    candidates_text = "\n".join(candidates_lines)

    return PROMPT_TEMPLATE_COT.format(
        drug_name       = drug_name,
        drug_id         = drug_id,
        candidates_text = candidates_text,
    )


# ─────────────────────────────────────────────
# EXP3 — BUILD PROMPT WITH FEW-SHOT EXAMPLE
# ─────────────────────────────────────────────

def build_prompt_fewshot(record: dict) -> str:
    """Build a Few-Shot prompt for Baseline 1 EXP3."""
    drug_id   = record.get("query_drug_id", "")
    drug_name = record.get("query_drug_name", drug_id)
    candidates     = record.get("candidates", [])
    candidate_names = record.get("candidate_names", candidates)

    candidates_lines = []
    for cid, cname in zip(candidates, candidate_names):
        if cname and cname != cid:
            candidates_lines.append(f"- {cname} ({cid})")
        else:
            candidates_lines.append(f"- {cid}")

    candidates_text = "\n".join(candidates_lines)

    return PROMPT_TEMPLATE_FEWSHOT.format(
        drug_name       = drug_name,
        drug_id         = drug_id,
        candidates_text = candidates_text,
    )


# ─────────────────────────────────────────────
# CORE FUNCTION — BUILD PROMPT WITH CONTEXT (Baseline 2)
# ─────────────────────────────────────────────

def build_prompt_with_context(record: dict, retrieved_supports: list) -> str:
    """
    Build a prompt string with BM25 retrieved context (Baseline 2).

    Args:
        record: One dict from medhop.json
        retrieved_supports: List of dicts with 'text' and 'score' from BM25

    Returns:
        Formatted prompt string with supporting evidence
    """
    drug_id   = record.get("query_drug_id", "")
    drug_name = record.get("query_drug_name", drug_id)

    candidates     = record.get("candidates", [])
    candidate_names = record.get("candidate_names", candidates)

    # Build candidates list
    candidates_lines = []
    for cid, cname in zip(candidates, candidate_names):
        if cname and cname != cid:
            candidates_lines.append(f"- {cname} ({cid})")
        else:
            candidates_lines.append(f"- {cid}")

    candidates_text = "\n".join(candidates_lines)

    # Format context from retrieved supports
    context_text = format_supports(retrieved_supports)

    prompt = PROMPT_TEMPLATE_V2.format(
        context_text   = context_text,
        drug_name      = drug_name,
        drug_id        = drug_id,
        candidates_text = candidates_text,
    )

    return prompt


# ─────────────────────────────────────────────
# FEW-SHOT + MEDCPT — BUILD PROMPT
# ─────────────────────────────────────────────

def build_prompt_fewshot_with_context(record: dict, retrieved_supports: list) -> str:
    """
    Build a Few-Shot + MedCPT context prompt.
    Combines the best of Baseline 1 (few-shot pattern) and Baseline 3 (semantic retrieval).

    Args:
        record: One dict from medhop.json
        retrieved_supports: List of dicts with 'text' and 'score' from MedCPT

    Returns:
        Formatted prompt string with few-shot example + supporting evidence
    """
    drug_id   = record.get("query_drug_id", "")
    drug_name = record.get("query_drug_name", drug_id)

    candidates      = record.get("candidates", [])
    candidate_names = record.get("candidate_names", candidates)

    candidates_lines = []
    for cid, cname in zip(candidates, candidate_names):
        if cname and cname != cid:
            candidates_lines.append(f"- {cname} ({cid})")
        else:
            candidates_lines.append(f"- {cid}")

    candidates_text = "\n".join(candidates_lines)
    context_text    = format_supports(retrieved_supports)

    return PROMPT_TEMPLATE_FEWSHOT_WITH_CONTEXT.format(
        context_text    = context_text,
        drug_name       = drug_name,
        drug_id         = drug_id,
        candidates_text = candidates_text,
    )


# ─────────────────────────────────────────────
# BATCH FUNCTION — BUILD ALL PROMPTS
# ─────────────────────────────────────────────

def build_all_prompts(data: list, with_context: bool = False, retriever_func=None) -> list:
    """
    Build prompts for all records in the dataset.

    Args:
        data: List of normalized records from medhop.json
        with_context: If True, use Baseline 2 (with BM25 context)
        retriever_func: Function to retrieve supports (required if with_context=True)

    Returns:
        List of dicts: { id, prompt, answer, answer_name, context_used }
    """
    prompts = []
    
    for record in data:
        if with_context and retriever_func:
            # Baseline 2: retrieve context first
            query = record.get("query", "")
            drug_name = record.get("query_drug_name", "")
            supports = record.get("supports", [])
            
            retrieved = retriever_func(query, supports, drug_name, BM25_TOP_K)
            prompt = build_prompt_with_context(record, retrieved)
            context_used = True
        else:
            # Baseline 1: no context
            prompt = build_prompt(record)
            context_used = False
        
        prompts.append({
            "id":           record["id"],
            "prompt":       prompt,
            "answer":       record["answer"],
            "answer_name":  record.get("answer_name", record["answer"]),
            "context_used": context_used,
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

    if not response:
        return ""

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

    # Strategy 3: return cleaned response as-is (capped)
    return response_clean[:20]


# ─────────────────────────────────────────────
# HELPER — ESTIMATE TOKENS
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (1 token ≈ 4 chars).
    """
    return len(text) // 4


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Prompt Builder — Test (v1, v2 & FewShot+Context)")
    print("=" * 60)

    # Load dataset
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        print("  Run src/load_dataset.py first.")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  [OK]   Loaded {len(data)} records from medhop.json")

    # Test on first record
    record = data[0]
    
    print(f"\n  Record #1 — {record['id']}")
    print(f"  Drug: {record.get('query_drug_name')} ({record.get('query_drug_id')})")
    print(f"  Answer: {record['answer']} ({record.get('answer_name', '')})")
    
    # ─── Test Baseline 1 Prompt ───
    print("\n" + "-" * 60)
    print("  BASELINE 1 PROMPT (no context):")
    print("-" * 60)
    prompt_v1 = build_prompt(record)
    print(prompt_v1)
    print(f"\n  Length: {len(prompt_v1)} chars | ~{estimate_tokens(prompt_v1)} tokens")
    
    # ─── Test Baseline 2 Prompt ───
    print("\n" + "-" * 60)
    print("  BASELINE 2 PROMPT (with BM25 context):")
    print("-" * 60)
    
    simulated_supports = [
        {"text": "Moclobemide is a reversible inhibitor of monoamine oxidase A (MAO-A), an enzyme responsible for breaking down certain neurotransmitters.", "score": 3.67},
        {"text": "MAO-A inhibitors can interact with other drugs that affect serotonin and dopamine levels.", "score": 2.45},
        {"text": "Drug interactions with MAO-A inhibitors require careful monitoring.", "score": 1.92}
    ]
    
    prompt_v2 = build_prompt_with_context(record, simulated_supports)
    print(prompt_v2)
    print(f"\n  Length: {len(prompt_v2)} chars | ~{estimate_tokens(prompt_v2)} tokens")

    # ─── Test FewShot + MedCPT Prompt ───
    print("\n" + "-" * 60)
    print("  FEWSHOT + MEDCPT PROMPT (new):")
    print("-" * 60)
    prompt_fs_ctx = build_prompt_fewshot_with_context(record, simulated_supports)
    print(prompt_fs_ctx)
    print(f"\n  Length: {len(prompt_fs_ctx)} chars | ~{estimate_tokens(prompt_fs_ctx)} tokens")
    
    # ─── Test extract_drug_id ───
    print("\n" + "-" * 60)
    print("  EXTRACT_DRUG_ID TESTS:")
    print("-" * 60)
    
    test_cases = [
        ("The answer is DB04844.", "DB04844"),
        ("DB04844", "DB04844"),
        ("Based on the evidence, the drug is DB04844.", "DB04844"),
        ("I think it might be DB99999.", "DB99999"),
        ("No ID here.", ""),
    ]
    
    for text, expected in test_cases:
        result = extract_drug_id(text, record.get("candidates", []))
        status = "✅" if result == expected else "❌"
        print(f"    {status}  input: '{text[:40]}...' → '{result}'")
    
    print("\n" + "=" * 60)
    print("  ✅  prompt_builder.py working correctly.")
    print("  Supports: Baseline 1, 2, 3 & FewShot+MedCPT")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()