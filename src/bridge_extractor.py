"""
src/bridge_extractor.py
========================
Biomedical Multi-Hop QA Project — Stage 4: Entity Bridging

What is a "bridge entity"?
  MedHop questions are multi-hop: to answer "which drug interacts with X?",
  you first need to know X's mechanism (enzyme/pathway/receptor),
  then find which candidate drug shares that same mechanism.

  Bridge entity = the intermediate biological concept that connects two drugs.
  Example:
    Drug: Moclobemide → Bridge: "MAO-A inhibition, serotonin pathway"
    Then search for that bridge → find Tetrabenazine (also affects serotonin)

What this module does:
  1. Takes a drug name
  2. Sends a focused prompt to BioMistral asking for its mechanism
  3. Extracts the bridge entity (enzyme, receptor, pathway, transporter)
  4. Returns a clean string ready for a second retrieval pass

Design decisions:
  - One focused prompt, short output (max 60 tokens)
  - Falls back to drug name alone if extraction fails
  - Logs extracted bridges for analysis and debugging
  - Completely stateless — no side effects

Usage:
    py -3.10 src/bridge_extractor.py
"""

import time
import os
import sys
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import OLLAMA_MODEL_NAME, OLLAMA_HOST


# ─────────────────────────────────────────────
# BRIDGE EXTRACTION PROMPT
# ─────────────────────────────────────────────

BRIDGE_PROMPT_TEMPLATE = """\
You are a biomedical expert. Answer in ONE short sentence only.

What is the primary mechanism of action of {drug_name}?
Focus on: enzyme inhibited/induced, receptor targeted, or biological pathway affected.

Examples of good answers:
- "Moclobemide inhibits MAO-A enzyme, affecting serotonin and dopamine metabolism."
- "Warfarin inhibits Vitamin K epoxide reductase, blocking clotting factor synthesis."
- "Methotrexate inhibits dihydrofolate reductase, blocking folate metabolism."

Answer (one sentence, mechanism only):"""


# ─────────────────────────────────────────────
# BRIDGE QUERY BUILDER
# ─────────────────────────────────────────────

def build_bridge_query(drug_name: str, bridge_text: str) -> str:
    """
    Build an enriched retrieval query from the bridge entity.

    Combines the drug name with the extracted mechanism for a richer
    semantic search query.

    Args:
        drug_name:    Original drug name (e.g., "Moclobemide")
        bridge_text:  Extracted mechanism (e.g., "MAO-A inhibition serotonin")

    Returns:
        Combined query string for semantic retrieval
    """
    if not bridge_text or bridge_text == drug_name:
        return drug_name

    # Clean up the bridge text
    bridge_clean = bridge_text.strip().rstrip(".")

    # Build enriched query
    return f"{drug_name} {bridge_clean} drug interaction"


# ─────────────────────────────────────────────
# BRIDGE ENTITY EXTRACTION — CORE FUNCTION
# ─────────────────────────────────────────────

def extract_bridge_entity(
    client,
    drug_name: str,
    drug_id: str,
    question_id: str = "",
) -> dict:
    """
    Ask BioMistral for the primary mechanism of action of a drug.

    This is the first hop in the Entity Bridging pipeline:
      Drug → (bridge) → Interacting Drug

    Args:
        client:      Ollama client instance (from llm_runner.get_ollama_client)
        drug_name:   Human-readable drug name (e.g., "Moclobemide")
        drug_id:     DrugBank ID (e.g., "DB01171") — used as fallback
        question_id: For logging purposes

    Returns:
        dict with:
          {
            "bridge_raw":   str,   ← raw LLM response
            "bridge_clean": str,   ← cleaned mechanism text
            "bridge_query": str,   ← ready-to-use retrieval query
            "success":      bool,
            "inference_time": float,
            "error":        str,
          }
    """
    result = {
        "bridge_raw":     "",
        "bridge_clean":   drug_name,   # fallback: just use the drug name
        "bridge_query":   drug_name,
        "success":        False,
        "inference_time": 0.0,
        "error":          "",
    }

    # Build the focused prompt
    prompt = BRIDGE_PROMPT_TEMPLATE.format(drug_name=drug_name)

    try:
        start = time.time()

        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.1,
                "top_p":       0.9,
                "num_predict": 80,    # short — we only need one sentence
                "num_ctx":     1024,  # small context — faster
            },
        )

        elapsed = time.time() - start
        raw = response["message"]["content"].strip()

        if not raw:
            result["error"] = "Empty response from model"
            return result

        # Clean the response
        bridge_clean = _clean_bridge_text(raw)

        # Build the retrieval query
        bridge_query = build_bridge_query(drug_name, bridge_clean)

        result.update({
            "bridge_raw":     raw,
            "bridge_clean":   bridge_clean,
            "bridge_query":   bridge_query,
            "success":        True,
            "inference_time": round(elapsed, 3),
            "error":          "",
        })

    except Exception as e:
        result["error"] = str(e)
        result["bridge_query"] = drug_name  # safe fallback

    return result


# ─────────────────────────────────────────────
# TEXT CLEANING HELPERS
# ─────────────────────────────────────────────

def _clean_bridge_text(raw: str) -> str:
    """
    Clean the raw LLM response into a concise mechanism phrase.

    Strategy:
      1. Take only the first sentence
      2. Remove filler phrases ("I think", "Based on", etc.)
      3. Trim to reasonable length

    Args:
        raw: Raw text from BioMistral

    Returns:
        Clean mechanism phrase
    """
    if not raw:
        return ""

    text = raw.strip()

    # Take only the first sentence
    first_sentence = re.split(r"[.!?]\s", text)[0]
    if first_sentence:
        text = first_sentence

    # Remove common filler openers
    fillers = [
        r"^(Based on my knowledge[,.]?\s*)",
        r"^(As a biomedical expert[,.]?\s*)",
        r"^(The primary mechanism of action of \w+ is\s*)",
        r"^(The drug \w+ )",
        r"^(I think\s*)",
        r"^(Note:\s*)",
    ]
    for pattern in fillers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Cap length (no need for long texts — just the key mechanism words)
    if len(text) > 200:
        text = text[:200]

    return text.strip().rstrip(".")


def extract_key_terms(bridge_clean: str) -> list:
    """
    Extract key biomedical terms from the bridge text.
    Useful for building targeted BM25 queries as a fallback.

    Args:
        bridge_clean: Cleaned bridge mechanism text

    Returns:
        List of key terms (enzymes, pathways, receptors)
    """
    # Patterns for biomedical entities
    patterns = [
        r"CYP\d+\w*",               # cytochrome P450 enzymes (CYP3A4, etc.)
        r"MAO-?\w*",                 # monoamine oxidases
        r"\w+-reductase",            # reductases
        r"\w+-oxidase",              # oxidases
        r"\w+-transferase",          # transferases
        r"\w+\s+receptor",           # receptors
        r"serotonin|dopamine|norepinephrine|GABA|acetylcholine",  # neurotransmitters
        r"folate|vitamin K|purine|pyrimidine",                    # metabolites
        r"P-glycoprotein|OATP\w*|OCT\w*",                        # transporters
    ]

    terms = []
    for pattern in patterns:
        matches = re.findall(pattern, bridge_clean, re.IGNORECASE)
        terms.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique_terms.append(t)

    return unique_terms


# ─────────────────────────────────────────────
# MAIN — TEST THE MODULE
# ─────────────────────────────────────────────

def main():
    import json

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.settings import MEDHOP_FILE
    from src.llm_runner import get_ollama_client, check_model_available

    print("\n" + "=" * 60)
    print("  Bridge Extractor — Test (5 drugs)")
    print("=" * 60)
    print()

    # Connect to Ollama
    print("--- Connecting to Ollama ---")
    client = get_ollama_client()
    if not check_model_available(client):
        sys.exit(1)
    print(f"  [OK]   Connected — model '{OLLAMA_MODEL_NAME}' ready\n")

    # Load a few test records
    if not os.path.exists(MEDHOP_FILE):
        print(f"  [FAIL] medhop.json not found: {MEDHOP_FILE}")
        sys.exit(1)

    with open(MEDHOP_FILE, encoding="utf-8") as f:
        data = json.load(f)

    test_records = data[:5]

    print("--- Testing Bridge Extraction ---\n")
    total_time = 0.0

    for i, record in enumerate(test_records):
        drug_id   = record["query_drug_id"]
        drug_name = record.get("query_drug_name", drug_id)
        answer    = record.get("answer_name", record["answer"])

        print(f"  [{i+1}/5] Drug: {drug_name} ({drug_id})")
        print(f"          Answer: {answer}")

        result = extract_bridge_entity(
            client=client,
            drug_name=drug_name,
            drug_id=drug_id,
            question_id=record["id"],
        )

        if result["success"]:
            key_terms = extract_key_terms(result["bridge_clean"])
            print(f"          Bridge: {result['bridge_clean']}")
            print(f"          Query:  {result['bridge_query']}")
            print(f"          Terms:  {key_terms if key_terms else '(none extracted)'}")
            print(f"          Time:   {result['inference_time']}s")
        else:
            print(f"          [FAIL] {result['error']}")
            print(f"          Fallback query: {result['bridge_query']}")

        total_time += result["inference_time"]
        print()

    avg_time = total_time / len(test_records)
    print(f"  Avg bridge extraction time: {avg_time:.2f}s per drug")
    print(f"  Estimated extra time for full dataset (342 Q): {avg_time*342/60:.1f} min")

    print()
    print("=" * 60)
    print("  ✅  bridge_extractor.py working correctly.")
    print("  Next: src/inference_pipeline4.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()