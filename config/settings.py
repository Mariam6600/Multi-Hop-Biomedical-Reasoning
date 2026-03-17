"""
config/settings.py
===================
Biomedical Multi-Hop QA Project — Baseline 1
Central configuration file.

All other modules import their settings from here.
To change any setting, edit ONLY this file.
"""

import os

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────

# Root of the project (Code Files folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subfolders
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
SRC_DIR     = os.path.join(PROJECT_ROOT, "src")

# ─────────────────────────────────────────────
# DATASET PATHS (actual paths on this machine)
# ─────────────────────────────────────────────

# Root of the QAnGaroo dataset
QANGAROO_DIR = os.path.join(DATA_DIR, "qangaroo_v1.1", "qangaroo_v1.1")

# MedHop folder
MEDHOP_DIR = os.path.join(QANGAROO_DIR, "medhop")

# MedHop split files
MEDHOP_TRAIN        = os.path.join(MEDHOP_DIR, "train.json")
MEDHOP_DEV          = os.path.join(MEDHOP_DIR, "dev.json")
MEDHOP_TRAIN_MASKED = os.path.join(MEDHOP_DIR, "train.masked.json")
MEDHOP_DEV_MASKED   = os.path.join(MEDHOP_DIR, "dev.masked.json")

# Active split used in Baseline 1
# Options: MEDHOP_TRAIN | MEDHOP_DEV | MEDHOP_TRAIN_MASKED | MEDHOP_DEV_MASKED
ACTIVE_DATASET = MEDHOP_DEV

# DrugBank vocabulary (drug ID -> drug name mapping)
DRUGBANK_VOCAB = os.path.join(
    DATA_DIR,
    "drugbank_all_drugbank_vocabulary.csv",
    "drugbank vocabulary.csv"
)

# Output JSON (processed dataset used by the pipeline)
MEDHOP_FILE = os.path.join(DATA_DIR, "medhop.json")

# Number of questions to process (None = all questions)
# Set a small number like 10 for quick testing
MAX_QUESTIONS = None

# ─────────────────────────────────────────────
# LLM SETTINGS
# ─────────────────────────────────────────────

# Ollama model name (registered in EnvironmentSetup.py)
OLLAMA_MODEL = "biomistral-7b"

# Ollama API base URL
OLLAMA_HOST = "http://localhost:11434"

# Generation parameters
LLM_TEMPERATURE = 0.1   # Low = more focused, factual answers
LLM_TOP_P       = 0.9
LLM_TOP_K       = 40
LLM_MAX_TOKENS  = 200   # Short answers for QA task
LLM_NUM_CTX     = 2048  # Context window size

# ─────────────────────────────────────────────
# PROMPT SETTINGS
# ─────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a biomedical expert.

Question: {question}

Provide a short factual answer in 1-2 sentences."""

# ─────────────────────────────────────────────
# OUTPUT FILES
# ─────────────────────────────────────────────

PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, "baseline1_predictions.json")
LOGS_FILE        = os.path.join(OUTPUTS_DIR, "baseline1_logs.json")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

LOG_EVERY_N = 10   # Print progress every N questions

# ─────────────────────────────────────────────
# SANITY CHECK (runs when you import this file)
# ─────────────────────────────────────────────

def verify_paths():
    """Check all required folders exist."""
    required = [DATA_DIR, OUTPUTS_DIR, MODELS_DIR, SRC_DIR]
    all_ok = True
    for path in required:
        if not os.path.exists(path):
            print(f"[settings] WARNING: folder not found: {path}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    # Quick test: print all settings
    print("=" * 50)
    print("  Project Settings — Baseline 1")
    print("=" * 50)
    print(f"  PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"  DATA_DIR         : {DATA_DIR}")
    print(f"  OUTPUTS_DIR      : {OUTPUTS_DIR}")
    print(f"  MODELS_DIR       : {MODELS_DIR}")
    print()
    print(f"  MEDHOP_TRAIN     : {MEDHOP_TRAIN}")
    print(f"  MEDHOP_DEV       : {MEDHOP_DEV}")
    print(f"  ACTIVE_DATASET   : {ACTIVE_DATASET}")
    print(f"  DRUGBANK_VOCAB   : {DRUGBANK_VOCAB}")
    print(f"  MEDHOP_FILE      : {MEDHOP_FILE}")
    print()
    print(f"  OLLAMA_MODEL     : {OLLAMA_MODEL}")
    print(f"  OLLAMA_HOST      : {OLLAMA_HOST}")
    print(f"  PREDICTIONS      : {PREDICTIONS_FILE}")
    print(f"  LOGS_FILE        : {LOGS_FILE}")
    print()
    ok = verify_paths()
    if ok:
        print("  ✅ All paths verified.")
    else:
        print("  ⚠️  Some folders missing — run EnvironmentSetup.py first.")
    print("=" * 50)
