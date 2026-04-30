"""
config/settings.py
==================
Biomedical Multi-Hop QA Project
يدعم: ollama (محلي) | huggingface | groq | openrouter

النماذج المحلية المتوفرة في models/:
  - qwen2.5-7b   ← Qwen2.5-7B-Instruct-Q4_K_M.gguf   (القديم)
  - qwen3.5-9b   ← Qwen3.5-9B-Q4_K_M_2.gguf           (الجديد)

للتبديل بين النماذج المحلية: غيّر OLLAMA_ACTIVE_MODEL فقط

الإعداد الأمني:
  API keys محفوظة في ملف .env (لا يُرفع على GitHub)
  أنشئ ملف .env من .env.example وضع مفاتيحك الحقيقية فيه
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# LOAD .env FILE (python-dotenv)
# ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=r"C:\Users\LOQ\Desktop\Graduation Project2\Code Files\.env")
except ImportError:
    print("[WARN] python-dotenv not installed. Run: pip install python-dotenv")

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONFIG_DIR  = os.path.join(PROJECT_ROOT, "config")
SRC_DIR     = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")

# ─────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────

MEDHOP_TRAIN   = os.path.join(DATA_DIR, "qangaroo_v1.1", "qangaroo_v1.1", "medhop", "train.json")
MEDHOP_DEV     = os.path.join(DATA_DIR, "qangaroo_v1.1", "qangaroo_v1.1", "medhop", "dev.json")
ACTIVE_DATASET = MEDHOP_DEV

MEDHOP_FILE    = os.path.join(DATA_DIR, "medhop.json")
DRUGBANK_VOCAB = os.path.join(DATA_DIR, "drugbank_all_drugbank_vocabulary.csv", "drugbank vocabulary.csv")

# ─────────────────────────────────────────────
# LOCAL MODELS (Ollama)
# ─────────────────────────────────────────────

OLLAMA_ACTIVE_MODEL = "qwen3.5-9b"   # <- "qwen2.5-7b" | "qwen3.5-9b"

OLLAMA_MODELS = {
    "qwen2.5-7b": {
        "name":      "qwen2.5-7b",
        "modelfile": "Modelfile-qwen",
        "gguf":      "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    },
    "qwen3.5-9b": {
        "name":      "qwen3.5-9b",
        "modelfile": "Modelfile-qwen3.5",
        "gguf":      "Qwen3.5-9B-Q4_K_M_2.gguf",
    },
}

OLLAMA_HOST       = "http://localhost:11434"
OLLAMA_MODEL_NAME = OLLAMA_MODELS[OLLAMA_ACTIVE_MODEL]["name"]
OLLAMA_MODEL      = OLLAMA_MODEL_NAME

LLM_TEMPERATURE = 0.1
LLM_TOP_P       = 0.9
LLM_TOP_K       = 40
LLM_MAX_TOKENS  = 200
LLM_NUM_CTX     = 4096

# ─────────────────────────────────────────────
# API PROVIDERS — keys loaded from .env
# ─────────────────────────────────────────────

# المفاتيح تُقرأ من .env — لا تكتبها هنا أبداً
HF_API_KEY         = os.getenv("HF_API_KEY", "")
HF_MODEL           = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"  #qwen/qwen3-32b  *llama-3.3-70b-versatile *deepseek-r1-distill-llama-70b

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = "openrouter/free"  #qwen/qwen3.6-flash *nvidia/nemotron-3-super-120b-a12b:free *openai/gpt-oss-120b:free

ACTIVE_PROVIDER = "openrouter"   # <- "ollama" | "huggingface" | "groq" | "openrouter"

if ACTIVE_PROVIDER == "groq":
    API_KEY   = GROQ_API_KEY
    API_MODEL = GROQ_MODEL
    API_URL   = "https://api.groq.com/openai/v1/chat/completions"
elif ACTIVE_PROVIDER == "huggingface":
    API_KEY   = HF_API_KEY
    API_MODEL = HF_MODEL
    API_URL   = "https://router.huggingface.co/v1/chat/completions"
elif ACTIVE_PROVIDER == "openrouter":
    API_KEY   = OPENROUTER_API_KEY
    API_MODEL = OPENROUTER_MODEL
    API_URL   = "https://openrouter.ai/api/v1/chat/completions"
else:  # ollama
    API_KEY   = "local"
    API_MODEL = OLLAMA_MODEL_NAME
    API_URL   = f"{OLLAMA_HOST}/api/chat"

# ─────────────────────────────────────────────
# PIPELINE SETTINGS
# ─────────────────────────────────────────────

EXPANSION_N_TERMS = 30
DIAG_SAMPLE_SIZE  = None
MAX_QUESTIONS     = None
LOG_EVERY_N       = 10

# ─────────────────────────────────────────────
# BASELINE 1 OUTPUT FILES
# ─────────────────────────────────────────────

EXP_NAME         = "fewshot"
PREDICTIONS_FILE = os.path.join(OUTPUTS_DIR, f"baseline1_{EXP_NAME}_predictions.json")
LOGS_FILE        = os.path.join(OUTPUTS_DIR, f"baseline1_{EXP_NAME}_logs.json")

# Baseline 1 API
PREDICTIONS_FILE_B1_API = os.path.join(OUTPUTS_DIR, "baseline1_api_original_predictions.json")
LOGS_FILE_B1_API        = os.path.join(OUTPUTS_DIR, "baseline1_api_original_logs.json")
B1_API_EXP_NAME         = "baseline1_api_original"

# ─────────────────────────────────────────────
# BASELINE 2 — BM25
# ─────────────────────────────────────────────

BM25_TOP_K          = 5
BM25_METHOD         = "lucene"
PREDICTIONS_FILE_B2 = os.path.join(OUTPUTS_DIR, "baseline2_k5_predictions.json")
LOGS_FILE_B2        = os.path.join(OUTPUTS_DIR, "baseline2_k5_logs.json")

# ─────────────────────────────────────────────
# BASELINE 3 — MedCPT SEMANTIC RETRIEVAL
# ─────────────────────────────────────────────

MEDCPT_QUERY_ENCODER        = os.path.join(MODELS_DIR, "MedCPT-Query-Encoder")
MEDCPT_ARTICLE_ENCODER      = "ncbi/MedCPT-Article-Encoder"
MEDCPT_TOP_K                = 5
MEDCPT_BATCH_SIZE           = 32
PREDICTIONS_FILE_B3         = os.path.join(OUTPUTS_DIR, "baseline3_k5_predictions.json")
LOGS_FILE_B3                = os.path.join(OUTPUTS_DIR, "baseline3_k5_logs.json")
PREDICTIONS_FILE_B3_FEWSHOT = os.path.join(OUTPUTS_DIR, "baseline3_fewshot_k5_predictions.json")
LOGS_FILE_B3_FEWSHOT        = os.path.join(OUTPUTS_DIR, "baseline3_fewshot_k5_logs.json")

# ─────────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────────

HYBRID_EXP_NAME         = "weighted_70_30_k5"
HYBRID_TOP_K            = 5
HYBRID_CANDIDATES       = 20
RRF_K_CONSTANT          = 60
MEDCPT_WEIGHT           = 0.70
BM25_WEIGHT             = 0.30
PREDICTIONS_FILE_HYBRID = os.path.join(OUTPUTS_DIR, f"hybrid_{HYBRID_EXP_NAME}_predictions.json")
LOGS_FILE_HYBRID        = os.path.join(OUTPUTS_DIR, f"hybrid_{HYBRID_EXP_NAME}_logs.json")

# ─────────────────────────────────────────────
# STAGE 4 — ENTITY BRIDGING
# ─────────────────────────────────────────────

STAGE4_BRIDGE_TOP_K     = 5
STAGE4_FINAL_TOP_K      = 8
PREDICTIONS_FILE_STAGE4 = os.path.join(OUTPUTS_DIR, "stage4_entity_bridge_predictions.json")
LOGS_FILE_STAGE4        = os.path.join(OUTPUTS_DIR, "stage4_entity_bridge_logs.json")

# ─────────────────────────────────────────────
# EMBEDDINGS CACHE
# ─────────────────────────────────────────────

EMBEDDINGS_CACHE_DIR = os.path.join(OUTPUTS_DIR, "embeddings_cache")


def verify_settings():
    print("\n" + "=" * 60)
    print("  Project Settings")
    print("=" * 60)
    print(f"  ACTIVE_PROVIDER  : {ACTIVE_PROVIDER}")
    if ACTIVE_PROVIDER == "ollama":
        print(f"  OLLAMA_ACTIVE    : {OLLAMA_ACTIVE_MODEL}")
        print(f"  OLLAMA_MODEL     : {OLLAMA_MODEL_NAME}")
        info = OLLAMA_MODELS[OLLAMA_ACTIVE_MODEL]
        print(f"  GGUF file        : {info['gguf']}")
        print(f"  Modelfile        : {info['modelfile']}")
        print(f"  OLLAMA_HOST      : {OLLAMA_HOST}")
    else:
        print(f"  API_MODEL        : {API_MODEL}")
        print(f"  API_URL          : {API_URL}")
        key_set = "SET" if (API_KEY and len(API_KEY) > 10) else "NOT SET"
        print(f"  API_KEY          : {key_set}")
    print(f"  MEDHOP_FILE      : {MEDHOP_FILE}")
    for name, path in [("DATA_DIR", DATA_DIR), ("OUTPUTS_DIR", OUTPUTS_DIR)]:
        exists = os.path.exists(path)
        print(f"  {'OK' if exists else 'MISSING'} {name}")
    print("=" * 60)


if __name__ == "__main__":
    verify_settings()