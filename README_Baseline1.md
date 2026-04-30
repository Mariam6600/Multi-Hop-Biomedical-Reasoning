# Baseline 1 — Direct LLM QA (No Retrieval)
## Biomedical Multi-Hop Reasoning on MedHop

### Overview
Evaluates BioMistral-7B-GGUF's raw parametric knowledge on multi-hop drug
interaction questions — **no retrieval, no external context**.

**Pipeline:** `Question → BioMistral-7B → Answer`

---

### Environment
| Component | Details |
|-----------|---------|
| Hardware | Lenovo LOQ, Intel i5-13450HX, 16 GB DDR5 |
| GPU | NVIDIA RTX 5050 8 GB VRAM |
| OS | Windows 11 |
| Python | 3.10.11 |
| LLM | BioMistral-7B-GGUF (Q4_K_M, 4.07 GB) |
| Inference | Ollama 0.13.5 (local) |

---

### Dataset
| Property | Value |
|----------|-------|
| Dataset | QAnGaroo MedHop v1.1 |
| Split | dev.json |
| Total Questions | 342 |
| Unique Answers | 133 |
| Avg Candidates / Question | 8.6 |
| Task | Drug–drug interaction identification |

---

### Prompt Design
You are a biomedical expert specializing in drug interactions.
Task: Identify which drug interacts with the given drug.
Drug in question: {drug_name} ({drug_id})
Choose the correct answer from the following candidates:

{candidate_1} ({id_1})
{candidate_2} ({id_2})  ...
Answer with the DrugBank ID only (format: DBxxxxx).
Answer:


---

### Results — Prompt Ablation Study
| Prompt Style | EM (%) | Correct / 342 |
|-------------|--------|--------------|
| Zero-shot (direct) | **13.74%** | 47 |
| Few-shot | 15.8% | 54 |
| Chain-of-Thought | 14.6% | 50 |
| Few-shot (temp=0) | 14.0% | 48 |

> **Best result: 15.8% EM (few-shot prompt)**
> Random baseline: 11.63% (+4.17pp above random)

---

### Why Exact Match (EM) Only?
MedHop answers are DrugBank IDs (e.g., `DB00563`) — exact strings with
no partial correctness. EM directly measures whether the model selected
the correct drug. F1, BLEU, and ROUGE are inapplicable to this task.
This is also the official metric used in BioCreative IX MedHop Track 2025.

---

### Key Finding
BioMistral-7B without retrieval achieves at most **15.8% EM** — only
**~4pp above random baseline (11.63%)**. Parametric knowledge alone is
insufficient for multi-hop drug interaction QA. This motivates the
retrieval-augmented approaches in subsequent stages.

---

### Project Structure
Code Files/
├── main_baseline1.py          ← Single entry point
├── EnvironmentSetup.py        ← Environment setup & model registration
├── .env.example               ← Template for environment variables
├── config/
│   └── settings.py            ← Central configuration
└── src/
├── load_dataset.py        ← Dataset loading & normalization
├── prompt_builder.py      ← Prompt construction
├── llm_runner.py          ← Ollama API communication
├── inference_pipeline1.py ← Full inference orchestration
└── utils.py               ← Evaluation & reporting

---

### How to Run
```bash
# Setup (first time only)
py -3.10 EnvironmentSetup.py

# Full run (342 questions)
py -3.10 main_baseline1.py

# Quick test (5 questions)
py -3.10 main_baseline1.py --test

# Report only (no re-inference)
py -3.10 main_baseline1.py --report-only

# Reset and start fresh
py -3.10 main_baseline1.py --reset
```

---

### Dependencies
```bash
pip install ollama bm25s PyStemmer scispacy
```
BioMistral-7B.Q4_K_M.gguf must be downloaded separately from HuggingFace.
Ollama must be running: `ollama serve`
