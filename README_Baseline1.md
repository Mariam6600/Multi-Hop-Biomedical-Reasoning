# Baseline 1 — Direct LLM QA
## Biomedical Multi-Hop QA Project

### Overview
Baseline 1 evaluates BioMistral-7B-GGUF's raw reasoning capability without any retrieval or decomposition.

**Pipeline:** Question → BioMistral → Answer

### Environment
| Component | Details |
|-----------|---------|
| Hardware | Lenovo LOQ, Intel i5-13450HX, 16GB DDR5 |
| GPU | NVIDIA RTX 5050 8GB VRAM |
| OS | Windows 11 |
| Python | 3.10.11 |
| LLM | BioMistral-7B-GGUF (Q4_K_M, 4.07GB) |
| Inference | Ollama 0.13.5 |

### Dataset
| Property | Value |
|----------|-------|
| Dataset | QAnGaroo MedHop v1.1 |
| Split | dev.json |
| Total Questions | 342 |
| Unique Answers | 133 |
| Avg Candidates/Q | 8.6 |
| Task | Drug-drug interaction identification |

### Project Structure
```
Code Files/
├── main_baseline1.py          ← Single entry point
├── EnvironmentSetup.py        ← Environment setup & model registration
├── config/
│   └── settings.py            ← Central configuration
├── src/
│   ├── load_dataset.py        ← Dataset loading & normalization
│   ├── prompt_builder.py      ← Prompt construction
│   ├── llm_runner.py          ← Ollama API communication
│   ├── inference_pipeline.py  ← Full inference orchestration
│   └── utils.py               ← Evaluation & reporting
└── data/
    └── medhop.json            ← Normalized dataset (generated)
```

### Prompt Design
```
You are a biomedical expert specializing in drug interactions.
Task: Identify which drug interacts with the given drug.
Drug in question: {drug_name} ({drug_id})
Choose the correct answer from the following candidates:
- {candidate_1} ({id_1})
- {candidate_2} ({id_2})
...
Answer with the DrugBank ID only (format: DBxxxxx).
Answer:
```

### Results
| Metric | Score |
|--------|-------|
| **Exact Match (EM)** | **13.74%** |
| Random Baseline | 11.63% |
| Above Random | +2.11% |
| Total Questions | 342 |
| Correct | 47 |
| Wrong | 295 |
| Failed | 0 |
| Avg Inference Time | 0.42 sec/question |
| Total Runtime | 2.4 minutes |

### Error Analysis
| Metric | Value |
|--------|-------|
| Valid but wrong predictions | 240 (81.4%) |
| Top missed answer | DB00850, DB00946, DB08820 (7× each) |

### Key Finding
> BioMistral-7B without retrieval achieves **13.74% EM** — only **2.11% above random baseline (11.63%)** — demonstrating that parametric knowledge alone is insufficient for multi-hop biomedical drug interaction QA. This establishes the motivation for retrieval-augmented approaches in subsequent stages.

### How to Run
```bash
# Full run (342 questions)
py -3.10 main_baseline1.py

# Quick test (5 questions)
py -3.10 main_baseline1.py --test

# Report only (no inference)
py -3.10 main_baseline1.py --report-only

# Reset and start fresh
py -3.10 main_baseline1.py --reset
```

### Dependencies
```bash
pip install bm25s PyStemmer scispacy langchain ollama datasets
```

### Notes
- BioMistral-7B.Q4_K_M.gguf must be downloaded separately from HuggingFace
- Ollama must be running before executing any script: `ollama serve`
- Model registered in Ollama as: `biomistral-7b`
