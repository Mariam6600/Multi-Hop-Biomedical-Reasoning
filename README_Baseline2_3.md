# Baselines 2 & 3 — Retrieval-Augmented QA (BM25 + Dense)

### Overview
Introduces retrieval to provide external evidence before answering.
Baseline 2 uses sparse BM25 retrieval.
Baseline 3 uses dense semantic retrieval via MedCPT embeddings.
Both run on the full MedHop dev set (342 questions) using BioMistral-7B.

**Pipeline:**
`Question → Query Expander → Retriever (top-K docs) → BioMistral-7B → Answer`

---

### Key Components
| File | Role |
|------|------|
| `src/retriever.py` | BM25 sparse retrieval (Baseline 2) |
| `src/retriever_expanded.py` | Expanded query dense retrieval (Baseline 3) |
| `src/query_expander.py` | Drug synonym and name expansion |
| `src/inference_pipeline2.py` | Baseline 2 full pipeline |
| `src/inference_pipeline3.py` | Baseline 3 full pipeline |
| `src/inference_pipeline3-1_hybrid.PY` | Hybrid BM25+dense variant |
| `src/Inference_pipeline3_fewshot.py` | Baseline 3 with few-shot prompting |

---

### Results — Top-K Ablation Study

#### Baseline 2 — BM25 Sparse Retrieval (BioMistral-7B)
| K | Prompt | EM (%) | Correct / 342 |
|---|--------|--------|--------------|
| 3  | zero-shot | 11.4% | 39 |
| 5  | zero-shot | 11.7% | 40 |
| 10 | zero-shot | **12.3%** | 42 |
| 20 | zero-shot | 6.1% | 21 |

> BM25 best: **12.3% at k=10** — context noise at k=20 hurts performance.

#### Baseline 3 — Dense Retrieval / MedCPT (BioMistral-7B)
| K | Prompt | EM (%) | Correct / 342 |
|---|--------|--------|--------------|
| 5  | zero-shot | **16.4%** | 56 |
| 5  | few-shot  | 12.9% | 44 |
| 10 | zero-shot | 9.9% | 34 |
| 20 | zero-shot | 8.8% | 30 |
| 5  | hybrid    | 13.2% | 45 |

> Dense best: **16.4% at k=5** — dense retrieval outperforms BM25 by +4.1pp.

---

### Comparison Against Baseline 1
| Stage | Best EM (%) | Δ vs Baseline 1 |
|-------|------------|-----------------|
| Baseline 1 (no retrieval) | 15.8% | — |
| Baseline 2 (BM25)         | 12.3% | −3.5pp |
| Baseline 3 (Dense MedCPT) | **16.4%** | **+0.6pp** |

> BM25 retrieval with BioMistral-7B hurts performance — the model struggles
> to extract relevant information from retrieved passages. Dense retrieval
> marginally improves over no retrieval, motivating the switch to a stronger
> local LLM (Qwen3.5-9B) in the next stage.

---

### Why Exact Match (EM) Only?
MedHop answers are DrugBank IDs — exact strings. EM is the official
BioCreative IX MedHop Track 2025 metric. No partial correctness exists.

---

### How to Run
```bash
# Baseline 2 — BM25
py -3.10 src/inference_pipeline2.py

# Baseline 3 — Dense
py -3.10 src/inference_pipeline3.py

# Baseline 3 — Hybrid variant
py -3.10 "src/inference_pipeline3-1_hybrid.PY"

# Baseline 3 — Few-shot
py -3.10 src/Inference_pipeline3_fewshot.py
```
