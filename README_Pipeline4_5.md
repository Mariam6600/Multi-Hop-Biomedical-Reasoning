# Pipeline 4 & 5 — Hybrid Scored Retrieval + Qwen3.5-9B

### Overview
Upgrades two dimensions simultaneously:
1. **Retriever**: BM25 + MedCPT hybrid with score fusion (RRF & weighted)
2. **LLM**: Switches from BioMistral-7B to **Qwen3.5-9B** (local, Ollama)

This stage yields the **best local results** in the project: **33.3% EM**.

**Pipeline:**
Question
→ Structured Query Expander
→ Hybrid Retriever (BM25 + MedCPT, score fusion)
→ [optional] Reranker
→ Qwen3.5-9B (local, Ollama)
→ Answer

---

### Key Components
| File | Role |
|------|------|
| `src/retriever_hybrid_scored.py` | BM25 + dense hybrid with score fusion |
| `src/retriever_combined.py` | Combined retrieval strategies |
| `src/retriever_semantic.py` | Pure cosine semantic retrieval |
| `src/retriever_weighted_struct.py` | Weighted structural retrieval |
| `src/retriever_adaptive_k.py` | Dynamic K selection per question |
| `src/query_expander_structured.py` | Drug-aware structured query expansion |
| `src/reranker.py` | Cross-encoder reranking |
| `src/gold_chain_retriever.py` | Oracle retrieval (upper-bound reference) |
| `src/kb_builder.py` | Knowledge base construction from DrugBank |
| `src/score_diagnostics.py` | Score distribution analysis |
| `src/inference_pipeline4.py` | Phase 1 — retriever development experiments |
| `src/inference_pipeline5.py` | Phase 2 — guided retrieval experiments |

---

### Results — Phase 1: Retriever Development (Qwen3.5-9B)

#### Hybrid Scored Retriever
| Experiment | K | Prompt | EM (%) |
|-----------|---|--------|--------|
| hybrid_scored | 3 | direct | 22.2% |
| hybrid_scored | 3 | **guided** | **33.3%** ← Best |
| hybrid_scored | 3 | few-shot | 12.0% |
| hybrid_scored | 5 | direct | 17.2% |

#### Other Retrievers (k=3, direct prompt, Qwen3.5-9B)
| Retriever | EM (%) |
|-----------|--------|
| Combined  | 20.2% |
| Expanded  | 19.6% |
| Weighted Struct | 18.1% |
| Query Drug | 16.4% |
| Expanded k=5 | 15.2% |
| Weighted Struct k=5 | 11.1% |

> Hybrid Scored + Guided Prompt at k=3 is the clear winner.

---

### Results — Phase 2: Guided Retrieval Variants (Hybrid Scored k=3)

| Variant | EM (%) | Notes |
|---------|--------|-------|
| **guided_retrieval** | **33.3%** | Best — candidate-aware guidance |
| cot_enriched | 32.2% | Chain-of-thought enriched |
| reranked | 24.9% | With cross-encoder reranking |
| enriched | 24.3% | Context-enriched prompt |
| guided_retrieval k=5 | 26.6% | More docs hurts |
| dual_retrieval | 17.2% | Two-pass retrieval |
| forced_reasoning | 16.7% | Explicit reasoning steps |

> Adding a cross-encoder reranker (24.9%) does NOT improve over guided prompt
> alone (33.3%) — the retriever quality is the bottleneck, not ranking.

---

### Oracle Upper Bound (NOT a production result)
| Gold Chain | K | EM (%) | Notes |
|-----------|---|--------|-------|
| Oracle retrieval | 3 | 76.9% | Upper bound — uses true answer path |
| Oracle retrieval | 5 | 67.0% | Upper bound — uses true answer path |

> Oracle shows a 43.6pp gap — significant room for retrieval improvement.
> Included for reference only; not comparable to real system performance.

---

### Progress Summary
| Stage | Best EM (%) | Δ vs Previous |
|-------|------------|---------------|
| Baseline 1 (BioMistral, no RAG) | 15.8% | — |
| Baseline 2 (BM25) | 12.3% | −3.5pp |
| Baseline 3 (Dense MedCPT) | 16.4% | +0.6pp |
| **Pipeline 4-5 (Hybrid + Qwen3.5-9B)** | **33.3%** | **+16.9pp** |

> Switching to Qwen3.5-9B + hybrid retrieval + guided prompting
> delivers a **+16.9pp improvement** — the largest single gain in the project.

---

### Why Exact Match (EM) Only?
MedHop answers are DrugBank IDs — exact strings with no partial correctness.
EM is the official BioCreative IX MedHop Track 2025 metric.

---

### How to Run
```bash
# Phase 1 — Retriever development
py -3.10 src/inference_pipeline4.py

# Phase 2 — Guided retrieval variants
py -3.10 src/inference_pipeline5.py

# Score diagnostics
py -3.10 src/score_diagnostics.py

# Evaluate all outputs
py -3.10 src/evaluate_all.py
```
