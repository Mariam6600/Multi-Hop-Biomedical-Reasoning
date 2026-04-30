# Ensemble — Majority Vote

### Overview
Combines predictions from the three best-performing pipeline configurations
using 3-way majority voting to improve robustness and reduce individual
model errors.

**Strategy:** For each of the 342 MedHop questions, three independent
configurations cast a "vote" (a predicted DrugBank ID). The final answer
is the ID with 2 or more votes (majority). Ties are broken by alphabetical order.

---

### Input Configurations

The ensemble combines three configurations that independently achieved
the best results in Pipeline 4-5 and Advanced Features stages:

| # | Label | Pipeline | File | Individual EM |
|---|-------|----------|------|---------------|
| 1 | B4-EXP4 (baseline) | Pipeline 4 | `eb_guided_retrieval_k3_qwen3_5-9b_predictions.json` | 33.33% (114/342) |
| 2 | Phase2-Guided | Pipeline 5 | `phase2_guided_retrieval_hybrid_scored_k3_qwen3_5-9b_predictions.json` | 33.33% (114/342) |
| 3 | OV-EXP4 | Advanced Features | `ov_ov-exp4_k3_qwen3_5-9b_predictions.json` | 32.16% (110/342) |

All three use Qwen3.5-9B (local, Ollama) with Hybrid Scored Retrieval (BM25 + MedCPT) at k=3.
The difference lies in the prompt strategy and post-retrieval processing.

---

### Results

| Metric | Value |
|--------|-------|
| **Ensemble EM (Strict)** | **35.38%** |
| Correct | 121 / 342 |
| Failed | 0 |
| **Improvement vs best single** | **+2.05pp** (33.33% → 35.38%) |

### Vote Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Unanimous agreement (3/3) | 241 | 70.5% |
| Split vote (2-1) | 101 | 29.5% |

> 241 out of 342 questions (70.5%) received identical predictions from all
> three configurations. The ensemble mainly helps by resolving disagreements
> on the remaining 101 questions.

---

### Comparison: Single Model vs Ensemble

| Configuration | EM (%) | Correct / 342 |
|---------------|--------|--------------|
| B4-EXP4 guided (single) | 33.33% | 114 |
| Phase2-Guided (single) | 33.33% | 114 |
| OV-EXP4 conservative (single) | 32.16% | 110 |
| **Ensemble majority vote (3-way)** | **35.38%** | **121** |

> The ensemble gains +7 correct answers over each individual configuration,
> demonstrating that different prompt strategies make complementary errors.

---

### Progress Summary — Full Project

| Stage | Model | Retriever | Best EM (%) | Delta |
|-------|-------|-----------|------------|-------|
| Baseline 1 | BioMistral-7B | None | 15.8% | -- |
| Baseline 2 | BioMistral-7B | BM25 (k=10) | 12.3% | -3.5pp |
| Baseline 3 | BioMistral-7B | Dense MedCPT (k=5) | 16.4% | +0.6pp |
| Pipeline 4-5 | Qwen3.5-9B | Hybrid Scored (k=3) | 33.3% | +16.9pp |
| Advanced Features | Qwen3.5-9B | Hybrid + Ontology | 32.2% | -1.1pp |
| **Ensemble** | **Qwen3.5-9B** | **Majority Vote (3-way)** | **35.4%** | **+2.1pp** |

---

### Why Exact Match (EM) Only?
MedHop answers are DrugBank IDs (e.g., DB00563) — exact strings with no partial
correctness. EM is the official BioCreative IX MedHop Track 2025 metric.
F1, BLEU, and ROUGE are inapplicable to this closed-candidate selection task.

---

### How to Run
```bash
# Run ensemble majority vote
py -3.10 src/ensemble_majority_vote.py

# Evaluate all outputs (generates evaluation_results.txt / .json)
py -3.10 src/evaluate_all.py
```

### Dependencies
Same as Pipeline 4-5 requirements. No additional packages needed.
