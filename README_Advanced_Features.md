# Advanced Features — Decomposition, Ontology, Entity Bridging, Adaptive

### Overview
Four independent augmentation modules built on top of the hybrid pipeline
(Pipeline 4-5 best config: Hybrid Scored k=3 + Qwen3.5-9B).

| Module | Hypothesis |
|--------|-----------|
| **Query Decomposition** | Split multi-hop question into sequential sub-queries |
| **Ontology Verification** | Validate/filter candidates using biomedical ontology signals |
| **Entity Bridging** | Find intermediate bridge entities between drug pairs |
| **Adaptive Retrieval** | Dynamically select retrieval strategy per question type |

---

### Architecture
Multi-Hop Question (MedHop dev, 342 questions)
│
┌─────┴──────────────────────────────┐
▼           ▼           ▼            ▼
Query        Ontology   Entity       Adaptive
Decomp       Verify     Bridge       Retrieval
(qd_)       (ov_)     (eb_)       (adaptive_)
└─────┬──────────────────────────────┘
▼
Qwen3.5-9B (local, Ollama)
▼
Final Answer

---

### Module 1 — Query Decomposition
Decomposes multi-hop drug interaction questions into sub-queries,
retrieves evidence for each hop separately, then fuses for final answer.

| Experiment | Strategy | EM (%) |
|-----------|---------|--------|
| qd-exp1 (qd_ircot) | IRCoT-style iterative | 13.7% |
| qd-exp2 (qd_hop2only) | 2nd hop only | 23.7% |
| qd-exp3 (qd_per_cand) | Per-candidate reasoning | 24.0% |
| qd-exp4a (qd_candfil) | Candidate filtering | 27.2% |
| qd-exp5 (qd_guided) | Guided decomposition | **28.4%** |

> Best: **28.4% (qd_guided)** — guided decomposition outperforms all,
> but does not surpass the simpler guided hybrid retrieval (33.3%).

---

### Module 2 — Ontology Verification
Uses biomedical ontology signals to re-score or filter candidate answers.

| Experiment | Strategy | EM (%) |
|-----------|---------|--------|
| ov-exp1 (conservative) | Conservative ontology filtering | **32.2%** |
| ov-exp2 (bridge_ref) | Bridge-referenced verification | 27.8% |
| ov-exp3 (consistenc) | Consistency checking | 20.5% |
| ov-exp4 (support_re) | Support re-ranking | 16.7% |
| ov-exp5 (supp_hints) | Hint-based support | 16.7% |

> Best: **32.2% (conservative)** — close to hybrid baseline (33.3%),
> suggesting ontology signals add limited benefit over strong retrieval.

---

### Module 3 — Entity Bridging
Finds intermediate bridge entities (e.g., shared protein targets) to
connect drug pairs across multiple hops.

| Experiment | Strategy | EM (%) |
|-----------|---------|--------|
| eb_guided_retrieval | Guided + bridge context | **33.3%** |
| eb_kb_guided | Knowledge-base guided bridging | 32.2% |
| eb_bridge_pivoted | Pivoted bridge entities | 27.8% |
| eb_cand_filtered | Candidate-filtered bridging | 26.9% |
| stage4_entity_bridge | Basic entity bridge | 12.9% |

> Entity bridging with guided retrieval matches Pipeline 4-5 best (33.3%),
> confirming that bridge-aware retrieval provides useful context. However,
> KB-guided (32.2%) and pivoted (27.8%) variants do not exceed the simpler
> guided approach.

---

### Module 4 — Adaptive Retrieval (+ Wikipedia)
Dynamically selects retrieval strategy based on question signals.
Also incorporates Wikipedia as an external supplementary source.

| Experiment | Strategy | EM (%) |
|-----------|---------|--------|
| adaptive_b4exp12_combined_v2 | Combined adaptive | **31.9%** |
| adaptive_adaptive_atk | Adaptive top-K selection | 31.3% |
| adaptive_b4exp13_signal | Signal-based selection | 29.0% |
| adaptive_b4exp11_wiki_v2_query2doc_k3 | Query2Doc expansion | 25.7% |
| adaptive_b4exp11_wiki_v2_k3 | Wiki v2, k=3 | 25.2% |
| adaptive_b4exp11_wiki_v2_sequential_k3 | Sequential wiki | 24.9% |
| adaptive_b4exp11_wiki_v2_medical_sections_k3 | Medical sections | 21.9% |

> Wikipedia augmentation does NOT improve over hybrid scored retrieval (22.2%)
> for most configurations. Query2Doc expansion (25.7%) shows the best Wikipedia
> result but still falls short of guided hybrid retrieval (33.3%). External
> knowledge introduces noise that confuses the smaller local model.

---

### Summary: All Modules vs Pipeline 4-5 Best
| Module | Best Config | Best EM (%) | vs Pipeline 4-5 Best |
|--------|------------|-------------|----------------------|
| Pipeline 4-5 (baseline for this stage) | guided, k=3 | **33.3%** | -- |
| Entity Bridging | eb_guided_retrieval | **33.3%** | 0.0pp |
| Ontology Verification | ov-exp1 conservative | 32.2% | -1.1pp |
| Adaptive Retrieval | combined_v2 | 31.9% | -1.4pp |
| Query Decomposition | qd_guided | 28.4% | -4.9pp |

> None of the advanced modules exceed the Pipeline 4-5 best (33.3%).
> However, the Entity Bridging module (33.3%) matches it with a different
> approach, providing diversity that proves valuable for ensemble voting
> (which achieves 35.4% EM — see ensemble branch).

---

### Why Exact Match (EM) Only?
MedHop answers are DrugBank IDs — exact strings with no partial correctness.
EM is the official BioCreative IX MedHop Track 2025 metric.

---

### How to Run
```bash
# Query Decomposition
py -3.10 src/inference_pipeline_query_decomp.py

# Ontology Verification
py -3.10 src/inference_pipeline_ontology.py

# Entity Bridging
py -3.10 src/inference_pipeline_entity_bridging.py
py -3.10 src/inference_pipeline_eb_kb.py

# Adaptive Retrieval
py -3.10 src/inference_pipeline_adaptive.py
```
