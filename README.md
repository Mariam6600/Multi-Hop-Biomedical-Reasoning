# Multi-Hop Biomedical Reasoning on MedHop
### Graduation Project — Biomedical NLP

A systematic study of **retrieval-augmented multi-hop reasoning** for drug interaction identification using only **local, open-source LLMs** (no paid APIs required). Evaluated on the QAnGaroo MedHop benchmark (342 questions) using **Exact Match (EM)** — the official BioCreative IX MedHop Track 2025 metric.

---

## 🎯 Problem Statement

**Task:** Given a drug name and a set of candidate drugs, identify which candidate drug interacts with the given drug — requiring multi-hop reasoning across biomedical literature.

**Dataset:** QAnGaroo MedHop v1.1, dev split — 342 questions, 133 unique answers, avg. 8.6 candidates/question.

**Why EM only?** MedHop answers are DrugBank IDs (e.g., `DB00563`) — exact strings with no partial correctness. EM directly measures whether the model selected the correct drug. F1, BLEU, and ROUGE are inapplicable to this closed-candidate selection task. EM is also the official metric used in BioCreative IX MedHop Track 2025.

---

## 📈 Results Summary

### Our Progression (Local Models Only)

| Stage | Model | Retriever | Best EM (%) | Δ vs Previous |
|-------|-------|-----------|------------|---------------|
| Baseline 1 | BioMistral-7B | None | 15.8% | — |
| Baseline 2 | BioMistral-7B | BM25 (k=10) | 12.3% | −3.5pp |
| Baseline 3 | BioMistral-7B | Dense MedCPT (k=5) | 16.4% | +0.6pp |
| Pipeline 4-5 | Qwen3.5-9B | Hybrid Scored (k=3) | 33.3% | +16.9pp |
| Advanced Features | Qwen3.5-9B | Hybrid + Ontology | 32.2% | — |
| Ensemble | Qwen3.5-9B | Majority Vote (3-way) | 35.4% | +2.1pp |
| **Meta-Classifier** | **Logistic Regression** | **3-Path Ensemble (A+B+C)** | **51.17%** | **+15.77pp** |

> **Best local result: 51.17% EM (meta-classifier)** — achieved by training a meta-learning ensemble that combines predictions from three independent retrieval paths (BM25 + biomistral-7b, MedCPT + qwen2.5-7b, Hybrid + qwen3.5-9b). Best single model: 33.3% EM with Hybrid Scored Retrieval + Guided Prompt at k=3. Best majority-vote ensemble: 35.4% EM.

### Oracle Upper Bound (Not a Production Result)
| Config | EM (%) | Notes |
|--------|--------|-------|
| Gold chain k=3 | 76.9% | Uses true answer path — upper bound only |
| Gold chain k=5 | 67.0% | Uses true answer path — upper bound only |

> **Gap to oracle (best meta-classifier): 25.73pp** — indicates room for retrieval improvement.  
> **Gap to oracle (best single model): 43.6pp** — meta-classifier closes 17.87pp of the gap.  
> **Gap to oracle (ensemble): 41.5pp** — ensemble closes 2.1pp of the gap.

---

### Comparison with BioCreative IX MedHop Track 2025

| System | EM (%) | Setup |
|--------|--------|-------|
| DMIS Lab *(Rank 1)* | 87.3% | GPT-4o + MedCPT + Web Search |
| UETQuintet *(Rank 2)* | 83.8% | GPT-4o-mini + Wikipedia |
| NHSRAG | 73.4% | MedReason-8B + Wikipedia |
| Fluxion | 68.1% | Gemini 2.0/2.5 Flash — No RAG |
| CLaC | 67.6% | Qwen2.5-Coder-32B + Wikipedia + PubMed |
| **Our Best (local)** | **51.17%** | **Meta-Classifier (3-path ensemble) — Local Ollama** |
| **BiDAF (Original Paper)** | **47.8%** | **Welbl et al., 2018 — Baseline** |
| **Orekhovich** | **43.9%** | **Llama3-Med42-8B + BM25S + MedEmbed — Local Ollama** |
| lasigeBioTM | 28.3% | Mistral-7B + Mondo Ontology |
| DeepRAG | 20.7% | DeepSeek R1 + Wikipedia + DPO |
| CaresAI | 18.6% | LoRA fine-tuned LLaMA-3 8B |
| Random Baseline | 11.6% | — |

> Our best local result (**51.17% meta-classifier**) **outperforms the original MedHop BiDAF baseline (47.8%)** by **+3.37pp** and surpasses four published systems. Best single model achieves 33.3%. Best majority-vote ensemble achieves 35.4%.

---

## 🏗️ Project Architecture

```
Multi-Hop Question (MedHop, 342 questions)
          │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Query Expansion                   │
│  Drug synonym expansion + structured        │
│  query reformulation                        │
└────────────────────┬────────────────────────┘
                     │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Hybrid Retrieval                  │
│  BM25 (sparse) + MedCPT (dense)             │
│  Score fusion via RRF / weighted averaging  │
│  Top-K document selection (k=3 optimal)     │
└────────────────────┬────────────────────────┘
                     │
          ▼
┌─────────────────────────────────────────────┐
│  Stage 3: LLM Inference (Local, Ollama)     │
│  Qwen3.5-9B (Q4_K_M, 4-bit quantized)       │
│  Guided prompt with candidate awareness     │
└────────────────────┬────────────────────────┘
                     │
          ▼
     DrugBank ID Answer (e.g., DB00563)
```

---

## 📂 Repository Structure

This repository is organized into **6 branches**, each representing a development stage:

| Branch | Stage | Key Contribution | Best EM |
|--------|-------|-----------------|---------|
| [`baseline1`](../../tree/baseline1) | Direct LLM QA | BioMistral-7B baseline, no retrieval | 15.8% |
| [`baseline2-3`](../../tree/baseline2-3) | Sparse + Dense RAG | BM25 and MedCPT retrieval with BioMistral | 16.4% |
| [`pipeline4-5`](../../tree/pipeline4-5) | Hybrid RAG + Qwen | Hybrid scored retrieval + Qwen3.5-9B | 33.3% |
| [`advanced-features`](../../tree/advanced-features) | Augmentation modules | Query decomp, ontology, entity bridging, adaptive | 32.2% |
| [`ensemble`](../../tree/ensemble) | Ensemble | Majority vote (3-way) over best configs | 35.4% |
| **[`meta-classifier`](../../tree/meta-classifier)** | **Meta-Learning** | **Trained ensemble combining 3 paths** | **51.17%** |

---

## 🔬 Key Findings

**1. Model matters more than retrieval architecture (with BioMistral-7B)**
BioMistral-7B struggles to extract relevant information from retrieved passages — BM25 retrieval (12.3%) actually hurts vs. no retrieval (15.8%). The model is the bottleneck, not the retriever.

**2. Switching to Qwen3.5-9B is the single largest gain (+16.9pp)**
The combination of Qwen3.5-9B + Hybrid Scored Retrieval + Guided Prompt delivers 33.3% EM — a +16.9pp jump over the previous best. Prompt engineering and model quality are decisive.

**3. k=3 outperforms k=5 and k=10 consistently**
Across all Qwen experiments, top-3 documents outperform top-5 and top-10. Context noise from additional documents hurts small local models more than it helps.

**4. Complex reasoning modules do not improve over well-tuned retrieval**
Query decomposition (28.4%), ontology verification (32.2%), and entity bridging (27.8%) all fall short of the simple guided hybrid retrieval (33.3%). For 7-9B models, retrieval precision matters more than reasoning complexity.

**6. 43.6pp gap to oracle upper bound (single model), reduced to 41.5pp with ensemble, further reduced to 25.73pp with meta-classifier**
Oracle (gold-chain) retrieval achieves 76.9%. The gap to our best single model (33.3%) is almost entirely an evidence retrieval problem — the LLM can reason correctly given perfect context. The ensemble (35.4%) closes 2.1pp of this gap by leveraging complementary errors across prompt strategies. The meta-classifier (51.17%) closes an additional 15.77pp by learning which path to trust for each question.

**7. Meta-learning dramatically outperforms voting-based ensemble (+15.77pp)**
The meta-classifier (51.17% EM) achieves a +15.77pp improvement over the best majority-vote ensemble (35.4% EM) and a +31.58pp improvement over the best individual path (19.59% EM). By training on 1,976 features (including drug identity embeddings), the classifier learns complementary strengths of each retrieval path rather than simply voting.

**8. Drug identity features are more important than rank features**
The top 15 most important features in the meta-classifier are dominated by drug-specific embeddings (13 out of 15). The classifier learns "which drug" is correct, not just "where it appears" in rankings. Feature set S4 (with drug identity) outperforms S1 (rank-only) by +23.98pp.

**9. Outperforms original MedHop baseline (+3.37pp)**
Our meta-classifier (51.17% EM) outperforms the BiDAF baseline from the original MedHop paper (Welbl et al., 2018: 47.8% EM) by +3.37 percentage points — achieved with fully local, open-source models.

---

## 🛠️ Environment

| Component | Details |
|-----------|---------|
| Hardware | Lenovo LOQ, Intel i5-13450HX, 16 GB DDR5 |
| GPU | NVIDIA RTX 5050 8 GB VRAM |
| OS | Windows 11 |
| Python | 3.10.11 |
| LLMs | BioMistral-7B-GGUF (Q4_K_M), Qwen3.5-9B-GGUF (Q4_K_M) |
| Inference | Ollama 0.13.5 (fully local) |
| Embeddings | MedCPT-Query-Encoder (local) |

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/Mariam6600/Multi-Hop-Biomedical-Reasoning.git
cd Multi-Hop-Biomedical-Reasoning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
cp .env.example .env
# Edit .env and fill in your API keys (only needed for API experiments)

# 4. Start Ollama
ollama serve

# 5. Setup models (downloads and registers LLMs)
py -3.10 EnvironmentSetup.py

# 6. Run Baseline 1 (quick test)
py -3.10 main_baseline1.py --test
```

---

## 📋 Data

**QAnGaroo MedHop v1.1** — download from [qangaroo.cs.ucl.ac.uk](http://qangaroo.cs.ucl.ac.uk/) and place under `data/qangaroo_v1.1/`.

**DrugBank Vocabulary** — download from [DrugBank Open Data](https://go.drugbank.com/releases/latest#open-data) and place under `data/drugbank_all_drugbank_vocabulary.csv/`.

**Model files** (not included — download separately):
- `BioMistral-7B.Q4_K_M.gguf` from HuggingFace: [BioMistral/BioMistral-7B-GGUF](https://huggingface.co/BioMistral/BioMistral-7B-GGUF)
- `Qwen2.5-7B-Instruct-Q4_K_M.gguf` / `Qwen3.5-9B-Q4_K_M.gguf` from HuggingFace

---

## 🔐 Security

API keys are managed via `.env` (never committed). See `.env.example` for required variables. The `.gitignore` excludes all model weights, dataset files, and output JSON files.

---

## 📚 Citation / Reference

This project evaluates on the MedHop benchmark from:
> Welbl, J., Stenetorp, P., & Riedel, S. (2018). Constructing Datasets for Multi-hop Reading Comprehension Across Documents. *TACL*.

Reference systems from:
> BioCreative IX MedHop Track (2025). [biocreative.bioinformatics.udel.edu](https://biocreative.bioinformatics.udel.edu/)
