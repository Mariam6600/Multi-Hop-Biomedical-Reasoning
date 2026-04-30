# Ensemble — Majority Vote

### Overview
Combines predictions from multiple top-performing pipeline configurations
using majority voting to improve robustness.

**Strategy:** Majority vote across best configs from Pipeline 4-5 and
Advanced Features stages.

---

### How to Run
```bash
py -3.10 src/ensemble_majority_vote.py
```

### Result
| Method | EM (%) |
|--------|--------|
| Best single pipeline (hybrid_scored guided k=3) | 33.3% |
| Ensemble majority vote | [see outputs/ensemble_majority_vote_predictions.json] |

---

### Note
Run `py -3.10 src/evaluate_all.py` to compute final ensemble EM.
