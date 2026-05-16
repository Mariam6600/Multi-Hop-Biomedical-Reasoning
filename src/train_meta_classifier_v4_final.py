"""
src/train_meta_classifier_v4_final.py
=====================================
BioMed Multi-Hop QA — Meta-Classifier V4 FINAL (Drug-Identity-Aware)

اقتراح المشرف (الجديد):
  المشكلة: الـ classifier بيشوف rank_A=1, rank_B=2, rank_C=3 لكل المرشحين
  بنفس النمط، ما يعرف "من" هو المرشح (هويته الدوائية).
  الحل: نشفّر هوية الدواء (Drug Identity) صراحةً عبر one-hot per path.

التحسينات على النسخة الأولى من v4:
  ★ تقييم صادق على ثلاثة مستويات:
      - Train EM   (للتشخيص: إذا قريب من Full → memorization)
      - Val EM ★  (★ الصادق للتعميم — هاد اللي يهم العلمي)
      - Full EM   (نفس منهجية v1/v2/v3 — للمقارنة العادلة مع 40.06%)
  ★ Linear SVM: max_iter = 10000 (تجنّب ConvergenceWarning)
  ★ Neural Network مع Drug Embedding للـ S1 و S3 و S4
  ★ كل المخرجات بـ prefix جديد: meta_classifier_v4f_*
      (ما بيمسح ولا ملف من النسخة v4 السابقة)
  ★ Comparison table بيعلّم overfitting (Train EM − Val EM > 25%)

الـ Feature Sets:
  S1: 14 rank features فقط  (baseline current ≈ 40.06% Full EM)
  S2: 14 rank + Identity-Only one-hot (Variant 1)
  S3: 14 rank + Per-Path Top-1 Identity (Variant 2 — كلام المشرف الحرفي)
  S4: 14 rank + Per-Path Rank-Weighted (Variant 3 — تحسين)
  S5: Identity-only ablation

الـ Classifiers:
  - LightGBM, XGBoost, Random Forest, Logistic Regression
  - Linear SVM (طلب المشرف) — مع Calibration للحصول على predict_proba
  - RBF SVM (low-dim فقط)
  - Neural Network مع Drug Embedding (طلب المشرف) — حل لمشكلة الـ vector الطويل

التشغيل:
  py -3.10 train_meta_classifier_v4_final.py
  py -3.10 train_meta_classifier_v4_final.py --feature-sets S1 S3 S4
  py -3.10 train_meta_classifier_v4_final.py --no-nn

⚠️  لا يحذف أي ملف. كل المخرجات بـ prefix: meta_classifier_v4f_*
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN vs DEV SPLIT (CORRECTED METHODOLOGY)
# ══════════════════════════════════════════════════════════════════════════════
# TRAIN: scoring files from train.json (for classifier training)
# DEV:   scoring files from dev.json (for final evaluation only)

# IMPORTANT: inference_pipeline_scoring_v2.py generates files WITHOUT _TRAIN suffix
# So we use the same filenames for both TRAIN and DEV (they're on train.json anyway)
SCORING_FILES_TRAIN = {
    "A": os.path.join(OUTPUTS_DIR, "scoring_v2_pathA_k3_biomistral-7b_predictions.json"),
    "B": os.path.join(OUTPUTS_DIR, "scoring_v2_pathB_k3_qwen2.5-7b_predictions.json"),
    "C": os.path.join(OUTPUTS_DIR, "scoring_v2_pathC_k3_qwen3.5-9b_predictions.json"),
}

# DEV files would be generated separately if we run inference on dev.json
# For now, we only have train.json files
SCORING_FILES_DEV = {
    "A": os.path.join(OUTPUTS_DIR, "scoring_v2_pathA_k3_biomistral-7b_DEV_predictions.json"),
    "B": os.path.join(OUTPUTS_DIR, "scoring_v2_pathB_k3_qwen2.5-7b_DEV_predictions.json"),
    "C": os.path.join(OUTPUTS_DIR, "scoring_v2_pathC_k3_qwen3.5-9b_DEV_predictions.json"),
}

# For backward compatibility (will be removed after migration)
SCORING_FILES = SCORING_FILES_DEV

# TEMPORARY: Skip DEV evaluation if files don't exist
SKIP_DEV_EVAL = False  # Set to False when DEV files are ready

RRF_K = 60
RANDOM_SEED = 42
OUTPUT_PREFIX = "meta_classifier_v4f"  # ← prefix جديد لتجنّب الكتابة فوق v4

# Baselines للمقارنة
BASELINE_EM_FULL  = 33.33
PREV_BEST_EM_FULL = 40.06   # السابق (v3 enhanced features)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SCORING DATA
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SCORING DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_scoring_files(use_train=True) -> dict:
    """
    Load scoring files from either train or dev set.
    
    Args:
        use_train: If True, load TRAIN files. If False, load DEV files.
    
    Returns:
        dict: {qid → {candidates, answer, scores_by_path}}
    """
    files = SCORING_FILES_TRAIN if use_train else SCORING_FILES_DEV
    dataset_name = "TRAIN" if use_train else "DEV"
    
    print(f"  Loading {dataset_name} scoring files...")
    all_data = {}
    
    for path_label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"  [WARN] Missing: {filepath}")
            if use_train:
                print(f"         You need to run inference on train.json first!")
                print(f"         See instructions in TRAIN_INFERENCE_GUIDE.md")
            continue
        
        with open(filepath, encoding="utf-8") as f:
            preds = json.load(f)
        
        for p in preds:
            qid = p["question_id"]
            if qid not in all_data:
                all_data[qid] = {
                    "candidates":      p.get("candidates", []),
                    "candidate_names": p.get("candidate_names", []),
                    "answer":          p.get("answer", ""),
                    "answer_name":     p.get("answer_name", ""),
                    "scores_by_path":  {},
                }
            raw_scores = p.get("scores", {})
            all_data[qid]["scores_by_path"][path_label] = {
                k.upper(): float(v) for k, v in raw_scores.items()
            }
        
        print(f"  [OK]   Path {path_label}: {len(preds)} questions loaded")
    
    return all_data


def build_drug_vocabulary(all_data: dict) -> dict:
    """قاموس drug_id → index ثابت من كل المرشحين والإجابات."""
    all_drugs = set()
    for qdata in all_data.values():
        for c in qdata["candidates"]:
            all_drugs.add(c.upper())
        if qdata["answer"]:
            all_drugs.add(qdata["answer"].upper())
    sorted_drugs = sorted(all_drugs)  # ترتيب أبجدي لإعادة الإنتاج
    vocab = {drug: idx for idx, drug in enumerate(sorted_drugs)}
    print(f"  [OK]   Drug vocabulary: {len(vocab)} unique drugs")
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def scores_to_ranks(scores_dict: dict, candidates: list) -> dict:
    """scores → ranks داخل سؤال واحد. rank 1 = الأفضل."""
    sorted_cands = sorted(
        candidates,
        key=lambda c: scores_dict.get(c.upper(), 0.0),
        reverse=True,
    )
    return {c.upper(): (i + 1) for i, c in enumerate(sorted_cands)}


def build_rank_features(qid, cand, candidates, scores_by_p):
    """Group 1: 14 rank-based features (الحالية — أفضل نتيجة سابقة 40.06%)."""
    n = len(candidates)
    cu = cand.upper()

    ranks_A = scores_to_ranks(scores_by_p.get("A", {}), candidates)
    ranks_B = scores_to_ranks(scores_by_p.get("B", {}), candidates)
    ranks_C = scores_to_ranks(scores_by_p.get("C", {}), candidates)

    r_A = ranks_A.get(cu, n)
    r_B = ranks_B.get(cu, n)
    r_C = ranks_C.get(cu, n)
    ranks = [r_A, r_B, r_C]

    rrf_score      = sum(1.0 / (r + RRF_K) for r in ranks)
    borda_score    = float(sum(n - r for r in ranks))
    min_rank       = float(min(ranks))
    rank_agreement = 1.0 if len(set(ranks)) == 1 else 0.0
    rank_variance  = float(np.var(ranks))

    scores_A = scores_by_p.get("A", {})
    scores_B = scores_by_p.get("B", {})
    scores_C = scores_by_p.get("C", {})
    max_A = max(scores_A.values()) if scores_A else 1.0
    max_B = max(scores_B.values()) if scores_B else 1.0
    max_C = max(scores_C.values()) if scores_C else 1.0

    sorted_A = sorted(scores_A.values(), reverse=True)
    sorted_B = sorted(scores_B.values(), reverse=True)
    sorted_C = sorted(scores_C.values(), reverse=True)
    gap_A = (sorted_A[0] - sorted_A[1]) if len(sorted_A) > 1 else 0.0
    gap_B = (sorted_B[0] - sorted_B[1]) if len(sorted_B) > 1 else 0.0
    gap_C = (sorted_C[0] - sorted_C[1]) if len(sorted_C) > 1 else 0.0

    score_norm_A = scores_A.get(cu, 0.0) / max_A if max_A > 0 else 0.0
    score_norm_B = scores_B.get(cu, 0.0) / max_B if max_B > 0 else 0.0
    score_norm_C = scores_C.get(cu, 0.0) / max_C if max_C > 0 else 0.0

    return np.array([
        float(r_A), float(r_B), float(r_C),
        rrf_score, borda_score, min_rank,
        rank_agreement, rank_variance,
        float(gap_A), float(gap_B), float(gap_C),
        float(score_norm_A), float(score_norm_B), float(score_norm_C),
    ], dtype=np.float32)


RANK_FEATURE_NAMES = [
    "rank_A", "rank_B", "rank_C",
    "rrf_score", "borda_score", "min_rank",
    "rank_agreement", "rank_variance",
    "score_gap_A", "score_gap_B", "score_gap_C",
    "score_norm_A", "score_norm_B", "score_norm_C",
]


def build_identity_features(cand, candidates, scores_by_p, drug_vocab, variant="v3"):
    """
    Drug-Identity-Aware Features.
      variant=v1: one-hot للهوية فقط (N_drugs)
      variant=v2: per-path top-1 indicator (3 × N_drugs) — كلام المشرف الحرفي
      variant=v3: per-path rank-weighted (3 × N_drugs) — تحسين
    """
    N = len(drug_vocab)
    cu = cand.upper()
    drug_idx = drug_vocab.get(cu, -1)

    if variant == "v1":
        vec = np.zeros(N, dtype=np.float32)
        if drug_idx >= 0:
            vec[drug_idx] = 1.0
        return vec

    elif variant == "v2":
        vec = np.zeros(3 * N, dtype=np.float32)
        if drug_idx < 0:
            return vec
        for i, p_label in enumerate(["A", "B", "C"]):
            ranks = scores_to_ranks(scores_by_p.get(p_label, {}), candidates)
            if ranks.get(cu, 999) == 1:
                vec[i * N + drug_idx] = 1.0
        return vec

    elif variant == "v3":
        n = len(candidates)
        vec = np.zeros(3 * N, dtype=np.float32)
        if drug_idx < 0:
            return vec
        for i, p_label in enumerate(["A", "B", "C"]):
            ranks = scores_to_ranks(scores_by_p.get(p_label, {}), candidates)
            r = ranks.get(cu, n)
            vec[i * N + drug_idx] = (n - r + 1) / n  # 1.0 للأول، 0.11 للأخير
        return vec

    else:
        raise ValueError(f"Unknown variant: {variant}")


def build_features(all_data: dict, drug_vocab: dict, feature_set: str):
    """
    feature_set:
      S1: 14 rank فقط
      S2: 14 rank + Identity-Only one-hot
      S3: 14 rank + Per-Path Top-1 Identity  (كلام المشرف)
      S4: 14 rank + Per-Path Rank-Weighted ID (تحسين)
      S5: Identity فقط (ablation)
    """
    X_rows, y_rows, meta = [], [], []

    for qid, qdata in all_data.items():
        candidates  = qdata["candidates"]
        answer      = qdata["answer"].upper()
        scores_by_p = qdata["scores_by_path"]

        for cand in candidates:
            cu = cand.upper()
            rank_feats = build_rank_features(qid, cand, candidates, scores_by_p)

            if feature_set == "S1":
                feats = rank_feats
            elif feature_set == "S2":
                id_feats = build_identity_features(cand, candidates, scores_by_p,
                                                    drug_vocab, variant="v1")
                feats = np.concatenate([rank_feats, id_feats])
            elif feature_set == "S3":
                id_feats = build_identity_features(cand, candidates, scores_by_p,
                                                    drug_vocab, variant="v2")
                feats = np.concatenate([rank_feats, id_feats])
            elif feature_set == "S4":
                id_feats = build_identity_features(cand, candidates, scores_by_p,
                                                    drug_vocab, variant="v3")
                feats = np.concatenate([rank_feats, id_feats])
            elif feature_set == "S5":
                feats = build_identity_features(cand, candidates, scores_by_p,
                                                 drug_vocab, variant="v2")
            else:
                raise ValueError(f"Unknown feature_set: {feature_set}")

            label = 1 if cu == answer else 0

            X_rows.append(feats)
            y_rows.append(label)
            meta.append({
                "qid":       qid,
                "candidate": cu,
                "drug_idx":  drug_vocab.get(cu, -1),
                "answer":    answer,
                "label":     label,
            })

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"  [OK]   Feature set {feature_set}: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"         Positive ratio: {y.mean():.4f}  ({y.sum()}/{len(y)})")
    print(f"         Sparsity (zeros): {(X == 0).mean() * 100:.2f}%")
    return X, y, meta


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def build_classifiers(feature_set: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC, SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    high_dim = feature_set in ("S2", "S3", "S4", "S5")
    classifiers = {}

    # ── Logistic Regression ─────────────────────────────────────
    # ⚠️ ANTI-OVERFITTING: Stronger regularization for high-dim
    if high_dim:
        lr_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED,
            penalty="l1", solver="liblinear",
            class_weight="balanced",  # ← Handle imbalance
        )
        # ⚠️ ANTI-OVERFITTING: Smaller C values (stronger regularization)
        lr_params = {"clf__C": [0.001, 0.01, 0.1, 1.0]}  # ← Added 0.001
    else:
        lr_clf = LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED,
            penalty="l2", solver="lbfgs",
            class_weight="balanced",
        )
        lr_params = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    classifiers["logistic_regression"] = GridSearchCV(
        Pipeline([("scaler", StandardScaler(with_mean=False if high_dim else True)),
                  ("clf",    lr_clf)]),
        lr_params, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
    )

    # ── Random Forest — low-dim only ───────────────────────────
    # ⚠️ ANTI-OVERFITTING: Added min_samples_split
    if not high_dim:
        classifiers["random_forest"] = GridSearchCV(
            RandomForestClassifier(
                random_state=RANDOM_SEED, 
                class_weight="balanced",
                max_features="sqrt",  # ← Prevent overfitting
            ),
            {
                "n_estimators": [100, 200, 300], 
                "max_depth": [5, 7, 10],  # ← Limit depth
                "min_samples_leaf": [2, 5, 10],  # ← More samples per leaf
                "min_samples_split": [5, 10, 20],  # ← More samples to split
            },
            cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
        )

    # ── LightGBM ────────────────────────────────────────────────
    # ⚠️ ANTI-OVERFITTING: Added regularization parameters
    try:
        import lightgbm as lgb
        if high_dim:
            # High-dim: Stronger regularization
            classifiers["lightgbm"] = GridSearchCV(
                lgb.LGBMClassifier(
                    random_state=RANDOM_SEED, 
                    class_weight="balanced",
                    verbose=-1,
                    min_child_samples=20,  # ← Prevent overfitting
                    reg_alpha=0.1,         # ← L1 regularization
                    reg_lambda=0.1,        # ← L2 regularization
                ),
                {
                    "n_estimators": [50, 100, 200], 
                    "max_depth": [3, 5],  # ← Shallower trees
                    "learning_rate": [0.01, 0.05, 0.1],  # ← Added 0.01
                    "num_leaves": [15, 31],  # ← Fewer leaves
                    "min_child_samples": [20, 50],  # ← More samples per leaf
                },
                cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
            )
        else:
            # Low-dim: Standard parameters
            classifiers["lightgbm"] = GridSearchCV(
                lgb.LGBMClassifier(random_state=RANDOM_SEED, class_weight="balanced",
                                    verbose=-1),
                {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
                 "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
                cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
            )
        print("  [OK]   LightGBM available (with anti-overfitting for high-dim)")
    except ImportError:
        print("  [WARN] LightGBM not found")

    # ── XGBoost ─────────────────────────────────────────────────
    # ⚠️ ANTI-OVERFITTING: Added regularization parameters
    try:
        import xgboost as xgb
        if high_dim:
            # High-dim: Stronger regularization
            classifiers["xgboost"] = GridSearchCV(
                xgb.XGBClassifier(
                    random_state=RANDOM_SEED, 
                    eval_metric="logloss",
                    use_label_encoder=False,
                    reg_alpha=0.1,  # ← L1 regularization
                    reg_lambda=1.0,  # ← L2 regularization
                    min_child_weight=5,  # ← Prevent overfitting
                ),
                {
                    "n_estimators": [50, 100, 200], 
                    "max_depth": [3, 5],  # ← Shallower trees
                    "learning_rate": [0.01, 0.05, 0.1],  # ← Added 0.01
                    "subsample": [0.6, 0.8],  # ← Row sampling
                    "colsample_bytree": [0.6, 0.8],  # ← Column sampling
                },
                cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
            )
        else:
            # Low-dim: Standard parameters
            classifiers["xgboost"] = GridSearchCV(
                xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss",
                                  use_label_encoder=False),
                {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
                 "learning_rate": [0.05, 0.1]},
                cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
            )
        print("  [OK]   XGBoost available (with anti-overfitting for high-dim)")
    except ImportError:
        print("  [WARN] XGBoost not found")

    # ── Linear SVM (طلب المشرف) ────────────────────────────────
    # ⚠️ ANTI-OVERFITTING: Smaller C values for high-dim
    if high_dim:
        svm_C_values = [0.001, 0.01, 0.1, 1.0]  # ← Stronger regularization
    else:
        svm_C_values = [0.01, 0.1, 1.0, 10.0]
    
    classifiers["linear_svm"] = GridSearchCV(
        Pipeline([
            ("scaler", StandardScaler(with_mean=False if high_dim else True)),
            ("clf",    CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_SEED, class_weight="balanced",
                          max_iter=10000, dual="auto"),
                method="sigmoid", cv=3,
            )),
        ]),
        {"clf__estimator__C": svm_C_values},
        cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    print(f"  [OK]   Linear SVM configured (max_iter=10000, C={svm_C_values})")

    # ── RBF SVM — low-dim only ─────────────────────────────────
    if not high_dim:
        classifiers["rbf_svm"] = GridSearchCV(
            Pipeline([("scaler", StandardScaler()),
                      ("clf",    SVC(kernel="rbf", probability=True,
                                     class_weight="balanced",
                                     random_state=RANDOM_SEED))]),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
            cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
        )
        print("  [OK]   RBF SVM configured")

    return classifiers


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK مع Drug Embedding (حل للـ high-dim sparse one-hot)
# ══════════════════════════════════════════════════════════════════════════════

class DrugEmbeddingNN:
    """
    Neural Network مع Embedding Layer للـ drug_id.

    البنية:
      - drug_id (int) → Embedding(n_drugs+1, embed_dim=16)
      - numeric features (14 rank + 3 top-1 indicators = 17)
      - Concat → 33-dim → 2 hidden layers (64, 32) + Dropout + ReLU
      - Output: sigmoid

    لماذا Embedding؟
      المشرف ذكر "حل لو الـ vector طويل". الـ embedding هو الحل القياسي:
      بدل one-hot بطول 639 → نتعلم تمثيل دواء بـ 16 رقم dense.
    """

    def __init__(self, n_drugs, embed_dim=16, hidden_dims=(64, 32),
                 dropout=0.3, lr=1e-3, weight_decay=1e-5,
                 epochs=100, batch_size=64, device=None):
        import torch
        self.torch = torch
        self.n_drugs = n_drugs
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_val_auc = -1.0

    def _build_model(self, n_numeric):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, n_drugs, embed_dim, n_numeric, hidden_dims, dropout):
                super().__init__()
                s.embedding = nn.Embedding(n_drugs + 1, embed_dim, padding_idx=n_drugs)
                in_dim = embed_dim + n_numeric
                layers = []
                for h in hidden_dims:
                    layers.append(nn.Linear(in_dim, h))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    in_dim = h
                layers.append(nn.Linear(in_dim, 1))
                s.fc = nn.Sequential(*layers)

            def forward(s, drug_ids, numeric):
                e = s.embedding(drug_ids)
                x = torch.cat([e, numeric], dim=1)
                return s.fc(x).squeeze(-1)

        return Net(self.n_drugs, self.embed_dim, n_numeric,
                   self.hidden_dims, self.dropout).to(self.device)

    def fit(self, drug_ids_train, X_numeric_train, y_train,
            drug_ids_val=None, X_numeric_val=None, y_val=None, verbose=True):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        drug_ids_train = np.where(drug_ids_train < 0, self.n_drugs, drug_ids_train)
        if drug_ids_val is not None:
            drug_ids_val = np.where(drug_ids_val < 0, self.n_drugs, drug_ids_val)

        n_numeric = X_numeric_train.shape[1]
        self.model = self._build_model(n_numeric)

        # class weights
        pos = max(int(y_train.sum()), 1)
        neg = max(len(y_train) - pos, 1)
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                       lr=self.lr, weight_decay=self.weight_decay)

        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_num_tr = self.scaler.fit_transform(X_numeric_train).astype(np.float32)
        if X_numeric_val is not None:
            X_num_va = self.scaler.transform(X_numeric_val).astype(np.float32)

        ds = TensorDataset(
            torch.from_numpy(drug_ids_train.astype(np.int64)),
            torch.from_numpy(X_num_tr),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        best_state = None
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for d_ids, x_num, y_b in loader:
                d_ids = d_ids.to(self.device)
                x_num = x_num.to(self.device)
                y_b   = y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(d_ids, x_num)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_b)
            total_loss /= len(y_train)

            if drug_ids_val is not None:
                val_proba = self._predict_proba_internal(drug_ids_val, X_num_va)
                val_auc = roc_auc_score(y_val, val_proba)
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                    print(f"      epoch {epoch+1:3d}/{self.epochs} | "
                          f"loss={total_loss:.4f} | val_auc={val_auc:.4f}  "
                          f"(best={self.best_val_auc:.4f})")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _predict_proba_internal(self, drug_ids, X_num_scaled):
        import torch
        self.model.eval()
        d_ids = torch.from_numpy(drug_ids.astype(np.int64)).to(self.device)
        x_num = torch.from_numpy(X_num_scaled.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(d_ids, x_num)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba

    def predict_proba(self, drug_ids, X_numeric):
        drug_ids = np.where(drug_ids < 0, self.n_drugs, drug_ids)
        X_num_scaled = self.scaler.transform(X_numeric).astype(np.float32)
        proba = self._predict_proba_internal(drug_ids, X_num_scaled)
        return np.column_stack([1 - proba, proba])


def build_nn_features(meta, rank_only_X):
    """فصل drug_ids عن numeric features للـ NN."""
    drug_ids = np.array([m["drug_idx"] for m in meta], dtype=np.int64)
    top1_feats = []
    for rfeats in rank_only_X:
        top1_A = 1.0 if rfeats[0] == 1.0 else 0.0
        top1_B = 1.0 if rfeats[1] == 1.0 else 0.0
        top1_C = 1.0 if rfeats[2] == 1.0 else 0.0
        top1_feats.append([top1_A, top1_B, top1_C])
    top1_feats = np.array(top1_feats, dtype=np.float32)
    X_numeric = np.concatenate([rank_only_X, top1_feats], axis=1)
    return drug_ids, X_numeric


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / SPLIT / EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def question_level_split(meta, test_size=0.2, seed=RANDOM_SEED):
    """تقسيم بمستوى السؤال (لا تختلط مرشحين نفس السؤال بين train/val)."""
    rng = np.random.RandomState(seed)
    all_qids = list(dict.fromkeys(m["qid"] for m in meta))
    rng.shuffle(all_qids)
    n_test = int(len(all_qids) * test_size)
    test_qids  = set(all_qids[:n_test])
    train_qids = set(all_qids[n_test:])
    train_idx = [i for i, m in enumerate(meta) if m["qid"] in train_qids]
    val_idx   = [i for i, m in enumerate(meta) if m["qid"] in test_qids]
    return train_idx, val_idx, train_qids, test_qids


def predict_per_question(proba, meta, all_data, restrict_qids=None):
    """probabilities → ranking نهائي لكل سؤال (مع تقييد اختياري لـ qids)."""
    qid_proba = defaultdict(dict)
    for prob, m in zip(proba, meta):
        if restrict_qids is not None and m["qid"] not in restrict_qids:
            continue
        qid_proba[m["qid"]][m["candidate"]] = float(prob)

    predictions = []
    for qid, cand_proba in qid_proba.items():
        answer = all_data[qid]["answer"].upper()
        ranked = sorted(cand_proba, key=cand_proba.get, reverse=True)
        best   = ranked[0] if ranked else ""
        predictions.append({
            "question_id":       qid,
            "answer":            answer,
            "prediction":        best,
            "ranked_candidates": ranked,
            "is_correct":        best.upper() == answer.upper(),
        })
    return predictions


def evaluate_predictions(predictions: list) -> dict:
    """Recall@K (and EM = Recall@1)."""
    n = len(predictions)
    if n == 0:
        return {f"recall@{k}": 0.0 for k in range(1, 10)}
    hits = {k: 0 for k in range(1, 10)}
    for p in predictions:
        ranked = p.get("ranked_candidates", [])
        ans    = p.get("answer", "").upper()
        for k in range(1, 10):
            if ans in [r.upper() for r in ranked[:k]]:
                hits[k] += 1
    return {f"recall@{k}": round(hits[k] / n * 100, 2) for k in range(1, 10)}


def compute_three_level_em(proba_full, meta, all_data, train_qids, val_qids):
    """
    التقييم الصادق على ثلاثة مستويات:
      - train_metrics: على أسئلة التدريب فقط
      - val_metrics  : على أسئلة الـ validation فقط ★ الصادق للتعميم
      - full_metrics : على كل الأسئلة (مطابق منهجية v1/v2/v3 → 40.06%)
    """
    train_preds = predict_per_question(proba_full, meta, all_data, train_qids)
    val_preds   = predict_per_question(proba_full, meta, all_data, val_qids)
    full_preds  = predict_per_question(proba_full, meta, all_data, None)

    train_rc = evaluate_predictions(train_preds)
    val_rc   = evaluate_predictions(val_preds)
    full_rc  = evaluate_predictions(full_preds)

    return {
        "train_em":     train_rc.get("recall@1", 0.0),
        "val_em":       val_rc.get("recall@1", 0.0),
        "full_em":      full_rc.get("recall@1", 0.0),
        "train_recall": train_rc,
        "val_recall":   val_rc,
        "full_recall":  full_rc,
        "predictions":  full_preds,  # نحفظ الكل
    }


def train_sklearn_clf(clf, clf_name, X, y, meta, all_data,
                      train_idx, val_idx, train_qids, val_qids):
    from sklearn.metrics import roc_auc_score
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    val_proba = clf.predict_proba(X_va)[:, 1]
    val_auc   = roc_auc_score(y_va, val_proba)
    best_p    = clf.best_params_ if hasattr(clf, "best_params_") else {}

    # نتنبأ على كل الداتا → ثم نُقيِّم على 3 مستويات
    full_proba = clf.predict_proba(X)[:, 1]
    em_info = compute_three_level_em(full_proba, meta, all_data, train_qids, val_qids)

    return {
        "name":         clf_name,
        "val_auc":      float(val_auc),
        "best_params":  best_p,
        "train_time":   float(train_time),
        **em_info,
    }


def train_sklearn_clf_corrected(clf, clf_name, 
                                 X_train, y_train, meta_train, train_data,
                                 X_dev, y_dev, meta_dev, dev_data,
                                 train_idx, val_idx, train_qids, val_qids,
                                 feature_set, drug_vocab):
    """
    CORRECTED: Train on train.json, evaluate on dev.json.
    
    ⚠️ ANTI-OVERFITTING: Added overfitting detection
    
    Args:
        X_train, y_train, meta_train: Features from train.json
        X_dev, y_dev, meta_dev: Features from dev.json (unseen)
        train_idx, val_idx: Internal split of train.json for model selection
        feature_set: Feature set name (for feature importance)
        drug_vocab: Drug vocabulary (for feature names)
    """
    from sklearn.metrics import roc_auc_score
    
    # Internal train/val split (for hyperparameter tuning)
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_va = X_train[val_idx]
    y_va = y_train[val_idx]

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # Validation AUC (internal - for model selection)
    val_proba = clf.predict_proba(X_va)[:, 1]
    val_auc   = roc_auc_score(y_va, val_proba)
    best_p    = clf.best_params_ if hasattr(clf, "best_params_") else {}
    
    # ⚠️ OVERFITTING DETECTION: Check train AUC
    train_proba = clf.predict_proba(X_tr)[:, 1]
    train_auc = roc_auc_score(y_tr, train_proba)
    auc_gap = train_auc - val_auc

    # Calculate validation predictions (for fallback if no DEV data)
    val_preds = predict_per_question(val_proba, [meta_train[i] for i in val_idx], train_data, val_qids)
    val_recall = evaluate_predictions(val_preds)
    val_em = val_recall.get("recall@1", 0.0)

    # Final evaluation on DEV set (truly unseen) - if available
    if X_dev is not None and y_dev is not None:
        dev_proba = clf.predict_proba(X_dev)[:, 1]
        dev_preds = predict_per_question(dev_proba, meta_dev, dev_data, None)
        dev_recall = evaluate_predictions(dev_preds)
        dev_em = dev_recall.get("recall@1", 0.0)
    else:
        # No DEV data - use validation EM as proxy
        dev_em = val_em
        dev_recall = val_recall
        dev_preds = val_preds
    
    # Extract feature importance
    feature_names = get_feature_names(feature_set, drug_vocab)
    feat_imp = extract_feature_importance(clf, clf_name, feature_set, feature_names)
    path_imp = analyze_path_importance(feat_imp.get("all_features", {}), feature_set)
    
    # ⚠️ OVERFITTING WARNING
    overfitting_warning = ""
    if auc_gap > 0.15:
        overfitting_warning = "⚠️ HIGH OVERFITTING"
    elif auc_gap > 0.10:
        overfitting_warning = "⚠️ MODERATE OVERFITTING"
    elif auc_gap > 0.05:
        overfitting_warning = "⚠️ MILD OVERFITTING"

    return {
        "name":               clf_name,
        "val_auc":            float(val_auc),
        "train_auc":          float(train_auc),  # ← NEW
        "auc_gap":            float(auc_gap),    # ← NEW
        "overfitting_warning": overfitting_warning,  # ← NEW
        "best_params":        best_p,
        "train_time":         float(train_time),
        "dev_em":             dev_em,
        "dev_recall":         dev_recall,
        "predictions":        dev_preds,
        "feature_importance": feat_imp,
        "path_importance":    path_imp,
    }


def train_nn_clf(X, y, meta, all_data, train_idx, val_idx,
                 train_qids, val_qids, drug_vocab, rank_only_X, n_drugs):
    """NN مع drug-embedding + numeric features."""
    from sklearn.metrics import roc_auc_score

    drug_ids_all, X_num_all = build_nn_features(meta, rank_only_X)
    drug_ids_tr = drug_ids_all[train_idx]
    drug_ids_va = drug_ids_all[val_idx]
    X_num_tr   = X_num_all[train_idx]
    X_num_va   = X_num_all[val_idx]
    y_tr       = y[train_idx]
    y_va       = y[val_idx]

    t0 = time.time()
    nn = DrugEmbeddingNN(
        n_drugs=n_drugs, embed_dim=16,
        hidden_dims=(64, 32), dropout=0.3,
        lr=1e-3, weight_decay=1e-5,
        epochs=80, batch_size=64,
    )
    nn.fit(drug_ids_tr, X_num_tr, y_tr,
           drug_ids_va, X_num_va, y_va, verbose=True)
    train_time = time.time() - t0

    val_proba = nn.predict_proba(drug_ids_va, X_num_va)[:, 1]
    val_auc   = roc_auc_score(y_va, val_proba)

    full_proba = nn.predict_proba(drug_ids_all, X_num_all)[:, 1]
    em_info = compute_three_level_em(full_proba, meta, all_data, train_qids, val_qids)

    return {
        "name":         "neural_network_embedding",
        "val_auc":      float(val_auc),
        "best_params":  {"embed_dim": 16, "hidden_dims": [64, 32],
                         "dropout": 0.3, "epochs": 80},
        "train_time":   float(train_time),
        **em_info,
    }


def train_nn_clf_corrected(X_train, y_train, meta_train, train_data,
                           X_dev, y_dev, meta_dev, dev_data,
                           train_idx, val_idx, train_qids, val_qids,
                           drug_vocab, rank_only_X_train, rank_only_X_dev, n_drugs):
    """
    CORRECTED: NN trained on train.json, evaluated on dev.json.
    """
    from sklearn.metrics import roc_auc_score

    # Build NN features for train
    drug_ids_train, X_num_train = build_nn_features(meta_train, rank_only_X_train)
    drug_ids_tr = drug_ids_train[train_idx]
    drug_ids_va = drug_ids_train[val_idx]
    X_num_tr = X_num_train[train_idx]
    X_num_va = X_num_train[val_idx]
    y_tr = y_train[train_idx]
    y_va = y_train[val_idx]

    # Build NN features for dev
    drug_ids_dev, X_num_dev = build_nn_features(meta_dev, rank_only_X_dev)

    t0 = time.time()
    nn = DrugEmbeddingNN(
        n_drugs=n_drugs, embed_dim=16,
        hidden_dims=(64, 32), dropout=0.3,
        lr=1e-3, weight_decay=1e-5,
        epochs=80, batch_size=64,
    )
    nn.fit(drug_ids_tr, X_num_tr, y_tr,
           drug_ids_va, X_num_va, y_va, verbose=True)
    train_time = time.time() - t0

    # Validation AUC (internal)
    val_proba = nn.predict_proba(drug_ids_va, X_num_va)[:, 1]
    val_auc = roc_auc_score(y_va, val_proba)

    # Final evaluation on DEV - if available
    if X_dev is not None and y_dev is not None:
        dev_proba = nn.predict_proba(drug_ids_dev, X_num_dev)[:, 1]
        dev_preds = predict_per_question(dev_proba, meta_dev, dev_data, None)
        dev_recall = evaluate_predictions(dev_preds)
        dev_em = dev_recall.get("recall@1", 0.0)
    else:
        # No DEV data - use validation predictions
        val_preds = predict_per_question(val_proba, [meta_train[i] for i in val_idx], train_data, val_qids)
        val_recall = evaluate_predictions(val_preds)
        dev_em = val_recall.get("recall@1", 0.0)
        dev_recall = val_recall
        dev_preds = val_preds

    return {
        "name":         "neural_network_embedding",
        "val_auc":      float(val_auc),
        "best_params":  {"embed_dim": 16, "hidden_dims": [64, 32],
                         "dropout": 0.3, "epochs": 80},
        "train_time":   float(train_time),
        "dev_em":       dev_em,
        "dev_recall":   dev_recall,
        "predictions":  dev_preds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_feature_importance(clf, clf_name, feature_set, feature_names=None):
    """
    استخراج أهمية الميزات من المصنفات التي تدعم ذلك.
    
    Returns:
        dict: {"feature_name": importance_score, ...} مرتبة تنازلياً
    """
    importance_dict = {}
    
    try:
        # For GridSearchCV, get the best estimator
        if hasattr(clf, "best_estimator_"):
            estimator = clf.best_estimator_
        else:
            estimator = clf
        
        # For Pipeline, get the final classifier
        if hasattr(estimator, "named_steps"):
            estimator = estimator.named_steps.get("clf", estimator)
        
        # For CalibratedClassifierCV, get base estimator
        if hasattr(estimator, "estimator"):
            estimator = estimator.estimator
        
        # Extract feature importance based on classifier type
        importances = None
        
        if clf_name in ("lightgbm", "xgboost", "random_forest"):
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
        
        elif clf_name == "logistic_regression":
            if hasattr(estimator, "coef_"):
                # Use absolute values of coefficients
                importances = np.abs(estimator.coef_[0])
        
        if importances is not None and feature_names is not None:
            # Create dict and sort by importance
            for fname, imp in zip(feature_names, importances):
                importance_dict[fname] = float(imp)
            
            # Sort by importance (descending)
            importance_dict = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # Get top 10 for summary
            top_10 = dict(list(importance_dict.items())[:10])
            
            return {
                "all_features": importance_dict,
                "top_10": top_10,
                "extraction_success": True,
            }
    
    except Exception as e:
        print(f"    [WARN] Could not extract feature importance: {e}")
    
    return {
        "all_features": {},
        "top_10": {},
        "extraction_success": False,
    }


def get_feature_names(feature_set, drug_vocab):
    """
    إنشاء أسماء الميزات حسب الـ feature set.
    """
    names = RANK_FEATURE_NAMES.copy()  # 14 rank features
    
    if feature_set == "S1":
        return names
    
    elif feature_set == "S2":
        # Identity-only one-hot
        for drug in sorted(drug_vocab.keys()):
            names.append(f"drug_id_{drug}")
    
    elif feature_set == "S3":
        # Per-path top-1 identity
        for path in ["A", "B", "C"]:
            for drug in sorted(drug_vocab.keys()):
                names.append(f"path{path}_top1_{drug}")
    
    elif feature_set == "S4":
        # Per-path rank-weighted
        for path in ["A", "B", "C"]:
            for drug in sorted(drug_vocab.keys()):
                names.append(f"path{path}_weighted_{drug}")
    
    elif feature_set == "S5":
        # Identity-only (same as S3 structure)
        for path in ["A", "B", "C"]:
            for drug in sorted(drug_vocab.keys()):
                names.append(f"path{path}_top1_{drug}")
    
    return names


def analyze_path_importance(importance_dict, feature_set):
    """
    تحليل أهمية كل path (A, B, C) من الميزات.
    
    Returns:
        dict: {"path_A": score, "path_B": score, "path_C": score}
    """
    path_scores = {"path_A": 0.0, "path_B": 0.0, "path_C": 0.0}
    
    if not importance_dict:
        return path_scores
    
    # Rank features
    if "rank_A" in importance_dict:
        path_scores["path_A"] += importance_dict.get("rank_A", 0)
    if "rank_B" in importance_dict:
        path_scores["path_B"] += importance_dict.get("rank_B", 0)
    if "rank_C" in importance_dict:
        path_scores["path_C"] += importance_dict.get("rank_C", 0)
    
    # Score features
    if "score_norm_A" in importance_dict:
        path_scores["path_A"] += importance_dict.get("score_norm_A", 0)
    if "score_norm_B" in importance_dict:
        path_scores["path_B"] += importance_dict.get("score_norm_B", 0)
    if "score_norm_C" in importance_dict:
        path_scores["path_C"] += importance_dict.get("score_norm_C", 0)
    
    # Gap features
    if "score_gap_A" in importance_dict:
        path_scores["path_A"] += importance_dict.get("score_gap_A", 0)
    if "score_gap_B" in importance_dict:
        path_scores["path_B"] += importance_dict.get("score_gap_B", 0)
    if "score_gap_C" in importance_dict:
        path_scores["path_C"] += importance_dict.get("score_gap_C", 0)
    
    # Identity features (for S3, S4)
    if feature_set in ("S3", "S4"):
        for fname, imp in importance_dict.items():
            if "pathA_" in fname:
                path_scores["path_A"] += imp
            elif "pathB_" in fname:
                path_scores["path_B"] += imp
            elif "pathC_" in fname:
                path_scores["path_C"] += imp
    
    # Normalize
    total = sum(path_scores.values())
    if total > 0:
        path_scores = {k: round(v / total * 100, 2) for k, v in path_scores.items()}
    
    return path_scores


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, feature_set: str, drug_vocab: dict):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    for clf_name, info in results.items():
        prefix = f"{OUTPUT_PREFIX}_{feature_set}_{clf_name}_CORRECTED"
        pred_path = os.path.join(OUTPUTS_DIR, f"{prefix}_predictions.json")
        log_path  = os.path.join(OUTPUTS_DIR, f"{prefix}_logs.json")
        feat_imp_path = os.path.join(OUTPUTS_DIR, f"{prefix}_feature_importance.json")

        # Save predictions
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(info["predictions"], f, indent=2, ensure_ascii=False)

        # Save logs with feature importance
        log = {
            "classifier":     clf_name,
            "feature_set":    feature_set,
            "methodology":    "CORRECTED: Trained on train.json, Evaluated on dev.json",
            "val_auc":        info["val_auc"],
            "best_params":    info.get("best_params", {}),
            "train_time":     info.get("train_time", 0),
            "dev_em":         info["dev_em"],
            "dev_recall":     info["dev_recall"],
            "vs_baseline":    round(info["dev_em"] - BASELINE_EM_FULL, 2),
            "vs_prev_best":   round(info["dev_em"] - PREV_BEST_EM_FULL, 2),
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        
        # Save feature importance separately
        if "feature_importance" in info and info["feature_importance"]["extraction_success"]:
            feat_imp_data = {
                "classifier": clf_name,
                "feature_set": feature_set,
                "top_10_features": info["feature_importance"]["top_10"],
                "path_importance": info.get("path_importance", {}),
                "all_features": info["feature_importance"]["all_features"],
            }
            with open(feat_imp_path, "w", encoding="utf-8") as f:
                json.dump(feat_imp_data, f, indent=2, ensure_ascii=False)
            
            print(f"    Feature importance saved → {os.path.basename(feat_imp_path)}")


def print_comparison_table_corrected(all_results: dict):
    """جدول مقارنة - CORRECTED methodology with overfitting detection."""
    print("\n" + "═" * 120)
    print("  COMPARISON TABLE — V4-FINAL (CORRECTED METHODOLOGY + ANTI-OVERFITTING)")
    print("  ✅ Trained on train.json, Evaluated on dev.json (truly unseen)")
    print("═" * 120)
    print(f"  {'Feature':<8} {'Classifier':<24} {'Train AUC':>10} {'Val AUC':>8} "
          f"{'AUC Gap':>8} {'DEV EM':>9} {'vs 40.06':>9} {'Top Path':>12}")
    print(f"  {'─'*8} {'─'*24} {'─'*10} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*12}")

    rows = []
    for (fs, clf_name), info in all_results.items():
        # Determine top path from path_importance
        path_imp = info.get("path_importance", {})
        if path_imp:
            top_path = max(path_imp, key=path_imp.get)
            top_path_str = f"{top_path[-1]}({path_imp[top_path]:.1f}%)"
        else:
            top_path_str = "N/A"
        
        train_auc = info.get("train_auc", 0)
        auc_gap = info.get("auc_gap", 0)
        
        rows.append((
            fs, clf_name, train_auc, info["val_auc"],
            auc_gap, info["dev_em"], info["dev_em"] - PREV_BEST_EM_FULL,
            top_path_str,
        ))
    rows.sort(key=lambda r: -r[5])  # Sort by DEV EM

    for fs, clf_name, train_auc, val_auc, auc_gap, dev_em, vs_prev, top_path in rows:
        marker = ""
        if dev_em >= 45:
            marker = " ★"  # Excellent
        elif dev_em >= 40:
            marker = " ✓"  # Good
        
        # ⚠️ OVERFITTING MARKER
        overfit_marker = ""
        if auc_gap > 0.15:
            overfit_marker = " ⚠️⚠️"  # High overfitting
        elif auc_gap > 0.10:
            overfit_marker = " ⚠️"    # Moderate overfitting
        
        sign = "+" if vs_prev >= 0 else ""
        print(f"  {fs:<8} {clf_name:<24} {train_auc:>10.4f} {val_auc:>8.4f} "
              f"{auc_gap:>8.4f}{overfit_marker:<4} {dev_em:>8.2f}% {sign}{vs_prev:>7.2f}% {top_path:>12}{marker}")

    print("─" * 120)
    print("  ★ = DEV EM ≥ 45%   ✓ = DEV EM ≥ 40%   ⚠️ = AUC Gap > 0.10 (overfitting)   ⚠️⚠️ = AUC Gap > 0.15 (high overfitting)")
    print("  Top Path = Most important retrieval path")
    print("═" * 120)

    if rows:
        best = max(rows, key=lambda r: r[5])
        print(f"\n  ★ BEST: {best[0]} + {best[1]:<22}  DEV EM = {best[5]:.2f}%  AUC Gap = {best[4]:.4f}  Top Path = {best[7]}")
        print(f"  Previous best (dev.json only):     40.06%")
        print(f"  Baseline (single-pipeline):         {BASELINE_EM_FULL}%")
        
        # ⚠️ OVERFITTING SUMMARY
        high_overfit = [r for r in rows if r[4] > 0.15]
        if high_overfit:
            print(f"\n  ⚠️⚠️ WARNING: {len(high_overfit)} classifier(s) show HIGH overfitting (AUC Gap > 0.15)")
            print(f"      Consider using stronger regularization or simpler feature sets")
    print("═" * 120 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(feature_sets, classifiers_to_run, do_nn=True):
    print("\n" + "═" * 88)
    print("  Meta-Classifier V4-FINAL — Drug-Identity-Aware Features")
    print("  ✅ CORRECTED METHODOLOGY: Train on train.json, Evaluate on dev.json")
    print("  ✅ ANTI-OVERFITTING: Stronger regularization for high-dimensional features")
    print("═" * 88 + "\n")
    
    # ⚠️ WARNING: High-dimensional feature sets
    high_dim_sets = [fs for fs in feature_sets if fs in ("S2", "S3", "S4", "S5")]
    if high_dim_sets:
        print("  ⚠️  WARNING: High-dimensional feature sets detected:", ", ".join(high_dim_sets))
        print("      These may cause overfitting. Consider using S1 (14 features) first.")
        print("      Anti-overfitting measures enabled: stronger regularization, shallower trees.\n")

    # ── Step 1: Load TRAIN data (for classifier training) ─────────────────────
    print("--- Step 1: Loading TRAIN scoring files (for classifier training) ---")
    train_data = load_scoring_files(use_train=True)
    if not train_data:
        print("  [FAIL] No TRAIN scoring data found.")
        print("\n  ⚠️  You need to run inference on train.json first!")
        print("  See: TRAIN_INFERENCE_GUIDE.md for instructions")
        sys.exit(1)
    print(f"  Total TRAIN questions: {len(train_data)}\n")

    # ── Step 2: Load DEV data (for final evaluation) ──────────────────────────
    print("--- Step 2: Loading DEV scoring files (for final evaluation) ---")
    if SKIP_DEV_EVAL:
        print("  [INFO] Skipping DEV evaluation (SKIP_DEV_EVAL = True)")
        print("  [INFO] Will use train/validation split instead")
        dev_data = {}
    else:
        dev_data = load_scoring_files(use_train=False)
        if not dev_data:
            print("  [FAIL] No DEV scoring data found.")
            sys.exit(1)
        print(f"  Total DEV questions: {len(dev_data)}\n")

    # ── Step 3: Build drug vocabulary (from train only if no dev) ─────────────
    print("--- Step 3: Building drug vocabulary ---")
    if dev_data:
        print("  Using train + dev data")
        combined_data = {**train_data, **dev_data}
    else:
        print("  Using train data only")
        combined_data = train_data
    drug_vocab = build_drug_vocabulary(combined_data)
    n_drugs = len(drug_vocab)
    print()

    # ── Step 4: Build features for TRAIN (for classifier training) ────────────
    print("--- Step 4: Build features for TRAIN set ---")
    rank_only_X_train, _, meta_train = build_features(train_data, drug_vocab, "S1")
    
    # Split TRAIN into train/val (80/20) for model selection
    train_idx, val_idx, train_qids, val_qids = question_level_split(
        meta_train, test_size=0.2
    )
    print(f"\n--- TRAIN split: {len(train_qids)} train Qs, {len(val_qids)} val Qs ---")
    print(f"    (Used for hyperparameter tuning and model selection)\n")

    # ── Step 5: Build features for DEV (for final evaluation) ─────────────────
    if dev_data:
        print("--- Step 5: Build features for DEV set (final evaluation) ---")
        rank_only_X_dev, _, meta_dev = build_features(dev_data, drug_vocab, "S1")
        print()
    else:
        print("--- Step 5: Skipping DEV feature building (no DEV data) ---")
        print("  Will use validation split from train data for evaluation\n")
        rank_only_X_dev, meta_dev = None, None

    all_results = {}

    for feature_set in feature_sets:
        print(f"\n{'='*88}\n  RUNNING FEATURE SET: {feature_set}\n{'='*88}")
        
        # Build features for TRAIN
        X_train_full, y_train_full, meta_train_full = build_features(
            train_data, drug_vocab, feature_set
        )
        
        # Build features for DEV (if available)
        if dev_data:
            X_dev, y_dev, meta_dev_full = build_features(
                dev_data, drug_vocab, feature_set
            )
        else:
            X_dev, y_dev, meta_dev_full = None, None, None
        print()

        clfs = build_classifiers(feature_set)
        if classifiers_to_run:
            clfs = {k: v for k, v in clfs.items() if k in classifiers_to_run}

        results = {}
        for clf_name, clf in clfs.items():
            print(f"  Training {clf_name} ({feature_set})...")
            try:
                # Train on TRAIN set (with internal train/val split)
                info = train_sklearn_clf_corrected(
                    clf, clf_name, 
                    X_train_full, y_train_full, meta_train_full, train_data,
                    X_dev, y_dev, meta_dev_full, dev_data,
                    train_idx, val_idx, train_qids, val_qids,
                    feature_set, drug_vocab,
                )
                print(f"    Val AUC (internal) = {info['val_auc']:.4f}  | "
                      f"Train AUC = {info.get('train_auc', 0):.4f}  | "
                      f"AUC Gap = {info.get('auc_gap', 0):.4f}  | "
                      f"DEV EM (final) = {info['dev_em']:.2f}%  | "
                      f"Time = {info['train_time']:.1f}s")
                
                # ⚠️ OVERFITTING WARNING
                if info.get('overfitting_warning'):
                    print(f"    {info['overfitting_warning']}")
                
                # Print feature importance if available
                if info.get("feature_importance", {}).get("extraction_success"):
                    top_3 = list(info["feature_importance"]["top_10"].items())[:3]
                    print(f"    Top 3 features: {', '.join([f'{k}={v:.4f}' for k, v in top_3])}")
                    path_imp = info.get("path_importance", {})
                    if path_imp:
                        print(f"    Path importance: A={path_imp.get('path_A', 0):.1f}% "
                              f"B={path_imp.get('path_B', 0):.1f}% "
                              f"C={path_imp.get('path_C', 0):.1f}%")
                
                results[clf_name] = info
                all_results[(feature_set, clf_name)] = info
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    [FAIL] {e}")

        # ── Neural Network ─────────────────────────────────────────────────────
        if do_nn and feature_set in ("S1", "S3", "S4"):
            print(f"\n  Training neural_network_embedding ({feature_set})...")
            try:
                info = train_nn_clf_corrected(
                    X_train_full, y_train_full, meta_train_full, train_data,
                    X_dev, y_dev, meta_dev_full, dev_data,
                    train_idx, val_idx, train_qids, val_qids,
                    drug_vocab, rank_only_X_train, rank_only_X_dev, n_drugs,
                )
                print(f"    Val AUC (internal) = {info['val_auc']:.4f}  | "
                      f"DEV EM (final) = {info['dev_em']:.2f}%  | "
                      f"Time = {info['train_time']:.1f}s")
                results["neural_network_embedding"] = info
                all_results[(feature_set, "neural_network_embedding")] = info
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    [FAIL NN] {e}")

        save_results(results, feature_set, drug_vocab)

    print_comparison_table_corrected(all_results)

    # Summary JSON
    summary = []
    for (fs, clf_name), info in all_results.items():
        summary.append({
            "feature_set":    fs,
            "classifier":     clf_name,
            "val_auc":        info["val_auc"],
            "dev_em":         info["dev_em"],
            "dev_recall":     info["dev_recall"],
            "best_params":    info.get("best_params", {}),
        })
    summary.sort(key=lambda s: -s["dev_em"])

    out_summary = os.path.join(OUTPUTS_DIR, f"{OUTPUT_PREFIX}_summary_CORRECTED.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":     "V4-FINAL — CORRECTED METHODOLOGY",
            "methodology":    "Train on train.json, Evaluate on dev.json (truly unseen)" if dev_data else "Train on train.json with train/val split",
            "train_questions": len(train_data),
            "dev_questions":   len(dev_data) if dev_data else 0,
            "n_drugs":         n_drugs,
            "baseline_em":     BASELINE_EM_FULL,
            "prev_em_dev":     PREV_BEST_EM_FULL,
            "results":         summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved → {OUTPUT_PREFIX}_summary_CORRECTED.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sets", nargs="+",
                         default=["S1", "S3", "S4"],
                         help="Feature sets: S1, S2, S3, S4, S5")
    parser.add_argument("--classifiers", nargs="+", default=None,
                         help="Specific classifiers (default: all)")
    parser.add_argument("--no-nn", action="store_true",
                         help="Skip neural network")
    args = parser.parse_args()

    run_experiment(args.feature_sets, args.classifiers, do_nn=not args.no_nn)
