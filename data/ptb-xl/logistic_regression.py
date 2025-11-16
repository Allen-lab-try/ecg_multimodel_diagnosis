#!/usr/bin/env python3
# ============================================================
# logistic_regression.py — Ridge Logistic Regression baseline (final version)
# ------------------------------------------------------------
# Using ECGFounder 1024-dim embeddings + One-vs-Rest Logistic Regression
# Features:
#   ✔ Parallel training (joblib)
#   ✔ tqdm progress bar
#   ✔ Automatic skip for rare labels
#   ✔ Fast convergence using SAG solver
#   ✔ Fixed random_state for reproducibility
# ============================================================

import numpy as np
import pandas as pd
import json
import os
import ast
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1) Load required data
# ============================================================
required_files = ["features_1024.npy", "ptbxl_label_modified.csv", "ecgfounder_label_order.json"]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"❌ Missing required file: {f}")

print("🚀 Loading embeddings and metadata...")

feats = np.load("features_1024.npy")  # shape (N, 1024)
meta = pd.read_csv("ptbxl_label_modified.csv")
with open("ecgfounder_label_order.json", "r") as f:
    label_order = json.load(f)["label_order"]

print(f"📦 feats shape: {feats.shape}")
print(f"📦 meta shape:  {meta.shape}")

# ============================================================
# 2) Convert scp_codes dictionary → multi-hot binary vector
# ============================================================
def extract_multilabel_vector(codes_str, label_order):
    vec = np.zeros(len(label_order))
    if isinstance(codes_str, str):
        try:
            codes = ast.literal_eval(codes_str)
            for c, v in codes.items():
                if c in label_order and v > 0:
                    vec[label_order.index(c)] = 1
        except:
            pass
    return vec

print("🔧 Converting scp_codes → multi-hot vector...")
Y_true = np.vstack(meta["scp_codes"].apply(lambda x: extract_multilabel_vector(x, label_order)))

# ============================================================
# 3) Train / Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    feats, Y_true, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 4) Train logistic regression for each class in parallel
# ============================================================
print("⚙️ Training Ridge Logistic Regression (parallel OVR, SAG solver)...")

def train_single_class(i):
    # If this class only contains a single value (all positive or all negative) → skip model training
    if np.unique(y_train[:, i]).size < 2:
        return np.zeros(len(X_test))
    clf = LogisticRegression(
        penalty="l2",
        C=5.0,
        solver="sag",      # faster than saga for ridge
        max_iter=1000,
        n_jobs=1
    )
    clf.fit(X_train, y_train[:, i])
    return clf.predict_proba(X_test)[:, 1]

results = Parallel(n_jobs=12)(
    delayed(train_single_class)(i) for i in tqdm(range(y_train.shape[1]), desc="Training 150 classifiers")
)

Y_prob = np.vstack(results).T
Y_pred = (Y_prob >= 0.5).astype(int)

# ============================================================
# 5) Evaluation Metrics
# ============================================================
valid_cols = [i for i in range(Y_true.shape[1]) if y_test[:, i].min() != y_test[:, i].max()]

macro_auc = roc_auc_score(y_test[:, valid_cols], Y_prob[:, valid_cols], average="macro")
micro_auc = roc_auc_score(y_test[:, valid_cols], Y_prob[:, valid_cols], average="micro")

print(f"\n🏆 AUROC Results:")
print(f"📈 Macro-AUROC: {macro_auc:.4f}")
print(f"📈 Micro-AUROC: {micro_auc:.4f}")

report = classification_report(
    y_test, Y_pred, target_names=label_order, zero_division=0, output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("ridge_logreg_baseline_report.csv")

print("\n🏅 Top-10 labels by F1 score:")
print(df_report.sort_values('f1-score', ascending=False).head(10)[["precision","recall","f1-score","support"]].round(3))

print("\n💾 Saved: ridge_logreg_baseline_report.csv")
print("⭐ Done — Final optimized version.")


