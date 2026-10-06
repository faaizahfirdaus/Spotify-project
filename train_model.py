# -*- coding: utf-8 -*-
"""
train_model.py
--------------
Standalone script to train and evaluate the Random Forest classifier.

Usage
-----
    python train_model.py --data path/to/dataset.csv

The trained model is saved to model/rf_model.joblib so that app.py can
load it without re-training on every run.
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_dataset, preprocess, FEATURE_COLS

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")


# ── Training ────────────────────────────────────────────────────────────────

def train(data_path: str, model_type: str = "rf") -> dict:
    """
    Load → preprocess → train → evaluate → save.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset.
    model_type : str
        'rf' for Random Forest, 'lr' for Logistic Regression.

    Returns
    -------
    dict with evaluation metrics and the fitted pipeline.
    """
    print(f"\n[1/4] Loading dataset from: {data_path}")
    df = load_dataset(data_path)
    print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    print("\n[2/4] Preprocessing ...")
    X, y = preprocess(df)
    print(f"      Samples after cleaning : {len(X):,}")
    print(f"      Popular songs (label=1): {y.sum():,}  "
          f"({y.mean() * 100:.1f} %)")
    print(f"      Not popular (label=0)  : {(y == 0).sum():,}")

    # ── Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Build pipeline ──────────────────────────────────────────────────────
    print(f"\n[3/4] Training {'Random Forest' if model_type == 'rf' else 'Logistic Regression'} ...")
    if model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        pipe = Pipeline([("clf", classifier)])
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ])

    pipe.fit(X_train, y_train)

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating ...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["Not Popular", "Popular"],
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision (Popular): {report['Popular']['precision']:.4f}")
    print(f"  Recall    (Popular): {report['Popular']['recall']:.4f}")
    print(f"  F1-score  (Popular): {report['Popular']['f1-score']:.4f}")

    # ── Feature importances (Random Forest only) ───────────────────────────
    importances = None
    if model_type == "rf":
        importances = dict(
            zip(FEATURE_COLS,
                pipe.named_steps["clf"].feature_importances_)
        )
        imp_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        print("\n  Feature Importances:")
        for feat, imp in imp_sorted:
            bar = "#" * int(imp * 40)
            print(f"    {feat:<20} {imp:.4f}  {bar}")

    # ── Save model ──────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"\n  Model saved -> {MODEL_PATH}")

    return {
        "pipeline": pipe,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ── Plotting helpers (called by app.py as well) ─────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, ax=None) -> plt.Figure:
    """Return a styled confusion-matrix figure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Popular", "Popular"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: dict, ax=None) -> plt.Figure:
    """Return a horizontal bar chart of feature importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    feats, vals = zip(*sorted_items)

    colors = ["#3b82d4" if v == max(vals) else "#93c5fd" for v in vals]
    ax.barh(feats, vals, color=colors)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Spotify Song Popularity classifier."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the CSV dataset (e.g. data/dataset.csv)"
    )
    parser.add_argument(
        "--model", default="rf", choices=["rf", "lr"],
        help="Model type: 'rf' = Random Forest (default), 'lr' = Logistic Regression"
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        sys.exit(1)

    result = train(args.data, model_type=args.model)
    print("\nDone. Training complete.")
