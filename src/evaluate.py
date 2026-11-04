"""
evaluate.py
===========
Loads the trained models and test data, produces:
  - Classification reports for every model
  - Confusion matrices (saved as PNG)
  - Per-class precision / recall / F1 for the best model
  - A summary table printed to the console

Run AFTER train.py.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import MODELS_DIR

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR  = os.path.join(BASE_DIR, "evaluation_plots")
os.makedirs(EVAL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────
def _load_artefacts():
    pipeline = joblib.load(os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    test_data  = joblib.load(os.path.join(MODELS_DIR, "test_data.pkl"))
    with open(os.path.join(MODELS_DIR, "comparison_results.json")) as f:
        val_results = json.load(f)
    return pipeline, all_models, test_data, val_results


# ─────────────────────────────────────────────────────────────────
# Plot confusion matrix
# ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test, class_names, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    safe_name = model_name.replace(" ", "_")
    path = os.path.join(EVAL_DIR, f"cm_{safe_name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")
    return y_pred


# ─────────────────────────────────────────────────────────────────
# Print classification report
# ─────────────────────────────────────────────────────────────────
def print_report(model, X_test, y_test, class_names, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{'─'*60}")
    print(f"  {model_name}  –  Test Set Classification Report")
    print(f"{'─'*60}")
    print(classification_report(y_test, y_pred,
                                 target_names=class_names, zero_division=0))
    acc   = accuracy_score(y_test, y_pred)
    f1_ma = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Macro-F1   : {f1_ma:.4f}")
    print(f"  Weighted-F1: {f1_wt:.4f}")

    # Explain false negatives (missed failures)
    non_failure_idx = list(class_names).index("No Failure") if "No Failure" in class_names else None
    if non_failure_idx is not None:
        cm = confusion_matrix(y_test, y_pred)
        total_failures = np.sum(np.delete(np.sum(cm, axis=1), non_failure_idx))
        missed = 0
        for i, name in enumerate(class_names):
            if name != "No Failure":
                # Missed = predicted as No Failure when actually a failure
                missed += cm[i, non_failure_idx]
        fn_rate = (missed / total_failures * 100) if total_failures > 0 else 0
        print(f"\n  ⚠  False Negatives (failures predicted as No Failure): {missed}")
        print(f"     That is {fn_rate:.1f}% of all actual failures.")
        print("     In predictive maintenance, these are CRITICAL – missed")
        print("     failures lead to unexpected downtime and safety risks.")

    return {"accuracy": acc, "macro_f1": f1_ma, "weighted_f1": f1_wt}


# ─────────────────────────────────────────────────────────────────
# Model comparison bar chart on test set
# ─────────────────────────────────────────────────────────────────
def plot_model_comparison(summary: dict) -> None:
    df = pd.DataFrame(summary).T.reset_index()
    df.columns = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]
    df = df.sort_values("Macro-F1", ascending=False)

    metrics = ["Accuracy", "Macro-F1", "Weighted-F1"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["steelblue", "tomato", "mediumseagreen"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df[metric], width, label=metric, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"], rotation=10, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Test Set Metrics",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(EVAL_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved model comparison chart → {path}")


# ─────────────────────────────────────────────────────────────────
# Per-class F1 bar chart for the best model
# ─────────────────────────────────────────────────────────────────
def plot_per_class_f1(model, X_test, y_test, class_names, model_name):
    y_pred = model.predict(X_test)
    f1s = f1_score(y_test, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tomato" if f < 0.5 else "steelblue" for f in f1s]
    bars = ax.bar(class_names, f1s, color=colors, edgecolor="black")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    ax.set_xlabel("Failure Type")
    ax.set_ylabel("F1-Score")
    ax.set_title(f"Per-Class F1-Score – {model_name}",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    path = os.path.join(EVAL_DIR, "per_class_f1_best_model.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per-class F1 chart    → {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION  –  TEST SET")
    print("=" * 60)

    pipeline, all_models, test_data, val_results = _load_artefacts()
    le          = pipeline["label_encoder"]
    class_names = le.classes_
    X_test      = test_data["X_test"]
    y_test      = test_data["y_test"]

    summary = {}
    for name, model in all_models.items():
        plot_confusion_matrix(model, X_test, y_test, class_names, name)
        metrics = print_report(model, X_test, y_test, class_names, name)
        summary[name] = metrics

    # Identify best by macro-F1 on test set
    best_name = max(summary, key=lambda n: summary[n]["macro_f1"])
    print(f"\n  ✔  Best model on TEST SET: {best_name}"
          f"  (Macro-F1 = {summary[best_name]['macro_f1']:.4f})")

    plot_model_comparison(summary)
    plot_per_class_f1(all_models[best_name], X_test, y_test, class_names, best_name)

    # Print final summary table
    print("\n" + "═" * 60)
    print("  FINAL COMPARISON TABLE  (Test Set)")
    print("═" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>13}")
    print("─" * 60)
    for name in sorted(summary, key=lambda n: summary[n]["macro_f1"], reverse=True):
        m      = summary[name]
        marker = " ◀ BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['accuracy']:>10.4f} "
            f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>13.4f}{marker}"
        )
    print("═" * 60)
    print(f"\nAll evaluation plots saved to: {EVAL_DIR}\n")


if __name__ == "__main__":
    run_evaluation()
