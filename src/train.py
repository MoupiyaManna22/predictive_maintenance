"""
train.py
========
Trains multiple classification models on the preprocessed data,
handles class imbalance with SMOTE, optimises hyperparameters via
RandomizedSearchCV, and saves the best model to disk.

Models compared
---------------
1. Logistic Regression      (baseline)
2. Decision Tree            (interpretable)
3. Random Forest            (ensemble)
4. Gradient Boosting        (HistGradientBoosting – fast & strong)

The best model is chosen by macro-F1 on the VALIDATION set so that
rare failure classes carry equal weight to the dominant "No Failure".
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics           import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling    import SMOTE
from imblearn.pipeline         import Pipeline as ImbPipeline
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import run_preprocessing, MODELS_DIR, FEATURE_COLS

# ─────────────────────────────────────────────────────────────────
# Model definitions + hyper-parameter search spaces
# ─────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs", "saga"],
        },
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "clf__max_depth":        [5, 10, 15, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__criterion":        ["gini", "entropy"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "params": {
            "clf__max_depth":        [10, 20, None],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features":     ["sqrt", "log2"],
        },
    },
    "Gradient Boosting": {
        "estimator": HistGradientBoostingClassifier(random_state=42, max_iter=300),
        "params": {
            "clf__max_depth":     [5, 10, 20],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__l2_regularization": [0.0, 0.5, 1.0],
        },
    },
}

CV_FOLDS   = 5      # stratified k-fold
N_ITER     = 15     # random search iterations per model
SCORE_METRIC = "f1_macro"


# ─────────────────────────────────────────────────────────────────
# SMOTE + model pipeline
# ─────────────────────────────────────────────────────────────────
def _make_pipeline(estimator):
    """
    Wrap SMOTE + classifier in an imbalanced-learn Pipeline so that
    SMOTE is applied ONLY inside each CV fold (no leakage).
    
    Note: features are already scaled by preprocess.py.
    """
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("clf",   estimator),
    ])


# ─────────────────────────────────────────────────────────────────
# Train + optimise one model
# ─────────────────────────────────────────────────────────────────
def train_model(name, config, X_train, y_train, X_val, y_val):
    print(f"\n  ▶ {name}")
    pipeline = _make_pipeline(config["estimator"])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=config["params"],
        n_iter=N_ITER,
        scoring=SCORE_METRIC,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate on validation set
    y_pred_val = best_model.predict(X_val)
    acc   = accuracy_score(y_val, y_pred_val)
    f1_ma = f1_score(y_val, y_pred_val, average="macro",    zero_division=0)
    f1_wt = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)

    print(f"     Best CV params : {search.best_params_}")
    print(f"     Val Accuracy   : {acc:.4f}")
    print(f"     Val Macro-F1   : {f1_ma:.4f}")
    print(f"     Val Weighted-F1: {f1_wt:.4f}")

    return best_model, {
        "accuracy":   acc,
        "macro_f1":   f1_ma,
        "weighted_f1": f1_wt,
        "best_params": str(search.best_params_),
    }


# ─────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────
def run_training():
    print("\n" + "=" * 60)
    print("  MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # ── Preprocess ────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     le, scaler) = run_preprocessing()

    results   = {}
    trained   = {}
    best_name = None
    best_f1   = -1.0

    # ── Train every model ─────────────────────────────────────────
    for name, config in MODELS.items():
        model, metrics = train_model(
            name, config, X_train, y_train, X_val, y_val
        )
        results[name] = metrics
        trained[name] = model

        if metrics["macro_f1"] > best_f1:
            best_f1   = metrics["macro_f1"]
            best_name = name

    # ── Print comparison table ────────────────────────────────────
    print("\n" + "─" * 60)
    print("  MODEL COMPARISON TABLE  (Validation Set)")
    print("─" * 60)
    hdr = f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>13}"
    print(hdr)
    print("─" * 60)
    for name, m in results.items():
        marker = " ◀ BEST" if name == best_name else ""
        print(
            f"{name:<25} {m['accuracy']:>10.4f} "
            f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>13.4f}{marker}"
        )
    print("─" * 60)
    print(f"\n  → Best model: {best_name}  (Macro-F1 = {best_f1:.4f})")

    # ── Save best model ───────────────────────────────────────────
    best_model = trained[best_name]
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  Saved best model → {model_path}")

    # Save all models (for evaluate.py)
    all_models_path = os.path.join(MODELS_DIR, "all_models.pkl")
    joblib.dump(trained, all_models_path)
    print(f"  Saved all models  → {all_models_path}")

    # Save comparison results as JSON
    results_path = os.path.join(MODELS_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved comparison  → {results_path}")

    # Save test split for evaluate.py
    test_data = {
        "X_test": X_test,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
    }
    joblib.dump(test_data, os.path.join(MODELS_DIR, "test_data.pkl"))

    return trained, best_name, le, X_test, y_test


if __name__ == "__main__":
    run_training()
