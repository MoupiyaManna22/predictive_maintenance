"""
preprocess.py
=============
Loads the raw CSV, cleans it, engineers features, encodes categories,
and splits into train / validation / test sets.

The preprocessing pipeline (scaler + encoder) is saved to disk so the
Streamlit app can reuse it without ever re-fitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "predictive_maintenance.csv")


# ─────────────────────────────────────────────────────────────────
# Helper: column name normalizer
# ─────────────────────────────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace special chars in column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[\[\]<>]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


# ─────────────────────────────────────────────────────────────────
# Step 1 – Load
# ─────────────────────────────────────────────────────────────────
def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {csv_path}\n"
            "Please download it from:\n"
            "  https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification\n"
            "and save the CSV as:  data/predictive_maintenance.csv\n"
        )
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────────────────────────
# Step 2 – Inspect
# ─────────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("\n── Data Types ──────────────────────────────────────────")
    print(df.dtypes)
    print("\n── Missing Values ──────────────────────────────────────")
    print(df.isnull().sum())
    print("\n── Duplicate Rows ──────────────────────────────────────")
    print(f"  {df.duplicated().sum()} duplicate rows found.")
    print("\n── Target Distribution (Failure Type) ──────────────────")
    if "Failure Type" in df.columns:
        print(df["Failure Type"].value_counts())
    print("\n── Machine Condition (Target) ───────────────────────────")
    if "Target" in df.columns:
        print(df["Target"].value_counts())
    print()


# ─────────────────────────────────────────────────────────────────
# Step 3 – Clean
# ─────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicate rows
    - Drop unnecessary identifier columns (UDI, Product ID) – they are
      random identifiers with no predictive value and would cause data
      leakage if used.
    - Fill any rare NaN (the dataset is nearly complete).
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before - after:
        print(f"[clean_data] Dropped {before - after} duplicate rows.")

    # Drop ID columns if present
    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[clean_data] Dropped identifier columns: {drop_cols}")

    # Fill numeric NaN with column median (defensive; dataset has none)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaN with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[clean_data] Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────────────────────────
# Step 4 – Encode & Feature Engineering
# ─────────────────────────────────────────────────────────────────
def encode_and_engineer(df: pd.DataFrame):
    """
    - Ordinal-encode 'Type' (machine type L/M/H) → numeric
    - Label-encode 'Failure Type' (multi-class target)
    - Keep 'Target' (binary failure/no-failure) as-is (0/1)
    - Add a temperature difference feature (process – air temperature)

    Returns:
        df_encoded     : full encoded DataFrame
        label_encoder  : fitted LabelEncoder for Failure Type
        type_map       : dict mapping Type → integer
    """
    type_order = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["Type"] = df["Type"].map(type_order)
        # Handle any unknown values
        df["Type"] = df["Type"].fillna(df["Type"].median()).astype(int)

    # Temperature difference feature
    if "Air temperature K" in df.columns and "Process temperature K" in df.columns:
        df["Temp Diff K"] = df["Process temperature K"] - df["Air temperature K"]

    # Encode the multi-class failure-type target
    le = LabelEncoder()
    if "Failure Type" in df.columns:
        df["Failure Type Encoded"] = le.fit_transform(df["Failure Type"])

    return df, le, type_order


# ─────────────────────────────────────────────────────────────────
# Step 5 – Split
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Type",
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
    "Temp Diff K",
]

def split_data(df: pd.DataFrame):
    """
    Splits into train (70%) / validation (15%) / test (15%).
    Uses stratification on the multi-class target to preserve
    class proportions in every split.

    Returns X_train, X_val, X_test, y_train, y_val, y_test
    (y = Failure Type Encoded, the multi-class label)
    """
    # Only keep feature columns that actually exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available_features].copy()
    y = df["Failure Type Encoded"].copy()

    # 70 / 30 first split, then 30 → 50/50 for val/test ≈ 15 % each
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"[split_data] Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────
# Step 6 – Scale & Save pipeline
# ─────────────────────────────────────────────────────────────────
def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler ONLY on training data; transform all splits.
    Prevents data leakage from validation/test statistics.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_val_sc, X_test_sc, scaler


def save_pipeline(scaler, label_encoder, type_map, feature_cols):
    """Persist the preprocessing artefacts for the Streamlit app."""
    pipeline = {
        "scaler":        scaler,
        "label_encoder": label_encoder,
        "type_map":      type_map,
        "feature_cols":  feature_cols,
    }
    path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    joblib.dump(pipeline, path)
    print(f"[save_pipeline] Saved preprocessing pipeline → {path}")


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────
def run_preprocessing():
    """Execute all preprocessing steps and return ready-to-train arrays."""
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    df, le, type_map = encode_and_engineer(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(
        X_train, X_val, X_test
    )
    save_pipeline(scaler, le, type_map, available_features)

    return (
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        le, scaler
    )


if __name__ == "__main__":
    run_preprocessing()
