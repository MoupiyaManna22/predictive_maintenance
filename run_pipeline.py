"""
run_pipeline.py
===============
Convenience script to run the full ML pipeline end-to-end.

Usage:
    python run_pipeline.py

This script runs:
  1. Data Preprocessing
  2. Exploratory Data Analysis
  3. Model Training
  4. Model Evaluation

After this script finishes, launch the app with:
    streamlit run app.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

print("\n" + "█" * 60)
print("  PREDICTIVE MAINTENANCE – FULL ML PIPELINE")
print("█" * 60)

# ── Step 1: EDA ────────────────────────────────────────────────
print("\n[1/3]  Running Exploratory Data Analysis …\n")
from eda import run_eda
run_eda()

# ── Step 2: Train ──────────────────────────────────────────────
print("\n[2/3]  Training Models …\n")
from train import run_training
run_training()

# ── Step 3: Evaluate ───────────────────────────────────────────
print("\n[3/3]  Evaluating Models …\n")
from evaluate import run_evaluation
run_evaluation()

print("\n" + "█" * 60)
print("  PIPELINE COMPLETE")
print("  Launch the app with:  streamlit run app.py")
print("█" * 60 + "\n")
