"""
eda.py
======
Exploratory Data Analysis for the Predictive Maintenance dataset.

Run this script AFTER placing the CSV in data/ to generate all
EDA visualizations saved to eda_plots/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Import preprocessing helpers (load + clean only; no splitting/scaling)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, clean_data, encode_and_engineer

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR  = os.path.join(BASE_DIR, "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Consistent color palette
PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", palette=PALETTE)


def _save(fig, filename: str) -> None:
    path = os.path.join(PLOT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# Plot 1 – Failure Type distribution
# ─────────────────────────────────────────────────────────────────
def plot_failure_distribution(df: pd.DataFrame) -> None:
    print("[EDA] Failure type distribution …")
    counts = df["Failure Type"].value_counts().reset_index()
    counts.columns = ["Failure Type", "Count"]
    total  = counts["Count"].sum()
    counts["Pct"] = counts["Count"] / total * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    sns.barplot(data=counts, x="Count", y="Failure Type",
                palette="tab10", ax=axes[0])
    axes[0].set_title("Failure Type – Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Samples")
    for bar, val in zip(axes[0].patches, counts["Count"]):
        axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                     f"{val}", va="center", fontsize=9)

    # Pie chart
    axes[1].pie(
        counts["Count"], labels=counts["Failure Type"],
        autopct="%1.1f%%", startangle=140,
        colors=sns.color_palette("tab10", len(counts))
    )
    axes[1].set_title("Failure Type – Proportions", fontsize=13, fontweight="bold")

    _save(fig, "01_failure_type_distribution.png")


# ─────────────────────────────────────────────────────────────────
# Plot 2 – Binary target (failure vs no failure)
# ─────────────────────────────────────────────────────────────────
def plot_binary_target(df: pd.DataFrame) -> None:
    print("[EDA] Binary target distribution …")
    if "Target" not in df.columns:
        return
    counts = df["Target"].value_counts().rename({0: "No Failure", 1: "Failure"})
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", color=["steelblue", "tomato"], ax=ax, edgecolor="black")
    ax.set_title("Machine Condition – Binary Target", fontsize=13, fontweight="bold")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No Failure", "Failure"], rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    _save(fig, "02_binary_target.png")


# ─────────────────────────────────────────────────────────────────
# Plot 3 – Machine Type distribution
# ─────────────────────────────────────────────────────────────────
def plot_machine_type(df: pd.DataFrame) -> None:
    print("[EDA] Machine type distribution …")
    # After encoding Type is numeric; work with original if available
    type_col = "Type_raw" if "Type_raw" in df.columns else "Type"
    if type_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    df[type_col].value_counts().sort_index().plot(
        kind="bar", color=sns.color_palette("Set2"), edgecolor="black", ax=ax
    )
    ax.set_title("Machine Type Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Machine Type")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    _save(fig, "03_machine_type_distribution.png")


# ─────────────────────────────────────────────────────────────────
# Plot 4 – Sensor reading distributions
# ─────────────────────────────────────────────────────────────────
def plot_sensor_distributions(df: pd.DataFrame) -> None:
    print("[EDA] Sensor distributions …")
    sensor_cols = [c for c in [
        "Air temperature K",
        "Process temperature K",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min",
    ] if c in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(sensor_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color=PALETTE[i], bins=40)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
    # Hide extra subplot
    for j in range(len(sensor_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sensor Reading Distributions", fontsize=14, fontweight="bold", y=1.01)
    _save(fig, "04_sensor_distributions.png")


# ─────────────────────────────────────────────────────────────────
# Plot 5 – Sensor readings by failure type (box plots)
# ─────────────────────────────────────────────────────────────────
def plot_sensors_by_failure(df: pd.DataFrame) -> None:
    print("[EDA] Sensor readings by failure type …")
    sensor_cols = [c for c in [
        "Air temperature K",
        "Process temperature K",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min",
    ] if c in df.columns]

    for col in sensor_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        order = df.groupby("Failure Type")[col].median().sort_values().index
        sns.boxplot(data=df, x="Failure Type", y=col, order=order,
                    palette="tab10", ax=ax)
        ax.set_title(f"{col}  by  Failure Type", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        plt.xticks(rotation=15)
        safe_name = col.replace(" ", "_").replace("/", "_")
        _save(fig, f"05_{safe_name}_by_failure.png")


# ─────────────────────────────────────────────────────────────────
# Plot 6 – Correlation heatmap
# ─────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    print("[EDA] Correlation heatmap …")
    num_df = df.select_dtypes(include="number").drop(
        columns=["Failure Type Encoded"], errors="ignore"
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        num_df.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    _save(fig, "06_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────
# Plot 7 – Scatter: Temperature vs Rotational Speed colored by failure
# ─────────────────────────────────────────────────────────────────
def plot_temp_vs_speed(df: pd.DataFrame) -> None:
    print("[EDA] Temperature vs Rotational Speed …")
    if not all(c in df.columns for c in [
        "Process temperature K", "Rotational speed rpm", "Failure Type"
    ]):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    failure_types = df["Failure Type"].unique()
    for ft, color in zip(sorted(failure_types), PALETTE):
        mask = df["Failure Type"] == ft
        ax.scatter(
            df.loc[mask, "Process temperature K"],
            df.loc[mask, "Rotational speed rpm"],
            label=ft, alpha=0.5, s=15, color=color
        )
    ax.set_xlabel("Process Temperature (K)")
    ax.set_ylabel("Rotational Speed (rpm)")
    ax.set_title("Process Temperature vs Rotational Speed  (by Failure Type)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "07_temp_vs_speed.png")


# ─────────────────────────────────────────────────────────────────
# Plot 8 – Torque vs Tool Wear
# ─────────────────────────────────────────────────────────────────
def plot_torque_vs_toolwear(df: pd.DataFrame) -> None:
    print("[EDA] Torque vs Tool Wear …")
    if not all(c in df.columns for c in ["Torque Nm", "Tool wear min", "Failure Type"]):
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    failure_types = sorted(df["Failure Type"].unique())
    for ft, color in zip(failure_types, PALETTE):
        mask = df["Failure Type"] == ft
        ax.scatter(
            df.loc[mask, "Tool wear min"],
            df.loc[mask, "Torque Nm"],
            label=ft, alpha=0.5, s=15, color=color
        )
    ax.set_xlabel("Tool Wear (min)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Tool Wear vs Torque  (by Failure Type)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "08_torque_vs_toolwear.png")


# ─────────────────────────────────────────────────────────────────
# Plot 9 – Pairplot of key sensors
# ─────────────────────────────────────────────────────────────────
def plot_pairplot(df: pd.DataFrame) -> None:
    print("[EDA] Pairplot (this may take a moment) …")
    cols = [c for c in [
        "Air temperature K",
        "Process temperature K",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min",
        "Failure Type",
    ] if c in df.columns]
    # Limit to 2000 rows for speed
    sample = df[cols].sample(min(2000, len(df)), random_state=42)
    g = sns.pairplot(sample, hue="Failure Type",
                     diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10})
    g.figure.suptitle("Pairplot of Key Sensor Features", y=1.01, fontsize=13, fontweight="bold")
    path = os.path.join(PLOT_DIR, "09_pairplot.png")
    g.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def run_eda():
    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    df_raw = load_data()
    df     = clean_data(df_raw)

    # Keep raw Type column for plot 3 before encoding
    if "Type" in df.columns:
        df["Type_raw"] = df["Type"].copy()

    df, le, _ = encode_and_engineer(df)

    plot_failure_distribution(df)
    plot_binary_target(df)
    plot_machine_type(df)
    plot_sensor_distributions(df)
    plot_sensors_by_failure(df)
    plot_correlation_heatmap(df)
    plot_temp_vs_speed(df)
    plot_torque_vs_toolwear(df)
    plot_pairplot(df)

    print(f"\n[EDA] All plots saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    run_eda()
