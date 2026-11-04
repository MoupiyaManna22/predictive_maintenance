"""
app.py
======
Streamlit Predictive Maintenance Application
============================================
Loads the saved model + preprocessing pipeline and lets users enter
machine operating parameters to receive:
  - Predicted machine condition (Healthy / Failure)
  - Predicted failure type
  - Class probabilities / confidence
  - Clear maintenance recommendation

Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH   = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")


# ─────────────────────────────────────────────────────────────────
# Load artefacts (cached so they load only once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_artefacts():
    pipeline   = joblib.load(PIPELINE_PATH)
    best_model = joblib.load(BEST_MODEL_PATH)
    return pipeline, best_model


# ─────────────────────────────────────────────────────────────────
# Maintenance recommendations per failure type
# ─────────────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "No Failure": (
        "✅  Machine is operating normally. "
        "Continue routine maintenance schedule."
    ),
    "Heat Dissipation Failure": (
        "🌡️  Heat Dissipation Failure detected.\n\n"
        "**Actions:**\n"
        "- Inspect and clean cooling fans / heat sinks.\n"
        "- Check coolant levels and flow rate.\n"
        "- Verify that air vents are not blocked.\n"
        "- Monitor temperature sensors for calibration drift."
    ),
    "Power Failure": (
        "⚡  Power Failure risk detected.\n\n"
        "**Actions:**\n"
        "- Inspect power supply unit and wiring connections.\n"
        "- Check voltage and current readings against spec.\n"
        "- Test circuit breakers and fuses.\n"
        "- Schedule electrical system inspection."
    ),
    "Overstrain Failure": (
        "💪  Overstrain Failure risk detected.\n\n"
        "**Actions:**\n"
        "- Reduce operational load immediately.\n"
        "- Inspect mechanical couplings and drive belts.\n"
        "- Check for misalignment in rotating parts.\n"
        "- Verify torque limits are within design range."
    ),
    "Tool Wear Failure": (
        "🔧  Tool Wear Failure risk detected.\n\n"
        "**Actions:**\n"
        "- Inspect cutting/machining tools immediately.\n"
        "- Replace worn tools before next production run.\n"
        "- Review tool wear monitoring logs.\n"
        "- Adjust feeds and speeds to extend tool life."
    ),
    "Random Failures": (
        "❓  Random Failure risk detected.\n\n"
        "**Actions:**\n"
        "- Perform a comprehensive machine inspection.\n"
        "- Check all subsystems (electrical, mechanical, thermal).\n"
        "- Review recent maintenance and operating logs.\n"
        "- Consider scheduling a full diagnostic cycle."
    ),
}


# ─────────────────────────────────────────────────────────────────
# Predict function
# ─────────────────────────────────────────────────────────────────
def predict(pipeline: dict, model, user_input: dict) -> dict:
    """
    Transform raw user inputs using the saved pipeline and run inference.

    Returns a dict with:
        predicted_class   : str  (failure type label)
        confidence        : float (probability of predicted class, if available)
        probabilities     : dict  {class_name: probability}
        is_failure        : bool
    """
    scaler    = pipeline["scaler"]
    le        = pipeline["label_encoder"]
    type_map  = pipeline["type_map"]
    feat_cols = pipeline["feature_cols"]

    # Map machine type string → integer
    type_enc = type_map.get(user_input["Type"], 0)

    air_temp  = user_input["Air temperature K"]
    proc_temp = user_input["Process temperature K"]
    temp_diff = proc_temp - air_temp

    row = {
        "Type":                    type_enc,
        "Air temperature K":       air_temp,
        "Process temperature K":   proc_temp,
        "Rotational speed rpm":    user_input["Rotational speed rpm"],
        "Torque Nm":               user_input["Torque Nm"],
        "Tool wear min":           user_input["Tool wear min"],
        "Temp Diff K":             temp_diff,
    }

    # Build DataFrame with only the columns the model was trained on
    X = pd.DataFrame([[row[c] for c in feat_cols]], columns=feat_cols)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    pred_encoded = model.predict(X_scaled)[0]
    pred_label   = le.inverse_transform([pred_encoded])[0]

    # Probabilities (not all estimators support predict_proba)
    probabilities = {}
    confidence    = None
    clf = model.named_steps.get("clf", None) if hasattr(model, "named_steps") else model
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        probabilities = {le.inverse_transform([i])[0]: float(p)
                         for i, p in enumerate(proba)}
        confidence = float(probabilities.get(pred_label, 0.0))

    return {
        "predicted_class": pred_label,
        "confidence":       confidence,
        "probabilities":    probabilities,
        "is_failure":       pred_label != "No Failure",
    }


# ─────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────
def main():
    # Page config
    st.set_page_config(
        page_title="Predictive Maintenance System",
        page_icon="🏭",
        layout="wide",
    )

    # ── Header ─────────────────────────────────────────────────
    st.title("🏭 Predictive Maintenance System")
    st.markdown(
        "Enter the machine's current operating parameters to predict "
        "its health condition and potential failure type."
    )
    st.divider()

    # ── Check that models exist ────────────────────────────────
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(BEST_MODEL_PATH):
        st.error(
            "**Model files not found.**\n\n"
            "Please run the training pipeline first:\n"
            "```\n"
            "python src/train.py\n"
            "```\n"
            "Then restart this app."
        )
        return

    pipeline, model = load_artefacts()

    # ── Input panel ────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("⚙️  Machine Parameters")

        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            help="L = Low quality, M = Medium quality, H = High quality variant",
        )

        air_temp = st.slider(
            "Air Temperature (K)",
            min_value=290.0, max_value=310.0, value=298.0, step=0.1,
            help="Ambient air temperature in Kelvin",
        )

        proc_temp = st.slider(
            "Process Temperature (K)",
            min_value=300.0, max_value=320.0, value=308.0, step=0.1,
            help="Internal process temperature in Kelvin",
        )

        rot_speed = st.slider(
            "Rotational Speed (rpm)",
            min_value=1000, max_value=3000, value=1500, step=10,
            help="Machine spindle / motor rotational speed",
        )

        torque = st.slider(
            "Torque (Nm)",
            min_value=1.0, max_value=80.0, value=40.0, step=0.5,
            help="Applied torque in Newton-metres",
        )

        tool_wear = st.slider(
            "Tool Wear (min)",
            min_value=0, max_value=250, value=100, step=1,
            help="Cumulative tool wear time in minutes",
        )

        predict_btn = st.button("🔍  Predict Machine Condition", type="primary", use_container_width=True)

    # ── Output panel ───────────────────────────────────────────
    with col2:
        st.subheader("📊  Prediction Results")

        if not predict_btn:
            st.info("Adjust the parameters on the left and click **Predict** to see results.")
        else:
            user_input = {
                "Type":                  machine_type,
                "Air temperature K":     air_temp,
                "Process temperature K": proc_temp,
                "Rotational speed rpm":  rot_speed,
                "Torque Nm":             torque,
                "Tool wear min":         tool_wear,
            }

            with st.spinner("Running prediction …"):
                result = predict(pipeline, model, user_input)

            pred_class = result["predicted_class"]
            is_failure = result["is_failure"]
            confidence = result["confidence"]

            # ── Status banner ──────────────────────────────────
            if is_failure:
                st.error(f"⚠️  **WARNING – Failure Detected!**")
                st.metric("Predicted Failure Type", pred_class)
            else:
                st.success(f"✅  **Machine is Healthy**")
                st.metric("Predicted Condition", pred_class)

            # ── Confidence ────────────────────────────────────
            if confidence is not None:
                st.progress(
                    min(confidence, 1.0),
                    text=f"Model Confidence: **{confidence*100:.1f}%**"
                )

            # ── Probability breakdown ─────────────────────────
            if result["probabilities"]:
                st.markdown("**Probability Breakdown by Class:**")
                prob_df = (
                    pd.DataFrame
                    .from_dict(result["probabilities"], orient="index", columns=["Probability"])
                    .sort_values("Probability", ascending=False)
                    .reset_index()
                    .rename(columns={"index": "Failure Type"})
                )
                prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
                st.dataframe(
                    prob_df[["Failure Type", "Probability (%)"]],
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Recommendation ────────────────────────────────
            st.divider()
            st.subheader("🛠️  Maintenance Recommendation")
            rec_text = RECOMMENDATIONS.get(pred_class, "Perform a full diagnostic inspection.")
            if is_failure:
                st.warning(rec_text)
            else:
                st.success(rec_text)

    # ── Footer / info ──────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️  About this Application"):
        st.markdown(
            """
**Predictive Maintenance System** – Problem Statement 39  
*Mechanical Engineering × Machine Learning*

This application uses a machine learning model trained on the  
[UCI Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
to predict potential industrial machine failures from sensor readings.

**Features used for prediction:**
| Feature | Description |
|---|---|
| Machine Type | Quality variant (L / M / H) |
| Air Temperature | Ambient temperature in Kelvin |
| Process Temperature | Process heat in Kelvin |
| Rotational Speed | Spindle speed in RPM |
| Torque | Applied torque in Nm |
| Tool Wear | Cumulative tool usage in minutes |
| Temp Diff | Derived: process – air temperature |

**Target classes:** No Failure · Heat Dissipation · Power Failure · Overstrain · Tool Wear · Random Failures

**Important:** This system assists maintenance decisions but should be  
used alongside expert judgment and physical inspection.
            """
        )


if __name__ == "__main__":
    main()
