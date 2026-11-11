# Predictive Maintenance of Industrial Machinery
**Problem Statement No. 39 – Mechanical Engineering (Machine Learning)**

---

## 📋 Project Overview

This project builds a complete **Predictive Maintenance System** for industrial machinery. It uses real sensor and operational data to classify machine health and predict the type of failure before it occurs — helping reduce unexpected downtime and enable proactive maintenance.

**Predicted Classes:**
| Label | Description |
|---|---|
| No Failure | Machine is operating normally |
| Heat Dissipation Failure | Cooling / thermal regulation issue |
| Power Failure | Electrical supply issue |
| Overstrain Failure | Mechanical overload |
| Tool Wear Failure | Cutting tool degraded beyond limits |
| Random Failures | Unexpected, non-systematic failure |

---

## 🗂️ Project Structure

```
predictive_maintenance/
│
├── data/                          ← Place the CSV here (see below)
│   └── predictive_maintenance.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py              ← Data loading, cleaning, encoding, splitting
│   ├── eda.py                     ← Exploratory Data Analysis & visualisations
│   ├── train.py                   ← Model training & comparison
│   └── evaluate.py                ← Model evaluation & confusion matrices
│
├── models/                        ← Auto-created; stores trained model files
│   ├── best_model.pkl
│   ├── all_models.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── test_data.pkl
│   └── comparison_results.json
│
├── eda_plots/                     ← Auto-created; EDA visualisation PNGs
├── evaluation_plots/              ← Auto-created; evaluation chart PNGs
│
├── app.py                         ← Streamlit prediction application
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from Kaggle.

**Download link:**  
👉 https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

**Steps to import the dataset:**
1. Visit the link above and log in to your Kaggle account.
2. Click **Download** to get the ZIP file.
3. Extract and locate the file named `predictive_maintenance.csv`.
4. Copy or move it into the `data/` folder inside this project:
   ```
   predictive_maintenance/data/predictive_maintenance.csv
   ```

The dataset contains **10,000 rows** and the following columns:

| Column | Description |
|---|---|
| UDI | Unique identifier (dropped – not a feature) |
| Product ID | Product serial number (dropped – not a feature) |
| Type | Machine quality variant: L / M / H |
| Air temperature [K] | Ambient temperature in Kelvin |
| Process temperature [K] | Process heat in Kelvin |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool usage time |
| Target | Binary label: 0 = No Failure, 1 = Failure |
| Failure Type | Multi-class failure label (main target) |

---

## 🚀 Installation & Setup

### 1. Clone / download this project
```bash
git clone <your-repo-url>
cd predictive_maintenance
```

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the steps **in order**. Each step depends on the previous one.

### Step 1 – Place the Dataset
Ensure `data/predictive_maintenance.csv` exists (see Dataset section above).

### Step 2 – Data Preprocessing (standalone check)
```bash
python src/preprocess.py
```
This prints a dataset summary and saves the preprocessing pipeline to `models/`.

### Step 3 – Exploratory Data Analysis
```bash
python src/eda.py
```
Generates **9 visualisation plots** in `eda_plots/`.

### Step 4 – Train Models
```bash
python src/train.py
```
Trains 4 classifiers, compares them on the validation set, and saves the best model.  
⏱️ *This may take 3–8 minutes depending on your hardware.*

### Step 5 – Evaluate Models
```bash
python src/evaluate.py
```
Prints classification reports, confusion matrices, and a final comparison table.  
Charts are saved in `evaluation_plots/`.

### Step 6 – Launch the Prediction Application
```bash
streamlit run app.py
```
Opens the web application in your browser. Enter machine parameters and click **Predict**.

---

## 🧠 Machine Learning Pipeline

```
Raw CSV
   │
   ▼
Data Cleaning (drop duplicates, fill NaN, drop ID columns)
   │
   ▼
Feature Engineering
   ├── Encode machine type (L/M/H → 0/1/2)
   └── Add Temp Diff feature (Process – Air temperature)
   │
   ▼
Train / Validation / Test Split  (70% / 15% / 15%, stratified)
   │
   ▼
StandardScaler  ← fit on TRAIN ONLY (no data leakage)
   │
   ▼
SMOTE (inside CV folds only) → handles class imbalance
   │
   ▼
RandomizedSearchCV (5-fold stratified CV, Macro-F1 scoring)
   │
   ├── Logistic Regression
   ├── Decision Tree
   ├── Random Forest
   └── Gradient Boosting (HistGBM)
   │
   ▼
Best model selected by Macro-F1 on Validation Set
   │
   ▼
Evaluate on Test Set (never seen during training)
```

---

## 📊 Model Comparison

After running `train.py` + `evaluate.py`, a printed table shows:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Logistic Regression | 69.47% | 0.4322 | 0.8020 |
| Decision Tree | 92.07% | 0.5572 | 0.9479 |
| Random Forest | 95.80% | 0.6019 | 0.9685 |
| **Gradient Boosting** | **98.13%** | **0.6627** | **0.9842 ✅ BEST** |

**Why Macro-F1 is the primary metric:**  
Failure classes are rare (< 5% of data). Accuracy is misleading — a model that always predicts "No Failure" gets ~97% accuracy but is useless. Macro-F1 gives equal weight to every class, including rare failures.

---

## ⚠️ Understanding False Negatives

A **false negative** in this system means: the machine actually has a failure, but the model predicts "No Failure".

This is **the most dangerous error** in predictive maintenance because:
- The maintenance team takes no action.
- The machine continues running.
- An unexpected breakdown occurs — causing downtime, damage, or safety incidents.

The model is tuned to **maximise recall on failure classes** using:
- `class_weight="balanced"` in classifiers
- SMOTE oversampling
- Macro-F1 as the CV optimisation metric

---

## 🖥️ Streamlit Application Features

| Feature | Detail |
|---|---|
| Machine Type selector | L / M / H dropdown |
| Sensor sliders | Air temp, process temp, RPM, torque, tool wear |
| Prediction | Failure type with confidence % |
| Probability table | All class probabilities shown |
| Warning banner | Red alert when failure is predicted |
| Maintenance advice | Specific action steps per failure type |

---

## 🔧 Technical Decisions

| Decision | Reason |
|---|---|
| Drop UDI / Product ID | Random identifiers — no predictive value; would cause leakage |
| Add Temp Diff feature | Process – Air temp captures thermal stress better than each alone |
| StandardScaler on train only | Prevents data leakage from test/val statistics |
| SMOTE inside CV folds | Prevents synthetic samples from leaking into validation folds |
| Macro-F1 for model selection | Ensures rare failure classes are not ignored |
| HistGradientBoosting | Faster and more memory-efficient than XGBoost for tabular data |

---

## ⚡ Limitations

1. **Class imbalance is severe** – some failure types have < 20 samples. Even SMOTE may not fully compensate.
2. **Dataset is synthetic** – the AI4I dataset is simulation-based, not from real industrial sensors. Real-world deployment needs recalibration.
3. **No temporal features** – the dataset has no time-series structure. Sequential degradation patterns are not captured.
4. **Single-machine assumption** – the model is trained on a mixed dataset; a machine-specific model would likely perform better.

---

## 🔮 Future Improvements

- Use time-series models (LSTM, Temporal CNN) to capture degradation trends.
- Integrate a real-time sensor data stream (MQTT / OPC-UA).
- Deploy as a REST API (FastAPI + Docker) for integration with SCADA systems.
- Add SHAP explainability to show which sensors triggered a failure prediction.
- Retrain periodically with incoming sensor data (online learning).
- Extend to multi-machine environments with individual model instances.


## 🌐 Live Deployment

The application is deployed and publicly accessible on **Streamlit Community Cloud**:

👉 **[https://predictivemaintenance-kgttu8sbmnha9xrjn3e7me.streamlit.app](https://predictivemaintenance-kgttu8sbmnha9xrjn3e7me.streamlit.app)**

### How to use the live app:
1. Open the link above in any browser
2. Select **Machine Type** (L / M / H)
3. Adjust the sensor sliders (Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear)
4. Click **"Predict Machine Condition"**
5. View the predicted failure type, confidence score, probability breakdown, and maintenance recommendations

---


---

## 📜 License & Credits

- **Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) – Kaggle / Shivam Bansal
- **Project:** Developed as B.Tech CSE semester project – Problem Statement No. 39
- **Libraries:** scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, seaborn, streamlit, joblib
