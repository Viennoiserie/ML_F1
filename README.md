# 🏎️ Formula 1 Machine Learning Project

## Predicting Race Outcomes with Data‑Driven Intelligence

---

## 📌 Project Overview

This repository contains a **complete, end‑to‑end machine‑learning pipeline** designed to predict **Formula 1 race outcomes** using historical qualifying data, race results, telemetry‑derived signals, and season‑level performance metrics.

The project is built with a **production‑grade ML mindset**:

- ✅ No data leakage
- ✅ Temporal realism
- ✅ Clear separation of concerns
- ✅ Reproducible datasets and models
- ✅ Interpretable predictions

It is structured to reflect the *real‑world lifecycle* of a machine‑learning system:

> **Data Ingestion → Validation → Training → Evaluation → Inference**

---

## 🎯 Objectives

The primary goal of this project is to:

- Predict the **expected finishing position** of each driver in a Formula 1 race
- Rank drivers accordingly
- Produce a **predicted podium (Top‑3)** for upcoming races

Rather than framing the problem as a simple classification task, the model performs **regression**, enabling more stable and interpretable race‑level rankings.

---

## 🧠 Modeling Philosophy

Formula 1 presents unique ML challenges:

- Small datasets (≈20 races / season)
- Strong temporal autocorrelation
- High variance due to incidents, weather, and safety cars

This project explicitly embraces these constraints by:

- Using **season‑to‑date performance features**
- Weighting recent seasons more heavily than older ones
- Avoiding unrealistic offline validation claims

The result is a model that estimates **expected performance**, not guaranteed outcomes.

---

## 🗂 Repository Structure

```
ML_F1/
│
├── data/
│   └── f1_ml/
│       ├── season_20XX/
│       │   ├── schedule.csv
│       │   ├── features_Q_pre_race.csv
│       │   └── labels_R.csv
│       ├── features_Q_pre_race_all.csv
│       └── labels_R_all.csv
│
├── models/
│   ├── xgb_finish_position_model.pkl
│   ├── event_target_encoding.pkl
│   └── team_target_encoding.pkl
│
├── download_F1_data.py
├── analyse_F1_data.py
├── train_F1_model.py
├── evaluate_F1_model.py
├── infer_F1_model.py
│
├── download_F1_data.md
├── analyse_F1_data.md
├── train_F1_model.md
├── evaluate_F1_model.md
├── infer_F1_model.md
│
└── README.md
```

---

## 🔄 Pipeline Components

### 1️⃣ Data Download – `download_F1_data.py`

- Uses **FastF1** to download official F1 data
- Extracts qualifying lap times, telemetry, and race results
- Produces clean, ML‑ready CSV files
- Fully cached for efficiency

📄 Documentation: `download_F1_data.md`

---

### 2️⃣ Data Analysis – `analyse_F1_data.py`

- Performs systematic exploratory data analysis (EDA)
- Validates schemas, missing values, physical constraints
- Visualizes distributions and feature correlations
- Ensures cross‑season consistency

📄 Documentation: `analyse_F1_data.md`

---

### 3️⃣ Model Training – `train_F1_model.py`

- Engineers season‑to‑date performance features
- Applies hierarchical GridPosition imputation
- Uses **target encoding** for circuit and team strength
- Trains a **weighted XGBoost regressor**

📄 Documentation: `train_F1_model.md`

---

### 4️⃣ Model Evaluation – `evaluate_F1_model.py`

- Evaluates the model on past seasons
- Reports MAE, RMSE, winner accuracy, and podium overlap
- Designed as a **sanity and stability check**, not a competition benchmark

📄 Documentation: `evaluate_F1_model.md`

---

### 5️⃣ Live Inference – `infer_F1_model.py`

- Predicts outcomes for upcoming races
- Uses historical circuit templates and current‑season form
- Handles missing grid and qualifying data safely
- Outputs a predicted podium

📄 Documentation: `infer_F1_model.md`

---

## 🧩 Features Used

The model combines multiple signal layers:

- **Qualifying performance** (lap & sector times, top speed)
- **Grid context** (with uncertainty flagging)
- **Current‑season driver form**
- **Structural knowledge** (circuit & team strength)

This hybrid representation allows the model to generalize despite noisy race dynamics.

---

All documentation is AI generated 

---

## ⚠️ Limitations

This project intentionally does **not** model:

- Weather conditions
- Safety cars & incidents
- Mechanical failures
- Strategy randomness

Predictions should be interpreted as **expected performance under normal conditions**.

---

## ▶️ How to Run the Full Pipeline

```bash
# 1. Download and refresh data
python download_F1_data.py

# 2. Validate and explore data
python analyse_F1_data.py

# 3. Train the model
python train_F1_model.py

# 4. Evaluate model sanity
python evaluate_F1_model.py

# 5. Predict an upcoming race
python infer_F1_model.py
```

---

## ✅ Project Status

✔ End‑to‑end pipeline implemented  
✔ Fully documented  
✔ ML‑safe and reproducible  
✔ Ready for iteration and extension  

---

## 📈 Possible Extensions

- Monte‑Carlo race simulations
- Confidence intervals for predictions
- Weather and tyre strategy modeling
- Live retraining after each race

---
