# 🧠 train_F1_model.md
## Formula 1 Race Result Prediction – Model Training Pipeline

---

## 1. Purpose of This Script

This script trains a **machine‑learning model to predict Formula 1 race finishing positions** using historical qualifying data, race outcomes, and season‑level performance metrics.

It consumes the datasets produced by the data ingestion pipeline (`download_F1_data.py`) and outputs a **ready‑to‑use prediction model** along with its supporting encoders.

The model is designed to:
- Predict expected finishing position (regression)
- Be robust across seasons
- Place higher emphasis on recent performance

---

## 2. Technologies Used

| Component | Role |
|--------|------|
| **Pandas** | Data loading, merging, feature engineering |
| **XGBoost** | Gradient‑boosted regression model |
| **Joblib** | Model and artifact persistence |
| **Pathlib** | File system safety |

---

## 3. Input Data

The script reads from the structured dataset:

```
f1_ml/
└── season_20XX/
    ├── features_Q_pre_race.csv
    └── labels_R.csv
```

For each season, features and labels are merged **per driver and per race**.

---

## 4. Season Weighting Strategy

Recent seasons better reflect:
- Current car regulations
- Team performance
- Competitive grid balance

Therefore, samples are weighted:

| Season | Weight |
|------|-------|
| 2026 | 2.0 |
| 2025 | 0.5 |
| 2024 | 0.2 |

These weights are passed directly to XGBoost using `sample_weight`.

---

## 5. Data Loading & Merging

For each season:

- Feature and label CSVs are loaded
- Primary keys are normalized as strings
- An inner join is performed on:

```
(DriverNumber, Season, Round)
```

This guarantees strict alignment between inputs and targets.

---

## 6. Feature Engineering

### 6.1 Season‑to‑Date Performance Metrics

All rolling metrics are computed using **lagged values** to avoid data leakage:

| Feature | Description |
|------|-------------|
| `races_so_far` | Number of races completed |
| `points_so_far` | Cumulative championship points |
| `avg_position_so_far` | Mean finishing position |
| `win_rate_so_far` | Win frequency |
| `podium_rate_so_far` | Podium frequency |

These features encode **driver form** across the season.

---

### 6.2 Grid Position Handling

Grid position is a highly predictive feature but may be missing.

The script applies a **hierarchical imputation strategy**:

1. Driver × Circuit historical mean
2. Driver‑only historical mean
3. Global grid median

A binary flag `grid_missing` is added to explicitly signal uncertainty.

This preserves predictive power without introducing bias.

---

### 6.3 Target Encoding (Categorical Features)

Instead of ordinal label encoding, the script applies **target encoding**:

- `EventName` → mean finishing position per circuit
- `TeamName` → mean finishing position per team

A smoothing factor reduces overfitting on low‑sample categories.

These encodings transform categorical knowledge into numeric signal while remaining model‑friendly.

---

## 7. Feature Set Used by the Model

```text
Event_enc, Team_enc,
Q_best_lap_s, Q_best_s1_s, Q_best_s2_s, Q_best_s3_s,
Q_quicklaps, Q_top_speed_kmh,
GridPosition, grid_missing,
races_so_far, points_so_far,
win_rate_so_far, podium_rate_so_far, avg_position_so_far
```

This combines:
- Raw pace (qualifying)
- Strategic context (grid)
- Driver form (season metrics)
- Structural knowledge (event & team strength)

---

## 8. Model Choice: XGBoost Regressor

XGBoost is used because it:
- Handles non‑linear interactions well
- Is robust to feature scaling
- Natively supports sample weighting
- Performs strongly on tabular sports data

Key hyperparameters:

| Parameter | Value |
|-------|------|
| Trees | 800 |
| Learning rate | 0.03 |
| Max depth | 6 |
| Subsample | 0.85 |
| L1 / L2 | 0.5 / 2.0 |

The model predicts **expected finishing position** as a continuous variable.

---

## 9. Model Training

```python
model.fit(X, y, sample_weight=w)
```

Weighted training ensures:
- Recent seasons dominate learning
- Older data stabilizes the model

This yields a bias‑variance balance suited for evolving motorsport contexts.

---

## 10. Saved Artifacts

The script outputs three files:

| File | Purpose |
|----|-------|
| `xgb_finish_position_model.pkl` | Trained regression model |
| `event_target_encoding.pkl` | Circuit encoding |
| `team_target_encoding.pkl` | Team encoding |

These artifacts are required for inference and must be loaded together.

---

## 11. How to Run

```bash
python train_F1_model.py
```

Run this after:
- Data download
- Data validation

---

## 12. ML Design Principles

✅ No data leakage  
✅ Temporal realism  
✅ Physics‑aware features  
✅ Interpretable structure  
✅ Production‑ready artifacts  

---

## 13. Summary

This script transforms raw F1 race data into a **high‑quality predictive model** capable of estimating race outcomes.

It is the **core intelligence layer** of the Formula 1 ML system.
