# 🚀 infer_F1_model.md
## Live Inference & Race Outcome Prediction – Formula 1 ML

---

## 1. Purpose of This Script

This script performs **live inference** using the trained Formula 1 machine‑learning model.

Its goal is to:
- Predict expected finishing positions for an **upcoming race**
- Rank drivers according to predicted performance
- Output a **predicted podium (Top‑3)**

It represents the **final stage** of the end‑to‑end ML pipeline:

> **Download → Analyse → Train → Evaluate → Infer**

---

## 2. Position in the ML Pipeline

Unlike training or evaluation scripts, this script:

- Does **not** learn from data
- Does **not** compute metrics
- Operates in a **forward‑looking context**

It uses:

✅ Historical data  
✅ Current‑season form  
✅ Qualifying‑based performance signals

To generate **actionable race predictions**.

---

## 3. Libraries Used

| Library | Purpose |
|------|-------|
| **Pandas** | Data manipulation |
| **Joblib** | Model & artifact loading |
| **Pathlib** | File system robustness |

---

## 4. Loaded Artifacts

The script loads the following trained assets:

| File | Role |
|----|----|
| `xgb_finish_position_model.pkl` | Regression model predicting finish position |
| `event_target_encoding.pkl` | Circuit strength encoding |
| `team_target_encoding.pkl` | Team performance encoding |

These artifacts must come from the same training run.

---

## 5. Input Data Sources

The script uses **historical datasets only**:

```
f1_ml/
└── season_20XX/
    ├── features_Q_pre_race.csv
    └── labels_R.csv
```

No future information is required.

---

## 6. Race Selection Logic

### 6.1 Target Season

```python
PREDICT_SEASON = 2026
```

Predictions are made for races in this season that:
- Exist in last year’s calendar
- Have **not yet been completed**

---

### 6.2 Calendar Construction

The calendar is built by:

1. Taking the previous season’s schedule
2. Removing races already run in the target season
3. Presenting the remaining races for user selection

This design allows **manual control** over which race to predict.

---

## 7. Race Template Construction

For the selected race:

- Driver list and baseline structure are copied from the **same circuit in the previous season**
- This ensures realistic grid composition and team presence

This race acts as a **structural template**, not training data.

---

## 8. Dynamic Season‑Form Features

The model incorporates **current‑season performance** using live statistics:

| Feature | Description |
|------|-------------|
| `races_so_far` | Races completed so far |
| `points_so_far` | Points accumulated |
| `avg_position_so_far` | Average finishing position |
| `win_rate_so_far` | Win frequency |
| `podium_rate_so_far` | Podium frequency |

If no race has yet been run, values default safely to zero.

---

## 9. Grid Position Handling

Grid position is highly predictive but may be unknown at inference time.

The script applies the **same hierarchical logic as training**:

1. Historical driver × circuit mean
2. Historical driver‑level mean
3. Global grid median

A binary feature `grid_missing` explicitly signals uncertainty to the model.

This preserves predictive validity while remaining ML‑safe.

---

## 10. Qualifying Feature Completion

Qualifying metrics may be absent or partially populated.

For each expected qualifying feature:

| Feature | Source |
|------|-------|
| Lap & sector times | Historical driver averages |
| Top speed | Telemetry‑derived averages |

Missing values are filled using **driver‑specific historical averages**, then global means.

---

## 11. Target Encoding During Inference

Categorical features are encoded using **pre‑trained target encodings**:

- `Event_enc` captures circuit difficulty
- `Team_enc` captures team competitiveness

Unseen values fall back to the global position mean.

No re‑training or fitting occurs during inference.

---

## 12. Feature Vector Used for Prediction

```text
Event_enc, Team_enc,
Q_best_lap_s, Q_best_s1_s, Q_best_s2_s, Q_best_s3_s,
Q_quicklaps, Q_top_speed_kmh,
GridPosition, grid_missing,
races_so_far, points_so_far,
win_rate_so_far, podium_rate_so_far, avg_position_so_far
```

This vector exactly matches the training feature space.

---

## 13. Prediction & Ranking Logic

The model predicts a **continuous expected finishing position** for each driver.

Drivers are then:
1. Sorted by predicted position (ascending)
2. Ranked accordingly
3. Top‑3 drivers are extracted as the predicted podium

This step transforms regression output into a race‑level decision.

---

## 14. Output

The script prints:

```text
P1: Driver Name (Expected position: X.xx)
P2: Driver Name (Expected position: X.xx)
P3: Driver Name (Expected position: X.xx)
```

This output is suitable for:
- Strategy discussion
- Media previews
- Simulation studies

---

## 15. Limitations & Intended Usage

⚠️ Predictions are **conditional on qualifying‑era information**  
⚠️ Weather, incidents, penalties, and safety cars are not modeled  
⚠️ The output is probabilistic, not deterministic

The model estimates **expected performance**, not guaranteed outcomes.

---

## 16. How to Run

```bash
python infer_F1_model.py
```

The script is typically run:
- After qualifying
- Before race day

---

## 17. Summary

✅ Uses only historical information  
✅ No data leakage  
✅ Consistent with training assumptions  
✅ Produces interpretable, race‑level predictions  

This script is the **operational endpoint** of the Formula 1 ML system.
