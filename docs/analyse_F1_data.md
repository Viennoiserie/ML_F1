# 📊 analyse_F1_data.md
## Exploratory Data Analysis & Data Quality Validation for Formula 1 ML

---

## 1. Purpose of This Script

This script performs **systematic exploratory data analysis (EDA)** and **data quality validation** on the Formula 1 datasets produced by the data collection pipeline.

It is designed to:
- Validate data integrity before ML training
- Detect anomalies, missing values, and schema issues
- Visualize numeric distributions
- Inspect correlations between features and labels
- Confirm consistency across seasons

It acts as a **gatekeeper between data ingestion and model training**.

---

## 2. Libraries Used

| Library | Role |
|------|------|
| **NumPy** | Numerical utilities |
| **Pandas** | Data manipulation and joins |
| **Seaborn** | Statistical visualizations |
| **Matplotlib** | Plot rendering |
| **Pathlib** | Safe filesystem traversal |

Seaborn is configured with a clean white-grid style for readability.

---

## 3. Input Data Structure

The script scans the base directory:
```
f1_ml/
└── season_20XX/
    ├── features_Q_pre_race.csv
    └── labels_R.csv
```

Each season is analyzed independently, then compared globally.

---

## 4. Per‑Season Data Loading

For each season directory:

1. Season number is inferred from folder name
2. Feature and label CSV files are loaded
3. Key columns are normalized as strings
4. Features and labels are merged using:

 ```text
 (DriverNumber, Season, Round)
 ```

Only **inner joins** are performed to guarantee label‑feature alignment.

---

## 5. Basic Data Inspection

For each merged seasonal dataset, the script outputs:

### Column & Data Type Overview
- Ensures numeric / categorical fields are correctly typed
- Detects silent type coercion issues

### Missing Value Summary
- Counts NaNs per column
- Identifies partially populated features

---

## 6. Data Sanity Rules

The script applies domain‑specific validation rules:

| Rule | Reason |
|----|-------|
| `Q_best_lap_s > 0` | Lap times cannot be negative |
| `Q_top_speed_kmh <= 380` | Physical F1 car limits |
| `GridPosition <= 25` | Max grid size |

Any violation is reported for human inspection.

---

## 7. Numeric Feature Distributions

For up to 12 numeric columns per season:

- Histograms are generated
- Outliers and long‑tail behaviors are visualized
- Feature scaling needs can be identified

Plots are automatically arranged in a dynamic grid layout.

---

## 8. Correlation Analysis

If multiple numeric columns exist:

- A Pearson correlation matrix is computed
- Displayed as a heatmap

This exposes:
- Redundant features
- Strongly predictive signals
- Potential multi‑collinearity

---

## 9. Driver and Team Consistency Checks

The script verifies:

### Driver Participation
- Counts driver appearances per season
- Detects missing or duplicated entries

### Team Representation
- Counts races per team
- Identifies name changes or inconsistencies

These checks protect against silent category drift.

---

## 10. Cross‑Season Schema Validation

Once all seasons are processed:

- Common column sets are computed
- Any season missing required columns is reported

This ensures:
- Training and inference schemas remain aligned
- Feature engineering logic is safe across years

---

## 11. ML‑Focused Design Philosophy

This analysis enforces:

✅ Schema consistency  
✅ Temporal validity  
✅ Physical realism  
✅ Label‑feature correctness  
✅ Stable downstream modeling  

It is intentionally **diagnostic, not destructive**: no data is modified.

---

## 12. How to Run

```bash
python analyse_F1_data.py
```

This script should be run:
- After fresh data ingestion
- Before any model training
- Whenever new seasons are added

---

## 13. Summary

✅ Detects data issues early  
✅ Builds trust in training data  
✅ Prevents silent ML failures  
✅ Essential for production‑grade models

This script is the **data quality backbone** of the F1 ML project.
