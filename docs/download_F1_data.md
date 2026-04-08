# 📥 download_F1_data.md
## Formula 1 Data Collection Pipeline (FastF1)

---

## 1. Purpose of This Script

This script automatically downloads, cleans, and structures Formula 1 data using the **FastF1** Python library.

Its goal is to create **machine-learning-ready datasets** containing:

- Pre-race qualifying and telemetry features
- Race result labels
- Official schedules
- Season-separated CSV files for modelling and inference

This script is the **data foundation** of the F1 ML project.

---

## 2. Technologies Used

- **FastF1** – Official F1 timing, lap, and telemetry data
- **Pandas** – Data processing and aggregation
- **Pathlib** – Robust file system interaction
- **FastF1 Cache** – Efficient reuse of downloaded data

---

## 3. Output Directory Structure

```
~/Downloads/f1_ml/
│
├── fastf1_cache/
│
├── season_20XX/
│   ├── schedule.csv
│   ├── features_Q_pre_race.csv
│   └── labels_R.csv
│
├── features_Q_pre_race_all.csv
└── labels_R_all.csv
```

---

## 4. Seasons Covered

The script automatically collects data for **three seasons**:

- Current year
- Previous year
- Two years prior

This ensures models learn from recent regulations while retaining historical stability.

---

## 5. FastF1 Cache

FastF1 caching is enabled to prevent redundant downloads and improve performance when re-running the script.

---

## 6. Schedule Collection

For each season:

- The official F1 calendar is downloaded
- Future races are filtered out
- Round numbers are normalized
- The schedule is saved as `schedule.csv`

If the API fails, a local backup is reused.

---

## 7. Qualifying Features (Pre-Race)

From the qualifying session, the script computes per-driver features:

- Best lap time
- Best sector times
- Number of fast laps
- Maximum top speed from telemetry

These features represent **pure pace before the race**.

---

## 8. Race Labels

From the race session the script extracts:

- Grid position
- Final position
- Classified position
- Points scored
- Race status (finished, DNF, etc.)

These are the **ground-truth targets** for supervised learning.

---

## 9. ML-Safe Design

The pipeline strictly separates:

- Features known *before* the race
- Labels known *after* the race

This prevents data leakage and ensures realistic model performance.

---

## 10. Rate Limiting

A short delay is added between API calls to ensure stability and avoid service throttling.

---

## 11. Combined Datasets

At the end of execution, global datasets are saved:

- `features_Q_pre_race_all.csv`
- `labels_R_all.csv`

These files are useful for exploration and diagnostics.

---

## 12. How to Run

```bash
python download_F1_data.py
```

Run once per season update or whenever data refresh is required.

---

## 13. Summary

✅ Fully automated
✅ Fault tolerant
✅ ML-ready
✅ Season-aware
✅ Reproducible

This script enables the entire F1 machine learning pipeline.