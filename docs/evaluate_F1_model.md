# ✅ evaluate_F1_model.md
## Model Evaluation & Validation – Formula 1 Race Prediction

---

## 1. Purpose of This Script

This script evaluates the **performance of the trained F1 ML model** on historical race data.

Its objective is **not to claim real‑world predictive superiority**, but to:

- Validate that the end‑to‑end ML pipeline is functional
- Verify feature / label alignment
- Sanity‑check model behavior on known data
- Provide interpretable diagnostics at race and season level

This script completes the **formal ML pipeline**:

> **Download → Analyse → Train → Evaluate → Infer**

---

## 2. Important Conceptual Limitation (F1‑Specific)

As noted in the script header, Formula 1 imposes **intrinsic evaluation limits**:

- The dataset is **small** (≈20 races per season)
- Driver performance is **highly autocorrelated**
- The model often evaluates races it has partially learned from

Therefore, this evaluation should be interpreted as:

✅ *Pipeline verification*  
✅ *Relative signal quality check*  
❌ *Not a realistic future‑race benchmark*

The truly correct evaluation loop would be:

1. Predict **before qualifying**
2. Predict **after qualifying**
3. Compare predictions once the race is complete
4. Retrain on newly observed data

---

## 3. Libraries Used

| Library | Purpose |
|------|------|
| **Pandas** | Data manipulation |
| **NumPy** | Numeric utilities |
| **Scikit‑learn** | Regression metrics |
| **Joblib** | Model loading |
| **Pathlib** | File handling |

---

## 4. Evaluation Scope

```python
EVAL_SEASONS = [2024, 2025]
```

Only these seasons are evaluated to:
- Avoid evaluating on the most recent training season
- Simulate "past‑season" validation

---

## 5. Model & Encoder Loading

The script loads:

- Trained XGBoost regression model
- Event encoder
- Team encoder

All artifacts must match the training pipeline exactly.

---

## 6. Dataset Construction

### 6.1 Per‑Season Loading

For every season directory:

- Features and labels are loaded
- Data is merged on:

```text
(DriverNumber, Season, Round)
```

Only **inner joins** are used to prevent feature/label mismatch.

---

### 6.2 Evaluation Filtering

After merging, the dataset is **restricted to evaluation seasons only**.

This prevents accidental information leakage from future seasons.

---

## 7. Feature Engineering (Evaluation Phase)

The same season‑to‑date features used during training are recomputed:

| Feature | Meaning |
|------|--------|
| `races_so_far` | Number of completed races |
| `points_so_far` | Championship points |
| `avg_position_so_far` | Mean finishing position |
| `win_rate_so_far` | Win frequency |
| `podium_rate_so_far` | Podium frequency |

All metrics use **lagged data only**, ensuring temporal correctness.

---

## 8. Feature Encoding

Categorical variables are encoded using the **same encoders as training**:

- Event circuit encoding
- Team performance encoding

This ensures feature space consistency.

---

## 9. Regression Evaluation Metrics

### 9.1 Global Metrics

Two standard regression metrics are computed:

| Metric | Interpretation |
|------|---------------|
| **MAE** | Average error in finishing positions |
| **RMSE** | Penalizes large prediction errors |

These metrics answer:
> *On average, how many positions off is the model?*

---

## 10. Race‑Level Evaluation Logic

Each race is evaluated independently.

### 10.1 Winner Accuracy

Checks whether the predicted P1 matches the true winner.

### 10.2 Winner in Top‑3 Accuracy

Checks whether the actual winner appears in the predicted podium.

### 10.3 Podium Overlap Score

Computes:

```text
|predicted podium ∩ actual podium|
```

Range: **0 → 3**

This metric is stable even when ordering is slightly incorrect.

---

## 11. Reported Race Metrics

| Metric | Meaning |
|------|--------|
| Winner accuracy | Exact P1 prediction rate |
| Winner in Top‑3 | Probability winner appears in top‑3 prediction |
| Avg podium overlap | Mean podium alignment |

These metrics align better with **real F1 decision‑making** than pure regression loss.

---

## 12. Interpretation Guidelines

✅ High podium overlap → correct performance tier
✅ High winner‑in‑top‑3 → strategic relevance
❌ Low MAE ≠ guaranteed real‑world accuracy

The metrics should be used for **relative comparison**, not absolute claims.

---

## 13. How to Run

```bash
python evaluate_F1_model.py
```

Run after:
- Model training
- Before inference deployment

---

## 14. Summary

✅ Confirms model sanity  
✅ Validates feature pipeline  
✅ Detects catastrophic errors  
✅ Completes the ML lifecycle  

This script ensures the **F1 ML system behaves coherently** before being used for live predictions.
