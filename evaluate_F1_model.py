import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

"""
I created this to make a clean complete ML pipeline, but in the case of F1 :
I cannot create 'objective' evaluations -> since the model learns from very limited values
When asked to perform on known races, it essentialy iterates over learnt data. The real deal would be to :

- predict races before qualifying
- predict them again after qualifying
- compare both results to reality once the race is over

--> then retrain over the new data
"""

# Config

BASE_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\data\f1_ml")
MODELS_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\models")

EVAL_SEASONS = [2024, 2025]  

# region : Loading 

# Models

model    = joblib.load(MODELS_DIR / "xgb_finish_position_model.pkl")
le_event = joblib.load(MODELS_DIR / "event_encoder.pkl")
le_team  = joblib.load(MODELS_DIR / "team_encoder.pkl")

print("Model and encoders loaded")

# Data

dfs = []

for season_dir in BASE_DIR.glob("season_*"):

    feat_file = season_dir / "features_Q_pre_race.csv"
    lab_file  = season_dir / "labels_R.csv"

    if not feat_file.exists() or not lab_file.exists():
        continue

    feat = pd.read_csv(feat_file)
    lab  = pd.read_csv(lab_file)

    df = feat.merge(lab,
                    on=["DriverNumber", "Season", "Round"],
                    how="inner")

    df["Season"] = df["Season"].astype(int)
    df["Round"]  = df["Round"].astype(int)

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Keep only evaluation seasons
data = data[data["Season"].isin(EVAL_SEASONS)].copy()

# endregion

# region : Feature engineering 

data = data.sort_values(["Season", "Round"])

grp = data.groupby(["Season", "DriverNumber"])

data["races_so_far"] = grp.cumcount()
data["points_so_far"] = grp["Points"].transform(lambda x: x.shift().cumsum())
data["avg_position_so_far"] = grp["Position"].transform(lambda x: x.shift().expanding().mean())
data["win_rate_so_far"] = grp["Position"].transform(lambda x: (x.shift() == 1).expanding().mean())
data["podium_rate_so_far"] = grp["Position"].transform(lambda x: (x.shift() <= 3).expanding().mean())

ROLLING_COLS = ["points_so_far",
                "avg_position_so_far",
                "win_rate_so_far",
                "podium_rate_so_far"]

data[ROLLING_COLS] = data[ROLLING_COLS].fillna(0)

# Encoding
data["Event_enc"] = le_event.transform(data["EventName_x"])
data["Team_enc"]  = le_team.transform(data["TeamName"])

# endregion

# Predictions

FEATURES = ["Event_enc",
            "Team_enc",

            "Q_best_lap_s",
            "Q_best_s1_s",
            "Q_best_s2_s",
            "Q_best_s3_s",
            "Q_quicklaps",
            "GridPosition",
            "Q_top_speed_kmh",

            "races_so_far",
            "points_so_far",
            "win_rate_so_far",
            "podium_rate_so_far",
            "avg_position_so_far"]

X = data[FEATURES]
y_true = data["Position"]

data["predicted_position"] = model.predict(X)

# region : Evaluation

# Metrics

mae  = mean_absolute_error(y_true, data["predicted_position"])
rmse = root_mean_squared_error(y_true, data["predicted_position"])

print("\n===== Global Regression Metrics =====")
print(f"MAE  : {mae:.2f} positions")
print(f"RMSE : {rmse:.2f} positions")

# Races

winner_hits = 0
top3_hits   = 0
podium_overlap = []

n_races = 0

for (season, rd), race in data.groupby(["Season", "Round"]):

    n_races += 1

    race_sorted = race.sort_values("predicted_position")
    true_sorted = race.sort_values("Position")

    pred_winner = race_sorted.iloc[0]["DriverNumber"]
    true_winner = true_sorted.iloc[0]["DriverNumber"]

    if pred_winner == true_winner:
        winner_hits += 1

    pred_top3 = set(race_sorted.head(3)["DriverNumber"])
    true_top3 = set(true_sorted.head(3)["DriverNumber"])

    if true_winner in pred_top3:
        top3_hits += 1

    podium_overlap.append(len(pred_top3 & true_top3))

winner_acc = winner_hits / n_races
top3_acc   = top3_hits / n_races
mean_podium_overlap = np.mean(podium_overlap)

print("\n===== Race-level Metrics =====")
print(f"Winner accuracy         : {winner_acc:.2%}")
print(f"Winner in predicted top3: {top3_acc:.2%}")
print(f"Avg podium overlap (0-3): {mean_podium_overlap:.2f}")

# endregion