import joblib
import pandas as pd

from pathlib import Path
from xgboost import XGBRegressor

# Config

BASE_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\data\f1_ml")

MODELS_DIR = Path.home() / "Downloads" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEASON_WEIGHTS = {2026: 2.0, 2025: 0.5, 2024: 0.2}

# region : Data Loading

dfs = []

for season_dir in BASE_DIR.glob("season_*"):

    feat_file = season_dir / "features_Q_pre_race.csv"
    lab_file  = season_dir / "labels_R.csv"

    if not feat_file.exists() or not lab_file.exists():
        continue

    feat = pd.read_csv(feat_file)
    lab  = pd.read_csv(lab_file)

    for c in ["DriverNumber", "Season", "Round"]:
        feat[c] = feat[c].astype(str)
        lab[c]  = lab[c].astype(str)

    df = feat.merge(lab,
                    on=["DriverNumber", "Season", "Round"],
                    how="inner")

    df["Season"] = df["Season"].astype(int)
    df["Round"]  = df["Round"].astype(int)

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# endregion

# region : Feature engineering

# 1. Data type transformation

data = data.sort_values(["Season", "Round"])
grp = data.groupby(["Season", "DriverNumber"])

data["races_so_far"] = grp.cumcount()
data["points_so_far"] = grp["Points"].transform(lambda x: x.shift().cumsum())
data["avg_position_so_far"] = grp["Position"].transform(lambda x: x.shift().expanding().mean())
data["win_rate_so_far"] = grp["Position"].transform(lambda x: (x.shift() == 1).expanding().mean())
data["podium_rate_so_far"] = grp["Position"].transform(lambda x: (x.shift() <= 3).expanding().mean())

ROLLING_COLS = ["points_so_far",
                "win_rate_so_far",
                "podium_rate_so_far",
                "avg_position_so_far"]

data[ROLLING_COLS] = data[ROLLING_COLS].fillna(0)

data["grid_missing"] = data["GridPosition"].isna().astype(int)

# 2. Historical means to replace missing values

grid_driver_event_mean = (data.dropna(subset=["GridPosition"])
                              .groupby(["DriverNumber", "EventName_x"])["GridPosition"]
                              .mean())

grid_driver_mean = (data.dropna(subset=["GridPosition"])
                        .groupby("DriverNumber")["GridPosition"]
                        .mean())

global_grid_median = data["GridPosition"].median()

def impute_grid(row):

    if pd.notna(row["GridPosition"]):
        return row["GridPosition"]

    key = (row["DriverNumber"], row["EventName_x"])

    if key in grid_driver_event_mean:
        return grid_driver_event_mean[key]

    if row["DriverNumber"] in grid_driver_mean:
        return grid_driver_mean[row["DriverNumber"]]

    return global_grid_median

data["GridPosition"] = data.apply(impute_grid, axis=1)

# 3. Target enconding

def target_encoding(df, col, target, smoothing=20):

    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["mean", "count"])

    smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / \
             (stats["count"] + smoothing)

    return smooth

event_te = target_encoding(data, "EventName_x", "Position")
team_te  = target_encoding(data, "TeamName", "Position")

data["Event_enc"] = data["EventName_x"].map(event_te)
data["Team_enc"]  = data["TeamName"].map(team_te)

data["Event_enc"] = data["Event_enc"].fillna(data["Position"].mean())
data["Team_enc"]  = data["Team_enc"].fillna(data["Position"].mean())

# 4. Applying season's weight (recent season > previous seasons)

data["sample_weight"] = data["Season"].map(SEASON_WEIGHTS)
data["sample_weight"] = data["sample_weight"].fillna(0.1)

# endregion

# region : Predictive model training

FEATURES = ["Event_enc",
            "Team_enc",

            "Q_best_lap_s",
            "Q_best_s1_s",
            "Q_best_s2_s",
            "Q_best_s3_s",
            "Q_quicklaps",
            "GridPosition",
            "grid_missing",
            "Q_top_speed_kmh",

            "races_so_far",
            "points_so_far",
            "win_rate_so_far",
            "podium_rate_so_far",
            "avg_position_so_far"]

X = data[FEATURES]
y = data["Position"]
w = data["sample_weight"]

model = XGBRegressor(n_estimators=800,
                     learning_rate=0.03,
                     max_depth=6,

                     subsample=0.85,
                     colsample_bytree=0.85,

                     reg_alpha=0.5,
                     reg_lambda=2.0,

                     objective="reg:squarederror",
                     random_state=42)

model.fit(X, y, sample_weight=w)

# endregion

# Saving models

joblib.dump(model, MODELS_DIR / "xgb_finish_position_model.pkl")
joblib.dump(event_te, MODELS_DIR / "event_target_encoding.pkl")
joblib.dump(team_te,  MODELS_DIR / "team_target_encoding.pkl")

print("Model trained")