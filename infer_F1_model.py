import joblib
import pandas as pd

from pathlib import Path

# Config

BASE_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\data\f1_ml")
MODELS_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\models")

PREDICT_SEASON = 2026

# region : Loading

# Models

model    = joblib.load(MODELS_DIR / "xgb_finish_position_model.pkl")
event_te = joblib.load(MODELS_DIR / "event_target_encoding.pkl")
team_te  = joblib.load(MODELS_DIR / "team_target_encoding.pkl")

print("Model and encodings loaded")

# Data

feat_dfs  = []
label_dfs = []

for season_dir in BASE_DIR.glob("season_*"):

    feat_file = season_dir / "features_Q_pre_race.csv"
    lab_file  = season_dir / "labels_R.csv"

    if feat_file.exists():
        feat_dfs.append(pd.read_csv(feat_file))

    if lab_file.exists():
        label_dfs.append(pd.read_csv(lab_file))

features = pd.concat(feat_dfs, ignore_index=True)
labels   = pd.concat(label_dfs, ignore_index=True)

if "GridPosition" not in features.columns:
    features["GridPosition"] = pd.NA

# endregion

# region : Race selection

LAST_COMPLETE_SEASON = PREDICT_SEASON - 1

season_curr_labels = labels[labels["Season"] == PREDICT_SEASON]
season_prev_feats  = features[features["Season"] == LAST_COMPLETE_SEASON]

completed_events = set(season_curr_labels["EventName"])

calendar_df = (season_prev_feats.sort_values("Round")[["Round", "EventName"]]
                                .drop_duplicates()
                                .reset_index(drop=True))

future_calendar = calendar_df[~calendar_df["EventName"].isin(completed_events)].reset_index(drop=True)

print("\nFuture races available for prediction:")
for i, row in future_calendar.iterrows():
    print(f"{i + 1}. {row['EventName']}")

choice = input("\nSelect race number to predict: ")

if not choice.isdigit():
    raise RuntimeError("Invalid input")

choice = int(choice)

if choice < 1 or choice > len(future_calendar):
    raise RuntimeError("Choice out of range")

PREDICT_EVENT = future_calendar.loc[choice - 1, "EventName"]

print(f"\nSelected race:")
print(f"{PREDICT_EVENT} ({PREDICT_SEASON})")

race_df = season_prev_feats[season_prev_feats["EventName"] == PREDICT_EVENT].copy()

# endregion

# region : Building season performance

completed_curr = labels[labels["Season"] == PREDICT_SEASON].copy()
grp = completed_curr.groupby("DriverNumber")

season_form = grp.agg(races_so_far=("Round", "count"),
                      points_so_far=("Points", "sum"),
                      avg_position_so_far=("Position", "mean"),
                      podium_rate_so_far=("Position", lambda x: (x <= 3).mean()),
                      win_rate_so_far=("Position", lambda x: (x == 1).mean())
                      ).reset_index()

race_df = race_df.merge(season_form, on="DriverNumber", how="left")

race_df[["races_so_far",
         "points_so_far",
         "avg_position_so_far",
         "podium_rate_so_far",
         "win_rate_so_far"]] = race_df[["races_so_far",
                                        "points_so_far",
                                        "avg_position_so_far",
                                        "podium_rate_so_far",
                                        "win_rate_so_far"]].fillna(0)

# endregion

# region : Feature engineering

# Grid Positions

if "GridPosition" not in race_df.columns:
    race_df["GridPosition"] = pd.NA

race_df["grid_missing"] = race_df["GridPosition"].isna().astype(int)

# Historical grid position statistics

grid_driver_event_mean = (features.dropna(subset=["GridPosition"])
                                  .groupby(["DriverNumber", "EventName"])["GridPosition"]
                                  .mean())

grid_driver_mean = (features.dropna(subset=["GridPosition"])
                            .groupby("DriverNumber")["GridPosition"]
                            .mean())

global_grid_median = features["GridPosition"].median()

def impute_grid(row):

    if pd.notna(row["GridPosition"]):
        return row["GridPosition"]

    key = (row["DriverNumber"], row["EventName"])

    if key in grid_driver_event_mean:
        return grid_driver_event_mean[key]

    if row["DriverNumber"] in grid_driver_mean:
        return grid_driver_mean[row["DriverNumber"]]

    return global_grid_median

race_df["GridPosition"] = race_df.apply(impute_grid, axis=1)

# Qualification data

QUALI_COLS = ["Q_best_lap_s",
              "Q_best_s1_s",
              "Q_best_s2_s",
              "Q_best_s3_s",
              "Q_quicklaps",
              "Q_top_speed_kmh"]

for col in QUALI_COLS:

    if col not in race_df.columns or race_df[col].isna().all():
        col_avg = features.groupby("DriverNumber")[col].mean()
        race_df[col] = race_df["DriverNumber"].map(col_avg)

race_df[QUALI_COLS] = race_df[QUALI_COLS].fillna(race_df[QUALI_COLS].mean())

# Target encoding

race_df["Event_enc"] = race_df["EventName"].map(event_te)
race_df["Team_enc"]  = race_df["TeamName"].map(team_te)

global_pos_mean = labels["Position"].mean()

race_df["Event_enc"] = race_df["Event_enc"].fillna(global_pos_mean)
race_df["Team_enc"]  = race_df["Team_enc"].fillna(global_pos_mean)

# endregion

# region : Prediction

print(f"\nPredicting podium for:")
print(f"{PREDICT_EVENT} ({PREDICT_SEASON})")

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

X = race_df[FEATURES]

race_df["expected_position"] = model.predict(X)
podium = race_df.sort_values("expected_position").head(3)

print("\nPredicted Podium:")

for i, row in enumerate(podium.itertuples(), start=1):
    print(f"P{i}: {row.FullName} "
          f"(Expected position: {row.expected_position:.2f})")

# endregion