import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(r"C:\Users\ThomasVienot\OneDrive - Bengs\Documents\Bengs\Missions\Formations Interne IA\ML_F1\data\f1_ml")

season_data = {}

# region : Data loading

for season_dir in sorted(BASE_DIR.glob("season_*")):

    try:
        season = int(season_dir.name.split("_")[1])

    except:
        continue

    print("\n" + "=" * 50)
    print(f"Exploring Season {season}")
    print("=" * 50)

    # feat -> feature
    # lab  -> label

    feat_file = season_dir / "features_Q_pre_race.csv"
    lab_file  = season_dir / "labels_R.csv"

    if not feat_file.exists() or not lab_file.exists():
        print("Missing feature or label file → skipping")
        continue

    features = pd.read_csv(feat_file)
    labels   = pd.read_csv(lab_file)

    for c in ["DriverNumber", "Season", "Round"]:
        features[c] = features[c].astype(str)
        labels[c]   = labels[c].astype(str)

    df = features.merge(labels,
                        on=["DriverNumber", "Season", "Round"],
                        how="inner")

    season_data[season] = df

    print(f"Merged rows: {len(df)}")

# endregion

# region : Quality checks

    print("\n--- Columns & dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isna().sum())

    # Sanity rules

    if "Q_best_lap_s" in df.columns:
        bad = df[df["Q_best_lap_s"] <= 0]
        print(f"Invalid Q lap times: {len(bad)}")

    if "Q_top_speed_kmh" in df.columns:
        insane = df[df["Q_top_speed_kmh"] > 380]
        print(f"Top speeds > 380 km/h: {len(insane)}")

    if "GridPosition" in df.columns:
        bad_grid = df[df["GridPosition"] > 25]
        print(f"Impossible grid positions: {len(bad_grid)}")

    # Numeric distributions

    num_cols = df.select_dtypes(include=["int", "float"]).columns
    cols_to_plot = num_cols[:12]

    if len(cols_to_plot) > 0:

        ncols = 3
        nrows = int(np.ceil(len(cols_to_plot) / ncols))

        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(18, 5 * nrows))

        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, cols_to_plot):
            df[col].hist(ax=ax, bins=25)
            ax.set_title(col)

        for ax in axes[len(cols_to_plot):]:
            ax.axis("off")

        fig.suptitle(f"Season {season} – Numeric Distributions", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Correlations

    if len(num_cols) > 1:

        corr = df[num_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr,
                    cmap="coolwarm",
                    center=0,
                    ax=ax)

        ax.set_title(f"Season {season} – Feature Correlations", fontsize=16)
        fig.tight_layout()
        plt.show()

    # Driver / team consistency

    id_col = "Abbreviation" if "Abbreviation" in df.columns else "DriverNumber"

    print("\nDriver appearance count:")
    print(df[id_col].value_counts().head(10))

    if "TeamName" in df.columns:
        print("\nTeam appearance count:")
        print(df["TeamName"].value_counts())

# endregion

# region : Cross-season checks

print("\n" + "=" * 50)
print("Cross‑Season Schema Comparison")
print("=" * 50)

if len(season_data) >= 2:

    col_sets = [set(df.columns) for df in season_data.values()]
    common_cols = set.intersection(*col_sets)

    print("\nCommon columns across all seasons:")
    print(sorted(common_cols))

    for s, df in season_data.items():

        missing = common_cols - set(df.columns)

        if missing:
            print(f"Season {s} missing columns: {missing}")

else:
    print("Not enough seasons to compare.")

# endregion