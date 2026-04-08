import time
import pandas as pd
import fastf1 as f1

from pathlib import Path
from datetime import date

def main():

    # region : Variables

    downloads_dir = Path.home() / "Downloads"
    base_dir = downloads_dir / "f1_ml"
    cache_dir = base_dir / "fastf1_cache"

    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Enable cache to reuse what we already have
    f1.Cache.enable_cache(cache_dir=str(cache_dir))

    current_year = date.today().year
    seasons = [current_year - 2, current_year - 1, current_year]

    # endregion

    all_labels = []
    all_features = []
    all_schedules = []

    # Seasons loop

    for season in seasons:

        print(f"\nSeason {season}")
        season_dir = base_dir / f"season_{season}"
        season_dir.mkdir(exist_ok=True)

        schedule_file = season_dir / "schedule.csv"

        # region : Schedules

        try:
            schedule = f1.get_event_schedule(season, include_testing=False).reset_index(drop=False)

            if "EventDate" in schedule.columns:
                today = pd.Timestamp(date.today())
                schedule = schedule[schedule["EventDate"] <= today].copy()

            if "RoundNumber" not in schedule.columns:

                if "index" in schedule.columns:
                    schedule = schedule.rename(columns={"index": "RoundNumber"})

                elif "Round" in schedule.columns:
                    schedule = schedule.rename(columns={"Round": "RoundNumber"})

            schedule["Season"] = season
            schedule.to_csv(schedule_file, index=False, encoding="utf-8")

            print(f"Saved schedule: {schedule_file}")
            all_schedules.append(schedule)

        except Exception as e:
            print(f"[WARN] Could not fetch schedule for {season}: {e}")

            if schedule_file.exists():
                print(f"→ Reusing local schedule for {season}")
                schedule = pd.read_csv(schedule_file)

            else:
                print(f"→ No schedule available, skipping season {season}")
                continue

        # endregion

        season_features = []
        season_labels = []

        round_col = "RoundNumber" if "RoundNumber" in schedule.columns else schedule.columns[0]

        # region : Events

        for _, event in schedule.iterrows():

            rnd = int(event[round_col])
            event_name = event.get("EventName", f"Round {rnd}")

            print(f"\n  -> {season} Round {rnd}: {event_name}")

            # region : Qualifications

            q_features = None

            try:
                q_session = f1.get_session(season, rnd, "Q")
                q_session.load(laps=True, weather=False, messages=False, telemetry=True)

            except Exception as e:
                print(f"    - Q: no data ({e})")
                q_session = None

            if q_session is not None:

                q_res = q_session.results
                q_laps = q_session.laps

                if q_res is None or q_res.empty or q_laps is None or q_laps.empty:
                    print("    - Q: results/laps empty")

                else:
                    base_cols = ["DriverNumber", "Abbreviation", "BroadcastName", "FullName", "TeamName"]

                    for c in base_cols:
                        if c not in q_res.columns:
                            q_res[c] = pd.NA

                    ql = q_laps.pick_quicklaps().copy()

                    def to_sec(s):
                        return pd.to_timedelta(s, errors="coerce").dt.total_seconds()

                    ql["LapTimeS"] = to_sec(ql["LapTime"])
                    ql["S1S"] = to_sec(ql["Sector1Time"])
                    ql["S2S"] = to_sec(ql["Sector2Time"])
                    ql["S3S"] = to_sec(ql["Sector3Time"])

                    q_agg = (ql.groupby("DriverNumber")
                               .agg(Q_best_lap_s=("LapTimeS", "min"),
                                    Q_best_s1_s=("S1S", "min"),
                                    Q_best_s2_s=("S2S", "min"),
                                    Q_best_s3_s=("S3S", "min"),
                                    Q_quicklaps=("LapNumber", "count")).reset_index())

                    # Top speeds
                    tops = []

                    for abbr in ql["Driver"].dropna().unique():

                        try:
                            car = q_session.laps.pick_drivers(abbr).get_car_data()

                            if car is not None and not car.empty:
                                tops.append({"Abbreviation": abbr,
                                             "Q_top_speed_kmh": car["Speed"].max()})
                                
                        except Exception:
                            pass

                    tops_df = pd.DataFrame(tops)

                    dim = q_res[["DriverNumber", "Abbreviation",
                                 "BroadcastName", "FullName", "TeamName"]].drop_duplicates()

                    q_features = (q_agg.merge(dim, on="DriverNumber", how="left")
                                       .merge(tops_df, on="Abbreviation", how="left")
                                       .assign(Season=season, Round=rnd, EventName=event_name))

            # endregion

            # region : Races

            labels = None

            try:
                r_session = f1.get_session(season, rnd, "R")
                r_session.load(laps=False, telemetry=False)
                r_res = r_session.results

                if r_res is not None and not r_res.empty:

                    r = r_res.copy()
                    r["DriverNumber"] = r["DriverNumber"].astype("string")

                    if "ClassifiedPosition" not in r.columns:
                        r["ClassifiedPosition"] = r["Position"]

                    labels = r[["DriverNumber", "GridPosition", "Position",
                                "ClassifiedPosition", "Status", "Points"]]

                    labels["Round"] = rnd
                    labels["Season"] = season
                    labels["EventName"] = event_name
                    labels["Winner"] = (labels["ClassifiedPosition"] == 1).astype("int8")

            except Exception as e:
                print(f"    - R: no data ({e})")

            # endregion

            if q_features is not None and not q_features.empty:
                season_features.append(q_features)
                print(f"    - Q: features rows={len(q_features)}")

            else:
                print("    - Q: no features")

            if labels is not None and not labels.empty:
                season_labels.append(labels)
                print(f"    - R: labels rows={len(labels)}")

            else:
                print("    - R: no labels")

            time.sleep(0.75)

        # endregion

        # region : Saving data

        if season_features:
            feat_df = pd.concat(season_features, ignore_index=True)
            feat_df.to_csv(season_dir / "features_Q_pre_race.csv", index=False)

            all_features.append(feat_df)
            print(f"\nSaved pre-race features: {season_dir/'features_Q_pre_race.csv'}")

        else:
            print("\n[WARN] No pre-race features for this season.")

        if season_labels:
            lab_df = pd.concat(season_labels, ignore_index=True)
            lab_df.to_csv(season_dir / "labels_R.csv", index=False)

            all_labels.append(lab_df)
            print(f"Saved race labels: {season_dir/'labels_R.csv'}")

        else:
            print("[WARN] No race labels for this season.")

        # endregion

    # region : Saving combined data

    if all_features:
        pd.concat(all_features).to_csv(base_dir / "features_Q_pre_race_all.csv", index=False)

    if all_labels:
        pd.concat(all_labels).to_csv(base_dir / "labels_R_all.csv", index=False)

    print("\nF1 data collected !")

    # endregion

if __name__ == "__main__":
    main()