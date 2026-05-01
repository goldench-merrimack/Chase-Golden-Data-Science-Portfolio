# scripts/build_player_values_eh.py

import pandas as pd

INPUT_CSV = "data/eh_skater_gar.csv"
OUTPUT_CSV = "data/player_values.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.copy()
    df["Season"] = df["Season"].astype(str)

    # Latest season per skater
    df = df.sort_values(["Player", "Season"])
    latest = df.groupby("Player", as_index=False).tail(1)

    latest["GAR_per_GP"] = latest["GAR"] / latest["GP"].replace(0, pd.NA)
    latest["GAR_per_GP"] = latest["GAR_per_GP"].fillna(0.0)

    # Impact rating: scaled GAR per game
    latest["impact_rating"] = latest["GAR_per_GP"] * 5.0

    out = latest[["Player", "Team", "Position", "impact_rating"]].rename(
        columns={
            "Player": "player",
            "Team": "team",
            "Position": "position",
        }
    )

    out.to_csv(OUTPUT_CSV, index=False)
    print("✓ Saved player values to", OUTPUT_CSV)
    print(out.sort_values("impact_rating", ascending=False).head(10))


if __name__ == "__main__":
    main()
