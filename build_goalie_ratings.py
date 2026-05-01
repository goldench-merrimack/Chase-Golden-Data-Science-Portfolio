# scripts/build_goalie_ratings_eh.py

import pandas as pd

INPUT_CSV = "data/eh_goalie_gar.csv"
OUTPUT_CSV = "data/goalies.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.copy()
    df["Season"] = df["Season"].astype(str)

    # Keep most recent season per goalie
    df = df.sort_values(["Player", "Season"])
    latest = df.groupby("Player", as_index=False).tail(1)

    # GAR per game as core signal
    latest["GAR_per_GP"] = latest["GAR"] / latest["GP"].replace(0, pd.NA)
    latest["GAR_per_GP"] = latest["GAR_per_GP"].fillna(0.0)

    # Goalie rating = GAR per game (we can rescale later if needed)
    latest["goalie_rating"] = latest["GAR_per_GP"]

    out = latest[["Player", "Team", "goalie_rating"]].rename(
        columns={"Player": "goalie", "Team": "team"}
    )

    out.to_csv(OUTPUT_CSV, index=False)
    print("✓ Saved goalie ratings to", OUTPUT_CSV)
    print(out.head())


if __name__ == "__main__":
    main()