# scripts/build_injury_scores.py

import pandas as pd

GAMES_CSV = "data/games_today.csv"
PLAYER_VALUES_CSV = "data/player_values.csv"
INJURIES_CSV = "data/injuries_today.csv"


def main():
    games = pd.read_csv(GAMES_CSV)
    values = pd.read_csv(PLAYER_VALUES_CSV)
    injuries = pd.read_csv(INJURIES_CSV)

    # Only OUT players matter
    injuries = injuries[injuries["status"].str.upper() == "OUT"].copy()

    injuries = injuries.merge(
        values[["player", "team", "impact_rating"]],
        on=["player", "team"],
        how="left",
    )

    injuries["impact_rating"] = injuries["impact_rating"].fillna(0.0)

    team_injury = (
        injuries.groupby(["date", "game_id", "team"])["impact_rating"]
        .sum()
        .reset_index()
        .rename(columns={"impact_rating": "total_impact"})
    )

    games["away_injury_score"] = 0.0
    games["home_injury_score"] = 0.0

    # Away team injuries
    games = games.merge(
        team_injury,
        left_on=["date", "game_id", "away_team"],
        right_on=["date", "game_id", "team"],
        how="left",
    )
    games["away_injury_score"] = -games["total_impact"].fillna(0.0)
    games = games.drop(columns=["team", "total_impact"])

    # Home team injuries
    games = games.merge(
        team_injury,
        left_on=["date", "game_id", "home_team"],
        right_on=["date", "game_id", "team"],
        how="left",
    )
    games["home_injury_score"] = -games["total_impact"].fillna(0.0)
    games = games.drop(columns=["team", "total_impact"])

    games.to_csv(GAMES_CSV, index=False)
    print("✓ Updated injury scores in", GAMES_CSV)
    print(
        games[
            [
                "date",
                "game_id",
                "away_team",
                "home_team",
                "away_injury_score",
                "home_injury_score",
            ]
        ]
    )


if __name__ == "__main__":
    main()
