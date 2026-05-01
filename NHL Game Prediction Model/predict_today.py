import pandas as pd
from math import exp

TEAMS_CSV = "data/teams.csv"
GOALIES_CSV = "data/goalies.csv"
GAMES_CSV = "data/games_today.csv"
OUTPUT_CSV = "data/today_picks.csv"

# ---------------- CONFIG / WEIGHTS ---------------- #
HOME_ICE_BONUS = 0.35

W_BASE = 0.25        # overall team strength
W_5V5 = 0.25         # 5v5 xG-based strength
W_RAPM = 0.20        # RAPM talent
W_SPECIAL = 0.10     # PP/PK special teams
W_HOME_ROAD = 0.00   # still neutral; we'll upgrade later
W_GOALIE = 0.35
W_REST = 0.15
W_INJURY = 0.40


def logistic(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def implied_prob_from_ml(ml: float) -> float:
    """Convert American moneyline to implied probability."""
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)


def confidence_from_edge(edge: float) -> str:
    """Map edge size to A/B/C/PASS tiers."""
    if edge >= 0.08:
        return "A"
    elif edge >= 0.04:
        return "B"
    elif edge >= 0.02:
        return "C"
    else:
        return "PASS"


def main():
    # ------------ LOAD DATA ------------ #
    teams = pd.read_csv(TEAMS_CSV)
    goalies = pd.read_csv(GOALIES_CSV)
    games = pd.read_csv(GAMES_CSV)

    # ------------ MERGE TEAM RATINGS ------------ #
    # Home team
    games = games.merge(
        teams,
        left_on="home_team",
        right_on="team",
        how="left",
    ).rename(
        columns={
            "base_rating": "home_base_rating",
            "fivev5_strength": "home_5v5",
            "rapm_strength": "home_rapm",
            "special_teams_rating": "home_special",
            "home_strength": "home_home_strength",
            "away_strength": "home_away_strength",
        }
    ).drop(columns=["team"])

    # Away team
    games = games.merge(
        teams,
        left_on="away_team",
        right_on="team",
        how="left",
    ).rename(
        columns={
            "base_rating": "away_base_rating",
            "fivev5_strength": "away_5v5",
            "rapm_strength": "away_rapm",
            "special_teams_rating": "away_special",
            "home_strength": "away_home_strength",
            "away_strength": "away_away_strength",
        }
    ).drop(columns=["team"])

    # ------------ MERGE GOALIES ------------ #
    games = games.merge(
        goalies[["goalie", "goalie_rating"]],
        left_on="home_goalie",
        right_on="goalie",
        how="left",
    ).rename(columns={"goalie_rating": "home_goalie_rating"}).drop(columns=["goalie"])

    games = games.merge(
        goalies[["goalie", "goalie_rating"]],
        left_on="away_goalie",
        right_on="goalie",
        how="left",
    ).rename(columns={"goalie_rating": "away_goalie_rating"}).drop(columns=["goalie"])

    games[["home_goalie_rating", "away_goalie_rating"]] = games[
        ["home_goalie_rating", "away_goalie_rating"]
    ].fillna(0.0)

    # ------------ FEATURE DIFFS ------------ #
    games["base_diff"] = games["home_base_rating"] - games["away_base_rating"]
    games["fivev5_diff"] = games["home_5v5"] - games["away_5v5"]
    games["rapm_diff"] = games["home_rapm"] - games["away_rapm"]
    games["special_diff"] = games["home_special"] - games["away_special"]

    games["home_road_diff"] = (
        games["home_home_strength"] - games["away_away_strength"]
    )

    games["goalie_diff"] = (
        games["home_goalie_rating"] - games["away_goalie_rating"]
    )

    # Rest & injuries should already be columns in games_today.csv
    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["injury_diff"] = (
        games["home_injury_score"] - games["away_injury_score"]
    )

    # ------------ SCORE DIFF (MODEL BRAIN) ------------ #
    games["score_diff"] = (
        W_BASE * games["base_diff"]
        + W_5V5 * games["fivev5_diff"]
        + W_RAPM * games["rapm_diff"]
        + W_SPECIAL * games["special_diff"]
        + W_HOME_ROAD * games["home_road_diff"]
        + W_GOALIE * games["goalie_diff"]
        + W_REST * games["rest_diff"]
        + W_INJURY * games["injury_diff"]
        + HOME_ICE_BONUS
    )

    # ------------ MODEL WIN PROBS ------------ #
    games["home_win_prob"] = games["score_diff"].apply(logistic)
    games["away_win_prob"] = 1.0 - games["home_win_prob"]

    # ------------ IMPLIED PROBS FROM ODDS ------------ #
    games["home_implied"] = games["home_ml"].apply(implied_prob_from_ml)
    games["away_implied"] = games["away_ml"].apply(implied_prob_from_ml)

    # ------------ EDGES & PICKS ------------ #
    games["home_edge"] = games["home_win_prob"] - games["home_implied"]
    games["away_edge"] = games["away_win_prob"] - games["away_implied"]

    games["model_pick"] = games.apply(
        lambda row: "HOME"
        if row["home_edge"] > row["away_edge"]
        else "AWAY",
        axis=1,
    )
    games["max_edge"] = games[["home_edge", "away_edge"]].max(axis=1)
    games["confidence"] = games["max_edge"].apply(confidence_from_edge)

    # ------------ OUTPUT ------------ #
    output_cols = [
        "date",
        "game_id",
        "away_team",
        "home_team",
        "away_goalie",
        "home_goalie",
        "away_ml",
        "home_ml",
        "away_win_prob",
        "home_win_prob",
        "away_implied",
        "home_implied",
        "away_edge",
        "home_edge",
        "max_edge",
        "model_pick",
        "confidence",
    ]
    out = games[output_cols].copy()

    prob_cols = ["away_win_prob", "home_win_prob", "away_implied", "home_implied"]
    edge_cols = ["away_edge", "home_edge", "max_edge"]
    out[prob_cols + edge_cols] = out[prob_cols + edge_cols].round(3)

    display_cols = [
        "date",
        "away_team",
        "home_team",
        "away_goalie",
        "home_goalie",
        "away_ml",
        "home_ml",
        "model_pick",
        "confidence",
        "max_edge",
    ]

    print("\n=== Today's NHL Model Picks (Evolving-Hockey powered) ===\n")
    print(out[display_cols].to_string(index=False))

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved full results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
