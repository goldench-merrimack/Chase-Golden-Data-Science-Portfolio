import pandas as pd

# Input CSVs (from Evolving-Hockey downloads)
TEAM_5V5_CSV = "data/eh_team_5v5.csv"
TEAM_RAPM_CSV = "data/eh_team_rapm.csv"
TEAM_PP_CSV = "data/eh_team_pp.csv"
TEAM_SH_CSV = "data/eh_team_sh.csv"

# Output
OUTPUT_CSV = "data/teams.csv"


def load_latest_per_team(df, team_col="Team", season_col="Season"):
    """
    Keep only the latest season row for each team.
    Evolving-Hockey includes a Season column; we just take the most recent.
    """
    df = df.copy()
    df[season_col] = df[season_col].astype(str)
    df = df.sort_values([team_col, season_col])
    return df.groupby(team_col, as_index=False).tail(1)


# ---------------------------------------------------------------------
# 5v5 CORE
# ---------------------------------------------------------------------
def build_core_5v5():
    tm = pd.read_csv(TEAM_5V5_CSV)

    df = load_latest_per_team(tm, "Team", "Season")

    # Main 5v5 possession metric
    # Column names from EH 5v5 standard team table:
    # "xGF%" should exist; if not, you can adjust here.
    df["xgf_pct"] = df["xGF%"]
    df["xgf_pct_centered"] = (df["xgf_pct"] - 50.0) / 5.0  # 5% ≈ 1 unit

    # Goal differential per 60
    df["goal_diff_per60"] = df["GF/60"] - df["GA/60"]

    # xG differential per 60
    if "xG±/60" in df.columns:
        df["xg_diff_per60"] = df["xG±/60"]
    else:
        # Fallback: find a column with xG and ±/60 if naming differs
        alt_cols = [c for c in df.columns if "xG" in c and "/60" in c and "±" in c]
        if alt_cols:
            df["xg_diff_per60"] = df[alt_cols[0]]
        else:
            df["xg_diff_per60"] = 0.0

    # Blend into a raw base rating
    df["base_rating_raw"] = (
        0.5 * df["xg_diff_per60"]
        + 0.3 * df["goal_diff_per60"]
        + 0.2 * df["xgf_pct_centered"]
    )

    # Normalize to z-score so everything is on a similar scale
    mean = df["base_rating_raw"].mean()
    std = df["base_rating_raw"].std()
    df["base_rating"] = (df["base_rating_raw"] - mean) / std

    # 5v5 strength slider (separate from base_rating)
    df["fivev5_strength"] = df["xgf_pct_centered"]

    return df[["Team", "base_rating", "fivev5_strength"]]


# ---------------------------------------------------------------------
# RAPM
# ---------------------------------------------------------------------
def add_rapm(df):
    """Merge RAPM team EV and build rapm_strength."""
    try:
        rapm = pd.read_csv(TEAM_RAPM_CSV)
    except FileNotFoundError:
        df["rapm_strength"] = 0.0
        return df

    rapm_latest = load_latest_per_team(rapm, "Team", "Season")

    # These column names come from EH RAPM team EV table; adjust if needed.
    # G±/60 and xG±/60 are usually present.
    g_col = "G±/60"
    xg_col = "xG±/60"

    if g_col not in rapm_latest.columns:
        cand = [c for c in rapm_latest.columns if "G±/60" in c]
        if cand:
            g_col = cand[0]
    if xg_col not in rapm_latest.columns:
        cand = [c for c in rapm_latest.columns if "xG±/60" in c]
        if cand:
            xg_col = cand[0]

    rapm_latest["rapm_raw"] = (
        0.6 * rapm_latest[xg_col] + 0.4 * rapm_latest[g_col]
    )

    mean = rapm_latest["rapm_raw"].mean()
    std = rapm_latest["rapm_raw"].std()
    rapm_latest["rapm_strength"] = (rapm_latest["rapm_raw"] - mean) / std

    return df.merge(rapm_latest[["Team", "rapm_strength"]], on="Team", how="left")


# ---------------------------------------------------------------------
# SPECIAL TEAMS (PP + SH)
# ---------------------------------------------------------------------
def add_special_teams(df):
    """
    Merge PP (eh_team_pp.csv) and SH (eh_team_sh.csv) and
    build a single special_teams_rating.
    """
    try:
        pp = pd.read_csv(TEAM_PP_CSV)
        sh = pd.read_csv(TEAM_SH_CSV)
    except FileNotFoundError:
        df["special_teams_rating"] = 0.0
        return df

    pp_latest = load_latest_per_team(pp, "Team", "Season")
    sh_latest = load_latest_per_team(sh, "Team", "Season")

    # --- Adjust these column names based on your actual CSV headers ---

    # From PP table: offensive metric like "xGF/60" or "GF/60"
    pp_xgf_col = "xGF/60"
    if pp_xgf_col not in pp_latest.columns:
        cand = [c for c in pp_latest.columns if "xGF/60" in c]
        pp_xgf_col = cand[0] if cand else None

    # From SH table: defensive metric like "xGA/60" or "GA/60"
    sh_xga_col = "xGA/60"
    if sh_xga_col not in sh_latest.columns:
        cand = [c for c in sh_latest.columns if "xGA/60" in c or "GA/60" in c]
        sh_xga_col = cand[0] if cand else None

    # If we can't find the columns, fall back to neutral special teams
    if pp_xgf_col is None or sh_xga_col is None:
        df["special_teams_rating"] = 0.0
        return df

    pp_latest = pp_latest[["Team", pp_xgf_col]].rename(
        columns={pp_xgf_col: "pp_xgf60"}
    )
    sh_latest = sh_latest[["Team", sh_xga_col]].rename(
        columns={sh_xga_col: "sh_xga60"}
    )

    st = pp_latest.merge(sh_latest, on="Team", how="inner")

    # Higher PP chance generation is good; lower SH chance against is good
    st["special_raw"] = st["pp_xgf60"] - st["sh_xga60"]

    mean = st["special_raw"].mean()
    std = st["special_raw"].std() if st["special_raw"].std() != 0 else 1.0
    st["special_teams_rating"] = (st["special_raw"] - mean) / std

    return df.merge(
        st[["Team", "special_teams_rating"]], on="Team", how="left"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    core = build_core_5v5()
    core = add_rapm(core)
    core = add_special_teams(core)

    # Home/road still equal to base_rating for now
    core["home_strength"] = core["base_rating"]
    core["away_strength"] = core["base_rating"]

    out = core[
        [
            "Team",
            "base_rating",
            "fivev5_strength",
            "rapm_strength",
            "special_teams_rating",
            "home_strength",
            "away_strength",
        ]
    ].rename(columns={"Team": "team"})

    out.to_csv(OUTPUT_CSV, index=False)
    print("✓ Saved team ratings to", OUTPUT_CSV)
    print(out.head())


if __name__ == "__main__":
    main()
