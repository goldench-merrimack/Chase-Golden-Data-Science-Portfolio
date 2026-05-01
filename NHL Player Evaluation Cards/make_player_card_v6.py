import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
OUT_DIR = "output_cards"
os.makedirs(OUT_DIR, exist_ok=True)
GAR_FILE  = os.path.join(DATA_DIR, "EH_gar_sk_stats_regular_2026-03-05.csv")
XGAR_FILE = os.path.join(DATA_DIR, "EH_xgar_sk_stats_regular_2026-03-05.csv")
RAPM_FILE = os.path.join(DATA_DIR, "EH_rapm_sk_stats_ev_regular_2026-03-05.csv")
STD_FILE  = os.path.join(DATA_DIR, "EH_std_sk_stats_all_regular_no_adj_2026-03-05 (1).csv")
MIN_TOI = 200
WEIGHTS = {"25-26": 0.50, "24-25": 0.30, "23-24": 0.20}
SEASON_ORDER = ["23-24", "24-25", "25-26"]
TEAM_PRIMARY = {
    "ANA":"#F47A38","ARI":"#8C2633","BOS":"#FFB81C","BUF":"#003087","CGY":"#C8102E","CAR":"#CC0000",
    "CHI":"#CF0A2C","CBJ":"#002654","COL":"#6F263D","DAL":"#006847","DET":"#CE1126","EDM":"#041E42",
    "FLA":"#C8102E","LAK":"#111111","MIN":"#154734","MTL":"#AF1E2D","NSH":"#FFB81C","NJD":"#CE1126",
    "NYI":"#00539B","NYR":"#0038A8","OTT":"#C52032","PHI":"#F74902","PIT":"#FFB81C","SJS":"#006D75",
    "SEA":"#001628","STL":"#002F87","TBL":"#002868","TOR":"#00205B","VAN":"#00205B","VGK":"#B4975A",
    "WSH":"#C8102E","WPG":"#041E42"
}

TIER_BANDS = [
    (0, 20,  "#d73027", "Poor"),
    (20, 40, "#b77e7e", "Below Avg"),
    (40, 60, "#D5D5A8", "Average"),
    (60, 80, "#68B386", "Good"),
    (80, 100,"#12AB4F", "Elite"),
]

CARD_METRICS = [
    ("WAR_3yr",            "WAR (3yr)"),
    ("xWAR_3yr",           "xWAR (3yr)"),
    ("GAR_3yr",            "GAR (3yr)"),
    ("xGAR_3yr",           "xGAR (3yr)"),
    ("RAPM_xGpm60_3yr",    "RAPM xG±/60"),
    ("RAPM_Cpm60_3yr",     "RAPM C±/60"),
    ("ixG60_3yr",          "ixG/60"),
    ("iCF60_3yr",          "iCF/60"),
    ("Fin60_3yr",          "(G-ixG)/60"),
    ("SPAR_3yr",           "SPAR (3yr)"),
]


def norm_team(team: str) -> str:
    if team is None:
        return ""
    t = str(team)
    return (t.replace("T.B", "TBL")
             .replace("N.J", "NJD")
             .replace("S.J", "SJS")
             .replace("L.A", "LAK"))


def pos_group(pos: str) -> str:
    return {"L":"F","R":"F","C":"F","LW":"F","RW":"F","D":"D"}.get(str(pos), str(pos))


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(s)).strip()
    s = re.sub(r"\s+", "_", s)
    return s


def prep_common(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Player","Season","Team","Position"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "TOI" in df.columns:
        df["TOI"] = pd.to_numeric(df["TOI"], errors="coerce")
    if "GP" in df.columns:
        df["GP"] = pd.to_numeric(df["GP"], errors="coerce")
    return df


def weighted_player_table(df: pd.DataFrame, value_cols: list[str], keep_meta: bool=True) -> pd.DataFrame:
    out = []
    season_rank = {s:i for i,s in enumerate(SEASON_ORDER)}
    for player, g in df.groupby("Player"):
        r = {"Player": player}
        g2 = g.copy()
        g2["season_rank"] = g2["Season"].map(season_rank).fillna(-1)
        g2 = g2.sort_values("season_rank")
        last = g2.iloc[-1]
        if keep_meta:
            for meta in ["Team","Position","GP","TOI"]:
                if meta in g2.columns:
                    r[meta] = last.get(meta, np.nan)
        for c in value_cols:
            val = 0.0
            for s,w in WEIGHTS.items():
                v = g2.loc[g2["Season"] == s, c]
                if len(v):
                    val += float(v.iloc[0]) * w
            r[c+"_3yr"] = val
        out.append(r)
    return pd.DataFrame(out)


def build_master() -> pd.DataFrame:
    gar  = prep_common(pd.read_csv(GAR_FILE)).rename(columns={"TOI_All":"TOI"})
    xgar = prep_common(pd.read_csv(XGAR_FILE)).rename(columns={"TOI_All":"TOI"})
    rapm = prep_common(pd.read_csv(RAPM_FILE))
    std  = prep_common(pd.read_csv(STD_FILE))
    gar  = gar[gar["TOI"].fillna(0)  >= MIN_TOI].copy()
    xgar = xgar[xgar["TOI"].fillna(0) >= MIN_TOI].copy()
    rapm = rapm[rapm["TOI"].fillna(0) >= MIN_TOI].copy()
    std  = std[std["TOI"].fillna(0)  >= MIN_TOI].copy()
    for c in ["ixG","G","iCF"]:
        if c in std.columns:
            std[c] = pd.to_numeric(std[c], errors="coerce")
    std["ixG60"] = std["ixG"] / std["TOI"] * 60
    std["iCF60"] = std["iCF"] / std["TOI"] * 60
    std["Fin60"] = (std["G"] - std["ixG"]) / std["TOI"] * 60
    rapm = rapm.rename(columns={
        "xG±/60":"RAPM_xGpm60",
        "C±/60":"RAPM_Cpm60",
        "xG+/-/60":"RAPM_xGpm60",
        "C+/-/60":"RAPM_Cpm60",
    })
    rapm["RAPM_xGpm60"] = pd.to_numeric(rapm.get("RAPM_xGpm60"), errors="coerce")
    rapm["RAPM_Cpm60"]  = pd.to_numeric(rapm.get("RAPM_Cpm60"),  errors="coerce")
    for c in ["WAR","GAR","Off_GAR","Def_GAR","Pens_GAR","SPAR"]:
        if c in gar.columns:
            gar[c] = pd.to_numeric(gar[c], errors="coerce")
    for c in ["xWAR","xGAR","xOff_GAR","xDef_GAR","xSPAR"]:
        if c in xgar.columns:
            xgar[c] = pd.to_numeric(xgar[c], errors="coerce")
    gar_w  = weighted_player_table(gar,  ["WAR","GAR","Off_GAR","Def_GAR","Pens_GAR","SPAR"], keep_meta=True)
    xgar_w = weighted_player_table(xgar, ["xWAR","xGAR","xOff_GAR","xDef_GAR","xSPAR"], keep_meta=False)
    rapm_w = weighted_player_table(rapm, ["RAPM_xGpm60","RAPM_Cpm60"], keep_meta=False)
    std_w  = weighted_player_table(std,  ["ixG60","iCF60","Fin60"], keep_meta=False)
    master = (gar_w.merge(xgar_w, on="Player", how="left")
                   .merge(rapm_w, on="Player", how="left")
                   .merge(std_w, on="Player", how="left"))
    master["Team"] = master["Team"].apply(norm_team)
    master["PosGroup"] = master["Position"].apply(pos_group)
    for col,_ in CARD_METRICS:
        master[col] = pd.to_numeric(master[col], errors="coerce")
        master[col+"_pct"] = master.groupby("PosGroup")[col].rank(pct=True, method="average") * 100
    return master


def tiered_segmented_bars(ax, labels, values):
    for lo, hi, color, _ in TIER_BANDS:
        ax.axvspan(lo, hi, color=color, alpha=0.18, zorder=0)
    y = np.arange(len(labels))
    for i, val in enumerate(values):
        ax.barh([i], [val], height=0.66, color="#111111", alpha=0.92, zorder=3)
        for x in range(0, 101, 5):
            ax.vlines(x, i-0.33, i+0.33, colors="white", linewidth=0.35, alpha=0.35, zorder=4)
        ax.text(102, i, f"{val:0.0f}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.10, zorder=1)
    ax.set_xlabel("Percentile vs position peers", fontsize=10)


def radar_nhl_style(ax, labels, values, color):
    vals = np.asarray([
        50 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        for v in values
    ], dtype=float)
    vals = np.clip(vals, 0, 100)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    vals = np.concatenate([vals, [vals[0]]])
    ax.set_ylim(0, 100)
    for r in [20, 40, 60, 80, 100]:
        ax.plot(np.linspace(0, 2*np.pi, 200), np.full(200, r),
                color="#dddddd", linewidth=1, alpha=0.9, zorder=0)
    for a in angles[:-1]:
        ax.plot([a, a], [0, 100], color="#eeeeee", linewidth=1, zorder=0)
    ax.plot(angles, vals, color=color, linewidth=3, zorder=3)
    ax.fill(angles, vals, color=color, alpha=0.10, zorder=2)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontsize=12)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
    ax.set_rlabel_position(18)
    ax.grid(False)
    ax.spines["polar"].set_color("#111111")
    ax.spines["polar"].set_linewidth(1.5)


def add_tier_legend(fig, x=0.64, y=0.92):
    fig.text(x, y+0.02, "Percentile tiers", fontsize=11, fontweight="bold")
    for idx, (_, _, color, label) in enumerate(TIER_BANDS):
        yy = y - idx*0.035
        fig.patches.append(plt.Rectangle(
            (x, yy-0.012), 0.015, 0.018,
            transform=fig.transFigure,
            facecolor=color, edgecolor="none", alpha=0.9
        ))
        fig.text(x+0.02, yy-0.012, label, fontsize=10, va="bottom")


def make_player_card(master: pd.DataFrame, player_name: str) -> str:
    hit = master[master["Player"].str.contains(player_name, case=False, na=False)]
    if hit.empty:
        raise ValueError(f"Player not found: {player_name}")
    row = hit.iloc[0]
    team = norm_team(row.get("Team",""))
    accent = TEAM_PRIMARY.get(team, "#333333")
    labels, vals = [], []
    for col, label in CARD_METRICS:
        v = row.get(col+"_pct", np.nan)
        if pd.notna(v):
            labels.append(label)
            vals.append(float(v))
    def getpct(base):
        v = row.get(base+"_pct", np.nan)
        return float(v) if pd.notna(v) else np.nan
    finishing = getpct("Fin60_3yr")
    shotq     = getpct("ixG60_3yr")
    play      = np.nanmean([getpct("RAPM_xGpm60_3yr"), getpct("RAPM_Cpm60_3yr")])
    defense   = getpct("Def_GAR_3yr")
    overall   = np.nanmean([getpct("WAR_3yr"), getpct("xWAR_3yr"), getpct("GAR_3yr"), getpct("xGAR_3yr")])
    radar_labels = ["Finishing","Shot Quality","Playdriving","Defense","Overall"]
    radar_vals   = [finishing, shotq, play, defense, overall]
    fig = plt.figure(figsize=(13, 7), facecolor="white")
    ax_strip = fig.add_axes([0, 0.965, 1, 0.02])
    ax_strip.axis("off")
    ax_strip.add_patch(plt.Rectangle((0,0), 1, 1, transform=ax_strip.transAxes, color=accent))
    fig.text(0.03, 0.93, f"{row['Player']}", fontsize=22, fontweight="bold")
    fig.text(0.03, 0.895, f"{team} | {row.get('Position','')} | 3-year weighted (50/30/20)", fontsize=12)
    gp = row.get("GP", np.nan)
    toi = row.get("TOI", np.nan)
    fig.text(0.03, 0.865, f"GP: {gp:.0f}   TOI: {toi:.0f} min   Peer: {row.get('PosGroup','')}", fontsize=11)
    add_tier_legend(fig, x=0.64, y=0.92)
    ax1 = fig.add_axes([0.05, 0.14, 0.55, 0.70])
    tiered_segmented_bars(ax1, labels, vals)
    ax2 = fig.add_axes([0.67, 0.16, 0.30, 0.68], polar=True)
    radar_nhl_style(ax2, radar_labels, radar_vals, color=accent)
    raw_parts = []
    for col, lab in [("WAR_3yr","WAR"),("xWAR_3yr","xWAR"),("GAR_3yr","GAR"),("xGAR_3yr","xGAR"),("RAPM_xGpm60_3yr","RAPM xG±/60")]:
        v = row.get(col, np.nan)
        if pd.notna(v):
            raw_parts.append(f"{lab}: {v:.2f}")
    fig.text(0.03, 0.06, "  |  ".join(raw_parts), fontsize=11)
    out_path = os.path.join(OUT_DIR, f"{safe_filename(row['Player'])}_v6.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_tableau_dataset(master):
    df = master.copy()
    needed_bases = [
        "RAPM_xGpm60_3yr",
        "RAPM_Cpm60_3yr",
        "ixG60_3yr",
        "Fin60_3yr",
        "Def_GAR_3yr",
        "WAR_3yr",
        "xWAR_3yr",
        "GAR_3yr",
        "xGAR_3yr",
    ]
    # Ensure percentile columns exist for any bases we will reference
    for base in needed_bases:
        pct_col = base + "_pct"
        if base in df.columns and pct_col not in df.columns:
            df[base] = pd.to_numeric(df[base], errors="coerce")
            df[pct_col] = df.groupby("PosGroup")[base].rank(pct=True, method="average") * 100

    df["PlayDriving_badge"] = df[["RAPM_xGpm60_3yr_pct","RAPM_Cpm60_3yr_pct"]].mean(axis=1)
    df["ShotQuality_badge"] = df.get("ixG60_3yr_pct")
    df["Finishing_badge"] = df.get("Fin60_3yr_pct")
    df["Defense_badge"] = df.get("Def_GAR_3yr_pct")
    df["Overall_badge"] = df[["WAR_3yr_pct","xWAR_3yr_pct","GAR_3yr_pct","xGAR_3yr_pct"]].mean(axis=1)

    cols = [
    "Player",
    "Team",
    "Position",
    "PosGroup",
    "GP",
    "TOI",

    "PlayDriving_badge",
    "ShotQuality_badge",
    "Finishing_badge",
    "Defense_badge",
    "Overall_badge",

    "WAR_3yr_pct",
    "xWAR_3yr_pct",
    "GAR_3yr_pct",
    "xGAR_3yr_pct",
    "RAPM_xGpm60_3yr_pct",
    "RAPM_Cpm60_3yr_pct",
    "ixG60_3yr_pct",
    "iCF60_3yr_pct",
    "Fin60_3yr_pct",
    "SPAR_3yr_pct",

    "WAR_3yr",
    "xWAR_3yr",
    "GAR_3yr",
    "xGAR_3yr",
    "RAPM_xGpm60_3yr"
]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv("tableau_player_cards.csv", index=False)
    print("Exported Tableau dataset")


if __name__ == "__main__":
    master = build_master()
    export_tableau_dataset(master)
    player_to_make = "Quinn Hughes"
    out = make_player_card(master, player_to_make)
    print("Saved:", out)
