import os
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import json

import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

# If you already have a consistent match_key builder, use it:
from pipeline.ml_update import make_match_key, add_asof_days_since_last_match  # MUST match how match_key was created in DB

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ---- point this at your GitHub dataset CSV(s) ----
CSV_PATHS = [
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2022.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2023.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2024.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2025.csv",
]

TRAIN_YEARS = {2022, 2023, 2024}
TEST_YEARS = {2025}

ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7
}

SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Carpet"}

ELO_INIT = 1500.0

# Keep surfaces consolidated; Unknown stays Unknown
ELO_SURFACES = ["Hard", "Clay", "Grass", "Carpet", "Unknown"]

LEVEL_IMPORTANCE = {
    "G": 1.30,   # Grand Slam
    "M": 1.20,   # Masters 1000
    "F": 1.15,   # Tour Finals (if present)
    "A": 1.10,   # ATP 500
    "D": 1.05,   # Davis Cup / team comps (if present)
    "O": 1.00,   # Olympics / other (optional)
    "250": 0.95,
    "C": 0.90,   # Challengers (if present)
}

ROUND_IMPORTANCE = {
    1: 0.90,  # R128/R64 etc
    2: 0.95,
    3: 1.00,
    4: 1.05,  # R16
    5: 1.10,  # QF
    6: 1.15,  # SF
    7: 1.20,  # F
}


def safe_div(num, den):
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full_like(num, np.nan, dtype="float64")
    np.divide(num, den, out=out, where=(den > 0))
    return out

def normalize_rank(rank, cap=300):
    if rank is None:
        return cap
    return min(rank, cap)

def load_db_matches():
    # Keep this minimal: just what DB currently has
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT
              match_key,
              tourney_date,
              surface,
              tourney_level,
              round,
              winner_name,
              loser_name,
              winner_rank,
              loser_rank,
              minutes
            FROM public.matches
            WHERE tourney_date IS NOT NULL
              AND winner_name IS NOT NULL
              AND loser_name IS NOT NULL
              AND COALESCE(score,'') NOT ILIKE '%w/o%';
        """)).fetchall()

    df = pd.DataFrame(rows, columns=[
        "match_key","tourney_date","surface","tourney_level","round",
        "winner_name","loser_name","winner_rank","loser_rank","minutes"
    ])
    df["year"] = (df["tourney_date"].astype(int) // 10000).astype(int)
    df["surface"] = df["surface"].map(lambda s: SURFACE_MAP.get(s, s))
    return df

def load_csv_matches():
    dfs = []
    for p in CSV_PATHS:
        d = pd.read_csv(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    # Normalize types
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64")
    df["match_num"] = pd.to_numeric(df["match_num"], errors="coerce").astype("Int64")
    df["year"] = (df["tourney_date"] // 10000).astype("Int64")

    # keep only columns we need for features
    keep = [
        "tourney_date","surface","tourney_level","round","best_of",
        "winner_name","loser_name",                 # ADD THESE
        "winner_seed","loser_seed","winner_entry","loser_entry",
        "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
        "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
    ]

    have = [c for c in keep if c in df.columns]
    df = df[have].copy()
    return df

def backfill_player_rolling_stats(conn, cutoff_date):
    conn.execute(text("TRUNCATE player_rolling_stats;"))

    conn.execute(text("""
    INSERT INTO player_rolling_stats
    SELECT
        player,
        COUNT(*) AS matches,
        AVG(aces / NULLIF(svpt,0)) AS ace_rate,
        AVG(df / NULLIF(svpt,0)) AS df_rate,
        AVG(first_in / NULLIF(svpt,0)) AS first_in,
        AVG(first_won / NULLIF(first_in,0)) AS first_won,
        AVG(second_won / NULLIF(svpt-first_in,0)) AS second_won,
        AVG(bp_saved / NULLIF(bp_faced,0)) AS bp_saved,
        MAX(tourney_date) AS last_updated
    FROM (
        SELECT
            winner_name AS player,
            w_ace AS aces, w_df AS df, w_svpt AS svpt,
            w_1stIn AS first_in, w_1stWon AS first_won,
            w_2ndWon AS second_won,
            w_bpSaved AS bp_saved, w_bpFaced AS bp_faced,
            tourney_date
        FROM matches
        WHERE tourney_date < :cut

        UNION ALL

        SELECT
            loser_name,
            l_ace, l_df, l_svpt,
            l_1stIn, l_1stWon,
            l_2ndWon,
            l_bpSaved, l_bpFaced,
            tourney_date
        FROM matches
        WHERE tourney_date < :cut
    ) x
    GROUP BY player;
    """), {"cut": cutoff_date})

def backfill_recent_form(conn, cutoff):
    conn.execute(text("TRUNCATE player_recent_form;"))

    conn.execute(text("""
    INSERT INTO player_recent_form
    SELECT
        player,
        AVG(win) FILTER (WHERE rn <= 5) AS last5,
        AVG(win) FILTER (WHERE rn <= 10) AS last10
    FROM (
        SELECT
            player,
            win,
            ROW_NUMBER() OVER (PARTITION BY player ORDER BY tourney_date DESC) rn
        FROM (
            SELECT winner_name AS player, 1 AS win, tourney_date
            FROM matches WHERE tourney_date < :cut
            UNION ALL
            SELECT loser_name, 0, tourney_date
            FROM matches WHERE tourney_date < :cut
        ) x
    ) y
    GROUP BY player;
    """), {"cut": cutoff})

def get_form(conn, player):
    r = conn.execute(text("""
        SELECT last5, last10
        FROM player_recent_form
        WHERE player_name=:p
    """), {"p": player}).fetchone()
    if not r or r[0] is None or r[1] is None:
        return (0.5, 0.5)
    return (float(r[0]), float(r[1]))


def make_features(df: pd.DataFrame):
    # --- ranks/minutes ---
    df["winner_rank"] = pd.to_numeric(df.get("winner_rank"), errors="coerce")
    df["loser_rank"]  = pd.to_numeric(df.get("loser_rank"), errors="coerce")
    df["minutes"] = pd.to_numeric(df.get("minutes"), errors="coerce")

    wr = df["winner_rank"].fillna(300).clip(upper=300)
    lr = df["loser_rank"].fillna(300).clip(upper=300)
    df["rank_diff"] = lr - wr

    # --- round_num / best_of ---
    df["round_num"] = df.get("round").map(ROUND_ORDER).fillna(0).astype(int) if "round" in df.columns else 0
    df["best_of"] = pd.to_numeric(df.get("best_of"), errors="coerce").fillna(3).astype(int) if "best_of" in df.columns else 3

    # --- importance scaling ---
    lvl = df.get("tourney_level")
    if lvl is not None:
        lvl = lvl.astype(str)
        df["tourney_importance"] = lvl.map(LEVEL_IMPORTANCE).fillna(1.0)
    else:
        df["tourney_importance"] = 1.0
    df["round_importance"] = df["round_num"].map(ROUND_IMPORTANCE).fillna(1.0)
    df["match_importance"] = df["tourney_importance"] * df["round_importance"]

    # ✅ CREATE X FIRST
    X = pd.DataFrame({
        "rank_diff": df["rank_diff"],
        "round_num": df["round_num"],
        "best_of": df["best_of"],
    })

    # --- Elo diffs ---
    for c in ["elo_diff_surface", "elo_diff_overall", "surface_match_count_diff"]:
        X[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    X["match_importance"] = pd.to_numeric(df.get("match_importance"), errors="coerce").fillna(1.0)

    # --- form + rolling ---
    for c in ["form5_diff","form10_diff",
              "roll_ace_rate_diff","roll_df_rate_diff","roll_first_in_diff",
              "roll_first_won_diff","roll_second_won_diff","roll_bp_saved_diff"]:
        X[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # ✅ NOW add days-since-last-match features (because X exists)
    def clip_days(x):
        x = pd.to_numeric(x, errors="coerce")
        return x.clip(lower=0, upper=3650)

    if "w_days_since_last" in df.columns:
        w_days = clip_days(df["w_days_since_last"])
        l_days = clip_days(df["l_days_since_last"])

        X["w_days_missing"] = pd.to_numeric(df.get("w_days_missing"), errors="coerce").fillna(1).astype(int)
        X["l_days_missing"] = pd.to_numeric(df.get("l_days_missing"), errors="coerce").fillna(1).astype(int)

        X["w_days_since_last_log"] = np.log1p(w_days.fillna(w_days.median()))
        X["l_days_since_last_log"] = np.log1p(l_days.fillna(l_days.median()))
        X["days_since_last_diff_log"] = X["w_days_since_last_log"] - X["l_days_since_last_log"]
    else:
        X["w_days_missing"] = 1
        X["l_days_missing"] = 1
        X["w_days_since_last_log"] = 0.0
        X["l_days_since_last_log"] = 0.0
        X["days_since_last_diff_log"] = 0.0

    # --- one-hots ---
    if "surface" in df.columns:
        X = pd.concat([X, pd.get_dummies(df["surface"], prefix="surface")], axis=1)
    if "tourney_level" in df.columns:
        X = pd.concat([X, pd.get_dummies(df["tourney_level"], prefix="level")], axis=1)

    # --- clean ---
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        if X[c].dtype.kind in "fc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0)

    # labels
    y = np.ones(len(df), dtype=int)

    # mirror
    X2 = X.copy()
    y2 = np.zeros(len(df), dtype=int)  # ✅ YOU NEED THIS

    flip_cols = [
        "rank_diff","form5_diff","form10_diff",
        "elo_diff_surface","elo_diff_overall","surface_match_count_diff",
        "days_since_last_diff_log"
    ]
    for c in flip_cols:
        if c in X2.columns:
            X2[c] = -X2[c]

    # swap winner/loser absolute columns for days-since
    for a, b in [("w_days_since_last_log","l_days_since_last_log"),
                 ("w_days_missing","l_days_missing")]:
        if a in X2.columns and b in X2.columns:
            tmp = X2[a].copy()
            X2[a] = X2[b]
            X2[b] = tmp

    X_final = pd.concat([X, X2], ignore_index=True)
    y_final = np.concatenate([y, y2])
    return X_final, y_final


def align_columns(X_train, X_other):
    return X_other.reindex(columns=X_train.columns, fill_value=0)

def get_rolling(conn, player):
    r = conn.execute(text("""
        SELECT ace_rate, df_rate, first_in, first_won,
               second_won, bp_saved
        FROM player_rolling_stats
        WHERE player_name=:p
    """), {"p": player}).fetchone()

    if not r:
        return [0]*6
    return list(r)

def add_asof_form_features(df):
    """
    df must have: tourney_date, winner_name, loser_name
    Returns df with form5_diff/form10_diff computed using ONLY prior matches.
    """
    df = df.sort_values("tourney_date").copy()

    # store prior results per player
    hist = {}  # player -> list of 1/0 results (chronological)

    f5 = []
    f10 = []

    def winrate_lastk(arr, k):
        if not arr:
            return 0.5
        tail = arr[-k:]
        return float(sum(tail) / len(tail))

    for _, r in df.iterrows():
        w = r["winner_name"]
        l = r["loser_name"]

        w_hist = hist.get(w, [])
        l_hist = hist.get(l, [])

        w5 = winrate_lastk(w_hist, 5)
        l5 = winrate_lastk(l_hist, 5)
        w10 = winrate_lastk(w_hist, 10)
        l10 = winrate_lastk(l_hist, 10)

        f5.append(w5 - l5)
        f10.append(w10 - l10)

        # update histories AFTER computing features (so no lookahead)
        hist.setdefault(w, []).append(1)
        hist.setdefault(l, []).append(0)

    df["form5_diff"] = f5
    df["form10_diff"] = f10
    return df

def add_asof_rolling_stats(df):
    """
    Builds rolling player serve stats from prior matches only.
    Adds roll_*_diff columns directly onto df.
    Requires serve stats columns to be present (from CSV merge).
    """
    df = df.sort_values("tourney_date").copy()

    # per player cumulative totals
    agg = {}  # player -> dict of sums

    def get(p):
        return agg.get(p, None)

    def rate(x, y):
        return x / y if y and y > 0 else np.nan

    out = {
        "roll_ace_rate_diff": [],
        "roll_df_rate_diff": [],
        "roll_first_in_diff": [],
        "roll_first_won_diff": [],
        "roll_second_won_diff": [],
        "roll_bp_saved_diff": [],
    }

    for _, r in df.iterrows():
        w = r["winner_name"]; l = r["loser_name"]

        aw = get(w); al = get(l)

        # compute winner/loser rates from prior totals
        def features(a):
            if not a:
                return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            ace_rate = rate(a["ace"], a["svpt"])
            df_rate = rate(a["df"], a["svpt"])
            first_in = rate(a["1stin"], a["svpt"])
            first_won = rate(a["1stwon"], a["1stin"])
            second_won = rate(a["2ndwon"], a["svpt"] - a["1stin"])
            bp_saved = rate(a["bpsaved"], a["bpfaced"])
            return (ace_rate, df_rate, first_in, first_won, second_won, bp_saved)

        fw = features(aw)
        fl = features(al)

        out["roll_ace_rate_diff"].append(fw[0] - fl[0])
        out["roll_df_rate_diff"].append(fw[1] - fl[1])
        out["roll_first_in_diff"].append(fw[2] - fl[2])
        out["roll_first_won_diff"].append(fw[3] - fl[3])
        out["roll_second_won_diff"].append(fw[4] - fl[4])
        out["roll_bp_saved_diff"].append(fw[5] - fl[5])

        # update totals AFTER using them
        def upd(p, prefix):
            a = agg.setdefault(p, {"svpt":0,"ace":0,"df":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0})
            a["svpt"]   += float(pd.to_numeric(r[f"{prefix}_svpt"], errors="coerce") or 0)
            a["ace"]    += float(pd.to_numeric(r[f"{prefix}_ace"], errors="coerce") or 0)
            a["df"]     += float(pd.to_numeric(r[f"{prefix}_df"], errors="coerce") or 0)
            a["1stin"]  += float(pd.to_numeric(r[f"{prefix}_1stIn"], errors="coerce") or 0)
            a["1stwon"] += float(pd.to_numeric(r[f"{prefix}_1stWon"], errors="coerce") or 0)
            a["2ndwon"] += float(pd.to_numeric(r[f"{prefix}_2ndWon"], errors="coerce") or 0)
            a["bpsaved"]+= float(pd.to_numeric(r[f"{prefix}_bpSaved"], errors="coerce") or 0)
            a["bpfaced"]+= float(pd.to_numeric(r[f"{prefix}_bpFaced"], errors="coerce") or 0)

        upd(w, "w")
        upd(l, "l")

    for k, v in out.items():
        df[k] = v

    return df


def load_rolling_maps(conn):
    rows = conn.execute(text("""
        SELECT player_name, ace_rate, df_rate, first_in, first_won, second_won, bp_saved
        FROM player_rolling_stats
    """)).fetchall()
    return {
        r[0]: (float(r[1] or 0), float(r[2] or 0), float(r[3] or 0),
               float(r[4] or 0), float(r[5] or 0), float(r[6] or 0))
        for r in rows
    }

def load_form_maps(conn):
    rows = conn.execute(text("""
        SELECT player_name, last5, last10
        FROM player_recent_form
    """)).fetchall()
    return {r[0]: (float(r[1] or 0.5), float(r[2] or 0.5)) for r in rows}

def build_rolling_maps_from_df(df_cut):
    # df_cut must contain winner/loser names + serve stat cols
    df = df_cut.copy()

    # numeric conversions
    for c in ["w_svpt","l_svpt","w_ace","l_ace","w_df","l_df","w_1stIn","l_1stIn",
              "w_1stWon","l_1stWon","w_2ndWon","l_2ndWon","w_bpSaved","l_bpSaved","w_bpFaced","l_bpFaced"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # winner rows
    w = pd.DataFrame({
        "player": df["winner_name"],
        "tourney_date": df["tourney_date"],
        "svpt": df["w_svpt"],
        "ace": df["w_ace"],
        "df": df["w_df"],
        "first_in": df["w_1stIn"],
        "first_won": df["w_1stWon"],
        "second_won": df["w_2ndWon"],
        "bp_saved": df["w_bpSaved"],
        "bp_faced": df["w_bpFaced"],
        "win": 1,
    })

    # loser rows
    l = pd.DataFrame({
        "player": df["loser_name"],
        "tourney_date": df["tourney_date"],
        "svpt": df["l_svpt"],
        "ace": df["l_ace"],
        "df": df["l_df"],
        "first_in": df["l_1stIn"],
        "first_won": df["l_1stWon"],
        "second_won": df["l_2ndWon"],
        "bp_saved": df["l_bpSaved"],
        "bp_faced": df["l_bpFaced"],
        "win": 0,
    })

    long = pd.concat([w, l], ignore_index=True)
    long = long.dropna(subset=["player"])
    long = long.sort_values("tourney_date")

    # aggregate rolling-style averages (simple global “pre-2025” avg per player)
    g = long.groupby("player")

    ace_rate   = (g["ace"].sum() / g["svpt"].sum()).replace([np.inf, -np.inf], np.nan)
    df_rate    = (g["df"].sum() / g["svpt"].sum()).replace([np.inf, -np.inf], np.nan)
    first_in   = (g["first_in"].sum() / g["svpt"].sum()).replace([np.inf, -np.inf], np.nan)
    first_won  = (g["first_won"].sum() / g["first_in"].sum()).replace([np.inf, -np.inf], np.nan)
    second_won = (g["second_won"].sum() / (g["svpt"].sum() - g["first_in"].sum())).replace([np.inf, -np.inf], np.nan)
    bp_saved   = (g["bp_saved"].sum() / g["bp_faced"].sum()).replace([np.inf, -np.inf], np.nan)

    rolling_map = {
        p: (
            float(ace_rate.get(p, 0) or 0),
            float(df_rate.get(p, 0) or 0),
            float(first_in.get(p, 0) or 0),
            float(first_won.get(p, 0) or 0),
            float(second_won.get(p, 0) or 0),
            float(bp_saved.get(p, 0) or 0),
        )
        for p in g.size().index
    }

    # form map: last 5 / last 10 winrate per player (pre-2025)
    def lastk_winrate(x, k):
        return float(np.mean(x[-k:])) if len(x) else 0.5

    form_map = {}
    for p, grp in long.groupby("player"):
        wins = grp.sort_values("tourney_date")["win"].to_numpy()
        form_map[p] = (lastk_winrate(wins, 5), lastk_winrate(wins, 10))

    return rolling_map, form_map

def add_surface_aware_elo(df: pd.DataFrame, k: float = 32.0) -> pd.DataFrame:
    """
    Adds pre-match surface-aware Elo and overall Elo diffs to df.

    Requires columns:
      - tourney_date (int-like, sortable)
      - surface (str)
      - winner_name (str)
      - loser_name (str)

    Output columns (pre-match values):
      - w_elo_surface, l_elo_surface
      - w_elo_overall, l_elo_overall
      - elo_diff_surface, elo_diff_overall
      - w_surface_matches, l_surface_matches (optional signal)
    """
    d = df.copy()

    # normalize surface labels
    d["surface"] = d["surface"].fillna("Unknown").astype(str)
    d.loc[~d["surface"].isin(ELO_SURFACES), "surface"] = "Unknown"

    # sort chronologically so we never look ahead
    # if you have match_num in df, include it here for same-day ordering
    sort_cols = ["tourney_date"]
    if "match_num" in d.columns:
        sort_cols.append("match_num")
    d = d.sort_values(sort_cols).reset_index(drop=True)

    # Elo state
    overall = {}  # player -> elo
    by_surface = {}  # (player, surface) -> elo
    surface_matches = {}  # (player, surface) -> count

    def get_overall(p: str) -> float:
        return overall.get(p, ELO_INIT)

    def get_surface(p: str, s: str) -> float:
        return by_surface.get((p, s), ELO_INIT)

    def exp_score(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    w_surf_pre, l_surf_pre = [], []
    w_all_pre,  l_all_pre  = [], []
    w_cnt, l_cnt = [], []

    for _, r in d.iterrows():
        w = str(r["winner_name"])
        l = str(r["loser_name"])
        s = str(r["surface"]) if pd.notna(r["surface"]) else "Unknown"
        if s not in ELO_SURFACES:
            s = "Unknown"

        # pre-match ratings (these are the features)
        w_over = get_overall(w)
        l_over = get_overall(l)
        w_surf = get_surface(w, s)
        l_surf = get_surface(l, s)

        w_surf_pre.append(w_surf)
        l_surf_pre.append(l_surf)
        w_all_pre.append(w_over)
        l_all_pre.append(l_over)

        w_cnt.append(surface_matches.get((w, s), 0))
        l_cnt.append(surface_matches.get((l, s), 0))

        # --- update ratings AFTER recording pre-match features (no leakage) ---
        # use a blend of overall + surface for expectation, so surface matters but overall still stabilizes
        # you can tune alpha; 0.5 is a good start
        alpha = 0.5
        w_comb = alpha * w_surf + (1 - alpha) * w_over
        l_comb = alpha * l_surf + (1 - alpha) * l_over

        e_w = exp_score(w_comb, l_comb)  # expected winner score
        e_l = 1.0 - e_w

        # actual scores: winner=1, loser=0
        s_w, s_l = 1.0, 0.0

        # rating updates
        delta_w = k * (s_w - e_w)
        delta_l = k * (s_l - e_l)

        overall[w] = w_over + delta_w
        overall[l] = l_over + delta_l

        by_surface[(w, s)] = w_surf + delta_w
        by_surface[(l, s)] = l_surf + delta_l

        surface_matches[(w, s)] = surface_matches.get((w, s), 0) + 1
        surface_matches[(l, s)] = surface_matches.get((l, s), 0) + 1

    d["w_elo_surface"] = w_surf_pre
    d["l_elo_surface"] = l_surf_pre
    d["w_elo_overall"] = w_all_pre
    d["l_elo_overall"] = l_all_pre

    d["elo_diff_surface"] = d["w_elo_surface"] - d["l_elo_surface"]
    d["elo_diff_overall"] = d["w_elo_overall"] - d["l_elo_overall"]

    d["w_surface_matches"] = w_cnt
    d["l_surface_matches"] = l_cnt
    d["surface_match_count_diff"] = d["w_surface_matches"] - d["l_surface_matches"]

    return d


def main():
    db = load_db_matches()

    db_key_cols  = ["tourney_date", "winner_name", "loser_name", "round"]
    csv_key_cols = ["tourney_date", "winner_name", "loser_name", "round"]

    csv = load_csv_matches()

    # keep the row that actually has serve stats when duplicates exist
    if "w_svpt" in csv.columns:
        csv["_has_stats"] = pd.to_numeric(csv["w_svpt"], errors="coerce").notna().astype(int)
        csv = csv.sort_values(["tourney_date", "_has_stats"], ascending=[True, False])
        csv = csv.drop_duplicates(csv_key_cols, keep="first").drop(columns=["_has_stats"])
    else:
        csv = csv.sort_values(["tourney_date"]).drop_duplicates(csv_key_cols, keep="first")

    merged = db.merge(
        csv,
        left_on=db_key_cols,
        right_on=csv_key_cols,
        how="left",
        suffixes=("", "_csv")
    )

    merged = add_surface_aware_elo(merged, k=32.0)

    has_stats = merged["w_svpt"].notna().mean() if "w_svpt" in merged.columns else 0.0
    print(f"Merged rows: {len(merged)}  with serve-stats: {has_stats:.1%}")

    train_df = merged[merged["year"].isin(TRAIN_YEARS)].copy()
    test_df  = merged[merged["year"].isin(TEST_YEARS)].copy()

    print("Train years:", sorted(train_df["year"].unique()), "n=", len(train_df))
    print("Test years:",  sorted(test_df["year"].unique()),  "n=", len(test_df))

    train_df = train_df.sort_values(["tourney_date"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["tourney_date"]).reset_index(drop=True)

    train_df = add_asof_form_features(train_df)
    train_df = add_asof_rolling_stats(train_df)
    train_df = add_asof_days_since_last_match(train_df)

    test_df  = add_asof_form_features(test_df)
    test_df  = add_asof_rolling_stats(test_df)
    test_df  = add_asof_days_since_last_match(test_df)

    # time split: train=2022-23, val=2024, test=2025
    train_mask = train_df["year"].isin([2022, 2023]).to_numpy()
    val_mask   = train_df["year"].isin([2024]).to_numpy()

    X_tr, y_tr     = make_features(train_df[train_mask].copy())
    X_val, y_val   = make_features(train_df[val_mask].copy())
    X_test, y_test = make_features(test_df.copy())

    X_val  = align_columns(X_tr, X_val)
    X_test = align_columns(X_tr, X_test)

    leak_cols = [
        "first_won_diff",
        "second_won_diff",
        "first_in_diff",
        "ace_rate_diff",
        "df_rate_diff",
        "bp_saved_diff",
        "sv_games_diff",
    ]

    found = [c for c in leak_cols if c in X_tr.columns]
    if found:
        raise RuntimeError(f"DATA LEAKAGE FEATURES FOUND: {found}")

    print(list(X_tr.columns))

    # --- DMatrix ---
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dval   = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_tr.columns))
    dtest  = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_tr.columns))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.02,                # learning_rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 30,
        "lambda": 5.0,              # reg_lambda
        "alpha": 1.0,               # reg_alpha
        "tree_method": "hist",
    }

    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=200
    )


    # --- Predict ---
    p_val  = bst.predict(dval)
    p_test = bst.predict(dtest)

    print("VAL logloss:", log_loss(y_val, p_val))
    print("VAL acc:", accuracy_score(y_val, p_val > 0.5))

    print("TEST(2025) logloss:", log_loss(y_test, p_test))
    print("TEST(2025) acc:", accuracy_score(y_test, p_test > 0.5))

    print("best_iteration:", bst.best_iteration)
    print("best_score:", bst.best_score)

    # --- Feature importance (gain) ---
    imp = bst.get_score(importance_type="gain")
    imp = pd.Series(imp).sort_values(ascending=False)
    print("\nTop 20 features:\n", imp.head(20).to_string())

    os.makedirs("artifacts", exist_ok=True)
    bst.save_model("artifacts/xgb_model.json")

    with open("artifacts/feature_columns.json", "w") as f:
        json.dump(list(X_tr.columns), f)

    print("✅ Saved artifacts/xgb_model.json and artifacts/feature_columns.json")


if __name__ == "__main__":
    main()
