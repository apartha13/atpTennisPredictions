import os
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from tennis_model import TennisPredictor, TourneyCtx

CSV_PATHS = [
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2022.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2023.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2024.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/2025.csv",
    "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/ongoing_tourneys.csv",
]

SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Hard"}  # treat carpet as hard

LEVEL_K = {
    "G": 38, "M": 34, "A": 30,
    "500": 26, "250": 22,
    "D": 18, "F": 40, "O": 18
}
ROUND_MULT = {"R128":1.0,"R64":1.05,"R32":1.1,"R16":1.15,"QF":1.2,"SF":1.25,"F":1.3}

# --- Environment ---
DATABASE_URL = os.environ["DATABASE_URL"]  # Supabase/Render Postgres URL
LEAGUE_YEAR = int(os.environ.get("LEAGUE_YEAR", "2026"))
COMMISSIONER_KEY = os.environ.get("COMMISSIONER_KEY", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
xgb_predictor = TennisPredictor(engine)

def expected(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def elo_update(ra, rb, sa, k):
    ea = expected(ra, rb)
    return ra + k * (sa - ea)

def safe_div(num, den):
    num = float(num) if num is not None else 0.0
    den = float(den) if den is not None else 0.0
    return (num / den) if den > 0 else np.nan

def load_csv():
    dfs = [pd.read_csv(p) for p in CSV_PATHS]
    df = pd.concat(dfs, ignore_index=True)

    keep = [
        "tourney_date","surface","tourney_level","round",
        "winner_name","loser_name",
        "w_svpt","w_ace","w_df","w_1stIn","w_1stWon","w_2ndWon","w_bpSaved","w_bpFaced",
        "l_svpt","l_ace","l_df","l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["tourney_date","winner_name","loser_name"])
    df["surface"] = df["surface"].map(lambda s: SURFACE_MAP.get(s, s))
    df["tourney_level"] = df["tourney_level"].astype(str)
    df["round"] = df["round"].astype(str)
    df = df.sort_values("tourney_date")
    return df

def build_player_state(df: pd.DataFrame, asof_date: int):
    df = df[df["tourney_date"] < asof_date].copy()
    df = df.sort_values("tourney_date")

    # Elo state
    elo_overall = {}
    elo_surface = {"Hard": {}, "Clay": {}, "Grass": {}}
    matches_overall = {}
    matches_surface = {"Hard": {}, "Clay": {}, "Grass": {}}

    # form history
    results_hist = {}  # player -> list of 1/0

    # rolling serve totals
    agg = {}  # player -> sums

    def get_elo(map_, p):
        return float(map_.get(p, 1500.0))

    def inc_match(p, surf):
        matches_overall[p] = matches_overall.get(p, 0) + 1
        if surf in matches_surface:
            matches_surface[surf][p] = matches_surface[surf].get(p, 0) + 1
    
    def nz(x):
        return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)


    def update_roll(p, svpt, ace, df_, first_in, first_won, second_won, bp_saved, bp_faced):
        a = agg.setdefault(p, {"svpt":0,"ace":0,"df":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0})
        a = agg.setdefault(p, {"svpt":0.0,"ace":0.0,"df":0.0,"1stin":0.0,"1stwon":0.0,"2ndwon":0.0,"bpsaved":0.0,"bpfaced":0.0})
        a["svpt"]    += nz(svpt)
        a["ace"]     += nz(ace)
        a["df"]      += nz(df_)
        a["1stin"]   += nz(first_in)
        a["1stwon"]  += nz(first_won)
        a["2ndwon"]  += nz(second_won)
        a["bpsaved"] += nz(bp_saved)
        a["bpfaced"] += nz(bp_faced)

    for _, r in df.iterrows():
        w = str(r["winner_name"]); l = str(r["loser_name"])
        surf = str(r.get("surface", "Hard"))
        lvl  = str(r.get("tourney_level", "250"))
        rnd  = str(r.get("round", "R32"))

        # K scaled by tournament importance
        base_k = LEVEL_K.get(lvl, 22)
        mult = ROUND_MULT.get(rnd, 1.0)
        k = base_k * mult

        # overall elo
        ew = get_elo(elo_overall, w)
        el = get_elo(elo_overall, l)
        elo_overall[w] = elo_update(ew, el, 1.0, k)
        elo_overall[l] = elo_update(el, ew, 0.0, k)

        # surface elo
        if surf in elo_surface:
            esw = get_elo(elo_surface[surf], w)
            esl = get_elo(elo_surface[surf], l)
            elo_surface[surf][w] = elo_update(esw, esl, 1.0, k)
            elo_surface[surf][l] = elo_update(esl, esw, 0.0, k)

        inc_match(w, surf)
        inc_match(l, surf)

        # form history
        results_hist.setdefault(w, []).append(1)
        results_hist.setdefault(l, []).append(0)

        # rolling totals from THIS match (safe because it’s used only for future matches)
        update_roll(
            w, r.get("w_svpt"), r.get("w_ace"), r.get("w_df"),
            r.get("w_1stIn"), r.get("w_1stWon"), r.get("w_2ndWon"),
            r.get("w_bpSaved"), r.get("w_bpFaced")
        )
        update_roll(
            l, r.get("l_svpt"), r.get("l_ace"), r.get("l_df"),
            r.get("l_1stIn"), r.get("l_1stWon"), r.get("l_2ndWon"),
            r.get("l_bpSaved"), r.get("l_bpFaced")
        )

    # finalize player_state frame
    players = sorted(set(list(elo_overall.keys()) + list(results_hist.keys()) + list(agg.keys())))
    out = []
    for p in players:
        wins = results_hist.get(p, [])
        f5 = float(np.mean(wins[-5:])) if wins else 0.5
        f10 = float(np.mean(wins[-10:])) if wins else 0.5

        a = agg.get(p, {"svpt":0,"ace":0,"df":0,"1stin":0,"1stwon":0,"2ndwon":0,"bpsaved":0,"bpfaced":0})
        ace_rate   = safe_div(a["ace"], a["svpt"])
        df_rate    = safe_div(a["df"], a["svpt"])
        first_in   = safe_div(a["1stin"], a["svpt"])
        first_won  = safe_div(a["1stwon"], a["1stin"])
        second_won = safe_div(a["2ndwon"], a["svpt"] - a["1stin"])
        bp_saved   = safe_div(a["bpsaved"], a["bpfaced"])

        out.append({
            "player_name": p,
            "asof_date": int(asof_date),

            "elo_overall": float(elo_overall.get(p, 1500.0)),
            "elo_hard": float(elo_surface["Hard"].get(p, 1500.0)),
            "elo_clay": float(elo_surface["Clay"].get(p, 1500.0)),
            "elo_grass": float(elo_surface["Grass"].get(p, 1500.0)),

            "matches_overall": int(matches_overall.get(p, 0)),
            "matches_hard": int(matches_surface["Hard"].get(p, 0)),
            "matches_clay": int(matches_surface["Clay"].get(p, 0)),
            "matches_grass": int(matches_surface["Grass"].get(p, 0)),

            "form5": f5,
            "form10": f10,

            "roll_ace_rate": ace_rate,
            "roll_df_rate": df_rate,
            "roll_first_in": first_in,
            "roll_first_won": first_won,
            "roll_second_won": second_won,
            "roll_bp_saved": bp_saved,
        })

    ps = pd.DataFrame(out)

    RATE_COLS = [
        "roll_ace_rate", "roll_df_rate", "roll_first_in",
        "roll_first_won", "roll_second_won", "roll_bp_saved"
    ]

    # compute means using available values (skip NaNs)
    means = {c: float(ps[c].mean(skipna=True)) for c in RATE_COLS}

    # if a column is entirely NaN (unlikely), use sane tennis defaults
    fallbacks = {
        "roll_ace_rate": 0.06,     # ~6%
        "roll_df_rate": 0.03,      # ~3%
        "roll_first_in": 0.62,     # ~62%
        "roll_first_won": 0.72,    # ~72% of first serves won
        "roll_second_won": 0.52,   # ~52% of second serves won
        "roll_bp_saved": 0.60,     # ~60%
    }

    for c in RATE_COLS:
        if np.isnan(means[c]):
            means[c] = fallbacks[c]
        ps[c] = ps[c].fillna(means[c])

    return ps

def upsert_player_state(engine, ps: pd.DataFrame):
    cols = list(ps.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in cols if c != "player_name"])

    sql = f"""
    insert into public.player_state ({", ".join(cols)})
    values ({placeholders})
    on conflict (player_name) do update set
      {updates};
    """
    with engine.begin() as conn:
        records = ps.replace({np.nan: None}).to_dict(orient="records")
        conn.execute(text(sql), records)

def main():
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)

    asof_date = int(os.environ.get("ASOF_DATE", "20260111")) 
    df = load_csv()
    print("min date:", int(df["tourney_date"].min()))
    print("max date:", int(df["tourney_date"].max()))
    print("2026+ rows:", int((df["tourney_date"] >= 20260101).sum()))
    df_sinner = df[(df["winner_name"] == "Jannik Sinner") | (df["loser_name"] == "Jannik Sinner")]
    print("Sinner matches:", len(df_sinner))

    cols = ["w_svpt","w_ace","w_1stIn","w_1stWon","l_svpt","l_ace","l_1stIn","l_1stWon"]
    print(df_sinner[cols].isna().mean().sort_values(ascending=False))
    ps = build_player_state(df, asof_date)
    print("player_state rows:", len(ps))

    upsert_player_state(engine, ps)
    print("✅ uploaded player_state to Supabase")

    xgb_predictor.clear_cache()

if __name__ == "__main__":
    main()
