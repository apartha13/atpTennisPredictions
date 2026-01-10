# predict_features.py
import numpy as np
import pandas as pd
from sqlalchemy import text

ROUND_ORDER = {"R128":1,"R64":2,"R32":3,"R16":4,"QF":5,"SF":6,"F":7}

def log1p_clip_days(x):
    if x is None:
        return 0.0
    x = max(0, min(int(x), 3650))
    return float(np.log1p(x))

def get_rank(conn, name):
    r = conn.execute(text("""
      SELECT winner_rank FROM matches
      WHERE winner_name=:p AND winner_rank IS NOT NULL
      ORDER BY tourney_date DESC LIMIT 1
    """), {"p": name}).fetchone()
    return int(r[0]) if r and r[0] else 300

def get_form(conn, name):
    r = conn.execute(text("""
      SELECT last5, last10 FROM player_recent_form WHERE player_name=:p
    """), {"p": name}).fetchone()
    if not r:
        return (0.5, 0.5)
    return (float(r[0] or 0.5), float(r[1] or 0.5))

def get_rolling(conn, name):
    r = conn.execute(text("""
      SELECT ace_rate, df_rate, first_in, first_won, second_won, bp_saved
      FROM player_rolling_stats WHERE player_name=:p
    """), {"p": name}).fetchone()
    if not r:
        return (0,0,0,0,0,0)
    return tuple(float(x or 0) for x in r)

def get_elo(conn, name, surface):
    # If you store elo in DB, query it. If not, use your existing elo code.
    r = conn.execute(text("""
      SELECT elo_overall, elo_surface, surface_matches
      FROM player_elo WHERE player_name=:p AND surface=:s
    """), {"p": name, "s": surface}).fetchone()
    if not r:
        return (1500.0, 1500.0, 0)
    return (float(r[0]), float(r[1]), int(r[2] or 0))

def get_days_since_last(conn, name, asof_date):
    r = conn.execute(text("""
      SELECT MAX(tourney_date) FROM matches
      WHERE (winner_name=:p OR loser_name=:p) AND tourney_date < :d
    """), {"p": name, "d": asof_date}).fetchone()
    if not r or r[0] is None:
        return (None, 1)  # missing
    last = int(r[0])
    # dates are YYYYMMDD; easiest is to store as real date in DB, but for now approximate:
    # BEST: convert to datetime; but quick approach below assumes you can parse.
    from datetime import datetime
    a = datetime.strptime(str(asof_date), "%Y%m%d")
    b = datetime.strptime(str(last), "%Y%m%d")
    return ((a - b).days, 0)

def build_match_features(conn, p1, p2, surface, level, round_code, best_of, tourney_date):
    # rank_diff = (p2_rank - p1_rank) because training used loser-winner
    r1 = get_rank(conn, p1)
    r2 = get_rank(conn, p2)

    (f1_5, f1_10) = get_form(conn, p1)
    (f2_5, f2_10) = get_form(conn, p2)

    ro1 = get_rolling(conn, p1)
    ro2 = get_rolling(conn, p2)

    # days since last
    d1, m1 = get_days_since_last(conn, p1, tourney_date)
    d2, m2 = get_days_since_last(conn, p2, tourney_date)

    # logs + diff
    d1log = log1p_clip_days(d1) if d1 is not None else 0.0
    d2log = log1p_clip_days(d2) if d2 is not None else 0.0
    d_diff = d1log - d2log

    # Elo (optional if you store)
    # If you don't have player_elo yet, keep elo_diff_overall=0 etc until you add it.
    try:
        e1_over, e1_surf, e1_cnt = get_elo(conn, p1, surface)
        e2_over, e2_surf, e2_cnt = get_elo(conn, p2, surface)
        elo_diff_overall = e1_over - e2_over
        elo_diff_surface = e1_surf - e2_surf
        surface_match_count_diff = e1_cnt - e2_cnt
    except Exception:
        elo_diff_overall = 0.0
        elo_diff_surface = 0.0
        surface_match_count_diff = 0.0

    round_num = ROUND_ORDER.get(round_code, 0)

    # match_importance should match your training mapping
    LEVEL_IMPORTANCE = {"G":1.30,"M":1.20,"F":1.15,"A":1.10,"D":1.05,"O":1.00,"250":0.95,"C":0.90}
    ROUND_IMPORTANCE = {1:0.90,2:0.95,3:1.00,4:1.05,5:1.10,6:1.15,7:1.20}
    match_importance = LEVEL_IMPORTANCE.get(level, 1.0) * ROUND_IMPORTANCE.get(round_num, 1.0)

    X = {
        "rank_diff": (r2 - r1),
        "round_num": round_num,
        "best_of": int(best_of or 3),
        "elo_diff_surface": elo_diff_surface,
        "elo_diff_overall": elo_diff_overall,
        "surface_match_count_diff": surface_match_count_diff,
        "match_importance": match_importance,
        "form5_diff": (f1_5 - f2_5),
        "form10_diff": (f1_10 - f2_10),

        "roll_ace_rate_diff": (ro1[0] - ro2[0]),
        "roll_df_rate_diff": (ro1[1] - ro2[1]),
        "roll_first_in_diff": (ro1[2] - ro2[2]),
        "roll_first_won_diff": (ro1[3] - ro2[3]),
        "roll_second_won_diff": (ro1[4] - ro2[4]),
        "roll_bp_saved_diff": (ro1[5] - ro2[5]),

        "w_days_missing": m1,
        "l_days_missing": m2,
        "w_days_since_last_log": d1log,
        "l_days_since_last_log": d2log,
        "days_since_last_diff_log": d_diff,
    }

    # one-hots (must match training naming)
    X[f"surface_{surface}"] = 1
    X[f"level_{level}"] = 1

    return pd.DataFrame([X])
