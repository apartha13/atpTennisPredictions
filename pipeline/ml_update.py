import csv
import io
import math
import requests
from datetime import datetime, date
from sqlalchemy import text
import pandas as pd
import numpy as np


SURFACE_MAP = {
    "Hard": "Hard",
    "Clay": "Clay",
    "Grass": "Grass",
    "Carpet": "Carpet",
}

def year_url(year: int) -> str:
    return f"https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"

def fetch_year_rows(year: int) -> list[dict]:
    url = year_url(year)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    f = io.StringIO(r.text)
    reader = csv.DictReader(f)
    return list(reader)

def parse_int_or_none(x):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return int(float(x))  # handles "8", "8.0" safely
    except Exception:
        return None

def _s(x):
    if x is None:
        return ""
    return str(x).strip()

def make_match_key(row: dict) -> str:
    tid = _s(row.get("tourney_id"))
    tdate = _s(row.get("tourney_date"))
    mnum = _s(row.get("match_num"))

    if mnum:
        return f"{tid}|{tdate}|{mnum}"

    w = _s(row.get("winner_name"))
    l = _s(row.get("loser_name"))
    rnd = _s(row.get("round"))
    score = _s(row.get("score"))
    minutes = _s(row.get("minutes"))
    return f"{tid}|{tdate}|{rnd}|{w}|{l}|{score}|{minutes}"

def upsert_matches(conn, rows: list[dict]) -> int:
    """
    Inserts match rows. Skips matches that already exist.
    Returns number actually inserted.
    """
    inserted = 0
    for row in rows:
        surface = SURFACE_MAP.get(row.get("surface", ""), row.get("surface", "") or "Unknown")
        mk = make_match_key(row)

        params = {
            "match_key": mk,
            "tourney_id": row.get("tourney_id"),
            "tourney_name": row.get("tourney_name"),
            "tourney_level": row.get("tourney_level"),
            "surface": surface,
            "tourney_date": int(row["tourney_date"]) if row.get("tourney_date") else None,
            "match_num": int(row["match_num"]) if row.get("match_num") else None,
            "round": row.get("round"),
            "winner_name": row.get("winner_name"),
            "loser_name": row.get("loser_name"),
            "winner_rank": parse_int_or_none(row.get("winner_rank")),
            "loser_rank":  parse_int_or_none(row.get("loser_rank")),
            "score": row.get("score"),
            "minutes": int(row["minutes"]) if row.get("minutes") else None,
        }

        res = conn.execute(text("""
        INSERT INTO matches (
        match_key, tourney_id, tourney_name, tourney_level, surface, tourney_date, match_num,
        round, winner_name, loser_name, score, minutes,
        winner_rank, loser_rank
        )
        VALUES (
        :match_key, :tourney_id, :tourney_name, :tourney_level, :surface, :tourney_date, :match_num,
        :round, :winner_name, :loser_name, :score, :minutes,
        :winner_rank, :loser_rank
        )
        ON CONFLICT (match_key) DO NOTHING;
        """), params)

        # rowcount should be 1 if inserted, 0 if conflict/no-op
        try:
            inserted += int(res.rowcount or 0)
        except Exception:
            # some DB/drivers may not reliably report rowcount
            pass

    return inserted

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def k_factor(matches_played: int, k_min: float = 16.0, k_max: float = 40.0) -> float:
    """
    Dynamic K: higher early, tapers as matches_played grows.
    - 0 matches => ~k_max
    - ~100+ matches => approaches k_min
    """
    mp = max(0, int(matches_played))
    return k_min + (k_max - k_min) * math.exp(-mp / 40.0)

def tourney_multiplier(tourney_level: str | None) -> float:
    """
    Slightly increase Elo update size for bigger events.
    Common ATP-style codes:
      G = Grand Slam
      M = Masters 1000
    If your dataset differs, this safely defaults to 1.0.
    """
    lvl = (tourney_level or "").strip().upper()
    if lvl == "G":
        return 1.25   # Grand Slams
    if lvl == "M":
        return 1.13   # Masters 1000
    return 1.0

def date_from_yyyymmdd(x: int | None) -> date | None:
    if not x:
        return None
    s = str(int(x))
    # expects YYYYMMDD
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))

def recency_multiplier(
    match_day: date | None,
    anchor_day: date,
    recent_days: int = 500,      # ~1.5 years
    recent_boost: float = 1.30,  # 50% more weight in last 2 years
    old_floor: float = 0.50      # older matches still matter, but less
) -> float:
    """
    If match is within last 2 years (relative to anchor_day), boost updates.
    If older, downweight smoothly down to old_floor.
    """
    if match_day is None:
        return 1.0

    age_days = (anchor_day - match_day).days
    if age_days <= recent_days:
        return recent_boost

    # Older than 2 years: linearly taper down toward old_floor over next 8 years (customizable)
    # You can make this exponential if you prefer, but linear is simple + stable.
    taper_days = 8 * 365
    extra = min(age_days - recent_days, taper_days)
    frac = extra / taper_days  # 0..1
    return recent_boost - (recent_boost - old_floor) * frac

def recompute_elos(
    conn,
    base_elo: float = 1500.0,
    write_per_match_elos: bool = False,
) -> None:
    """
    Recompute:
      - overall Elo -> elo_overall
      - surface Elo -> elo_surface

    If write_per_match_elos=True, this will also try to write Elo before/after
    back onto the matches table (requires you to add those columns).
    """
    rows = conn.execute(text("""
        SELECT match_key, surface, tourney_level, tourney_date, winner_name, loser_name
        FROM matches
        WHERE winner_name IS NOT NULL
            AND loser_name IS NOT NULL
        ORDER BY tourney_date ASC NULLS LAST, match_num ASC NULLS LAST;
    """)).fetchall()

    # anchor = date of the latest match we have (so "last 2 years" is relative to your dataset)
    last_td = None
    for r in reversed(rows):
        # r = (match_key, surface, tourney_level, tourney_date, winner, loser)
        td = r[3]
        if td:
            last_td = int(td)
            break
    anchor_day = date_from_yyyymmdd(last_td) or date.today()

    overall = {}   # player -> elo
    overall_n = {} # player -> matches played

    surf = {}      # (player, surface) -> elo
    surf_n = {}    # (player, surface) -> matches played

    def get_overall(p: str) -> float:
        return float(overall.get(p, base_elo))

    def get_surface(p: str, s: str) -> float:
        return float(surf.get((p, s), base_elo))

    def inc_overall(p: str) -> int:
        overall_n[p] = overall_n.get(p, 0) + 1
        return overall_n[p]

    def inc_surface(p: str, s: str) -> int:
        surf_n[(p, s)] = surf_n.get((p, s), 0) + 1
        return surf_n[(p, s)]

    for match_key, surface, tourney_level, tourney_date, winner, loser in rows:
        surface = SURFACE_MAP.get(surface or "Unknown", surface or "Unknown")
        mult = tourney_multiplier(tourney_level)

        match_day = date_from_yyyymmdd(int(tourney_date) if tourney_date else None)
        mult *= recency_multiplier(match_day, anchor_day)   

        # ----- OVERALL ELO -----
        ra_o = get_overall(winner)
        rb_o = get_overall(loser)
        ea_o = expected_score(ra_o, rb_o)

        k_w_o = k_factor(overall_n.get(winner, 0)) * mult
        k_l_o = k_factor(overall_n.get(loser, 0)) * mult

        ra_o_new = ra_o + k_w_o * (1.0 - ea_o)
        rb_o_new = rb_o + k_l_o * (0.0 - (1.0 - ea_o))

        overall[winner] = ra_o_new
        overall[loser] = rb_o_new
        inc_overall(winner)
        inc_overall(loser)

        # ----- SURFACE ELO -----
        ra_s = get_surface(winner, surface)
        rb_s = get_surface(loser, surface)
        ea_s = expected_score(ra_s, rb_s)

        k_w_s = k_factor(surf_n.get((winner, surface), 0)) * mult
        k_l_s = k_factor(surf_n.get((loser, surface), 0)) * mult

        ra_s_new = ra_s + k_w_s * (1.0 - ea_s)
        rb_s_new = rb_s + k_l_s * (0.0 - (1.0 - ea_s))

        surf[(winner, surface)] = ra_s_new
        surf[(loser, surface)] = rb_s_new
        inc_surface(winner, surface)
        inc_surface(loser, surface)

        # ----- OPTIONAL: STORE PER-MATCH ELO BACK INTO matches -----
        if write_per_match_elos:
            # Requires columns like:
            # overall_winner_elo_before, overall_winner_elo_after,
            # overall_loser_elo_before,  overall_loser_elo_after,
            # surface_winner_elo_before, surface_winner_elo_after,
            # surface_loser_elo_before,  surface_loser_elo_after
            try:
                conn.execute(text("""
                    UPDATE matches
                    SET
                      overall_winner_elo_before = :owb,
                      overall_winner_elo_after  = :owa,
                      overall_loser_elo_before  = :olb,
                      overall_loser_elo_after   = :ola,
                      surface_winner_elo_before = :swb,
                      surface_winner_elo_after  = :swa,
                      surface_loser_elo_before  = :slb,
                      surface_loser_elo_after   = :sla
                    WHERE match_key = :mk;
                """), {
                    "mk": match_key,
                    "owb": float(ra_o), "owa": float(ra_o_new),
                    "olb": float(rb_o), "ola": float(rb_o_new),
                    "swb": float(ra_s), "swa": float(ra_s_new),
                    "slb": float(rb_s), "sla": float(rb_s_new),
                })
            except Exception:
                # If columns don't exist, ignore.
                pass

    # ---------- WRITE OVERALL TABLE ----------
    conn.execute(text("DELETE FROM elo_overall;"))
    for player, elo in overall.items():
        conn.execute(text("""
            INSERT INTO elo_overall (player_name, elo, matches_played, last_updated)
            VALUES (:p, :e, :m, NOW())
            ON CONFLICT (player_name) DO UPDATE SET
                elo = EXCLUDED.elo,
                matches_played = EXCLUDED.matches_played,
                last_updated = NOW();
        """), {
            "p": player,
            "e": float(elo),
            "m": int(overall_n.get(player, 0)),
        })

    # ---------- WRITE SURFACE TABLE ----------
    conn.execute(text("DELETE FROM elo_surface;"))
    for (player, surface), elo in surf.items():
        conn.execute(text("""
            INSERT INTO elo_surface (player_name, surface, elo, matches_played, last_updated)
            VALUES (:p, :s, :e, :m, NOW())
            ON CONFLICT (player_name, surface) DO UPDATE SET
                elo = EXCLUDED.elo,
                matches_played = EXCLUDED.matches_played,
                last_updated = NOW();
        """), {
            "p": player,
            "s": surface,
            "e": float(elo),
            "m": int(surf_n.get((player, surface), 0)),
        })

def backfill_h2h(conn) -> None:
    """
    Rebuild h2h table from matches using SQL aggregation (FAST).
    Produces:
      - surface-specific rows (Hard/Clay/Grass/Carpet/Unknown)
      - an "All" surface row per pair
    """
    # Ensure table exists
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS h2h (
      player_lo TEXT NOT NULL,
      player_hi TEXT NOT NULL,
      surface   TEXT NOT NULL,
      lo_wins   INT  NOT NULL DEFAULT 0,
      hi_wins   INT  NOT NULL DEFAULT 0,
      last_match_date INT NULL,
      PRIMARY KEY (player_lo, player_hi, surface)
    );
    """))

    # Rebuild
    conn.execute(text("TRUNCATE h2h;"))

    base_where = """
    WHERE winner_name IS NOT NULL
      AND loser_name  IS NOT NULL
      AND COALESCE(score, '') NOT ILIKE '%w/o%'
    """

    # 1) Surface-specific H2H
    conn.execute(text(f"""
    INSERT INTO h2h (player_lo, player_hi, surface, lo_wins, hi_wins, last_match_date)
    SELECT
      LEAST(winner_name, loser_name)  AS player_lo,
      GREATEST(winner_name, loser_name) AS player_hi,
      COALESCE(surface, 'Unknown')    AS surface,
      SUM(CASE WHEN winner_name = LEAST(winner_name, loser_name) THEN 1 ELSE 0 END) AS lo_wins,
      SUM(CASE WHEN winner_name = GREATEST(winner_name, loser_name) THEN 1 ELSE 0 END) AS hi_wins,
      MAX(tourney_date) AS last_match_date
    FROM matches
    {base_where}
    GROUP BY 1,2,3;
    """))

    # 2) All-surfaces H2H
    conn.execute(text(f"""
    INSERT INTO h2h (player_lo, player_hi, surface, lo_wins, hi_wins, last_match_date)
    SELECT
      LEAST(winner_name, loser_name)  AS player_lo,
      GREATEST(winner_name, loser_name) AS player_hi,
      'All'                           AS surface,
      SUM(CASE WHEN winner_name = LEAST(winner_name, loser_name) THEN 1 ELSE 0 END) AS lo_wins,
      SUM(CASE WHEN winner_name = GREATEST(winner_name, loser_name) THEN 1 ELSE 0 END) AS hi_wins,
      MAX(tourney_date) AS last_match_date
    FROM matches
    {base_where}
    GROUP BY 1,2;
    """))

def build_event_key(tourney_name: str | None, surface: str | None) -> str:
    """
    Stable across years. Does NOT include tourney_level.
    """
    n = (tourney_name or "").strip().lower()
    if n == "shanghai":
        n = "shanghai masters"
    n = n or "unknown"

    s = (surface or "Unknown").strip()
    return f"{n}|{s}"

def backfill_event_record(conn) -> None:
    """
    Aggregate record across all years for the same tournament name + surface.
    (tourney_level is intentionally ignored because it becomes unreliable in later years)
    """

    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS player_event_record (
      player_name TEXT NOT NULL,
      event_key   TEXT NOT NULL,
      wins        INT  NOT NULL DEFAULT 0,
      losses      INT  NOT NULL DEFAULT 0,
      last_played_date INT NULL,
      PRIMARY KEY (player_name, event_key)
    );
    """))

    conn.execute(text("TRUNCATE player_event_record;"))

    base_where = """
    WHERE winner_name IS NOT NULL
      AND loser_name  IS NOT NULL
      AND COALESCE(score, '') NOT ILIKE '%w/o%'
    """

    # Build event_key in SQL so it's fast
    conn.execute(text(f"""
    INSERT INTO player_event_record (player_name, event_key, wins, losses, last_played_date)
    SELECT
      player_name,
      event_key,
      SUM(win)  AS wins,
      SUM(loss) AS losses,
      MAX(tourney_date) AS last_played_date
    FROM (
      SELECT
        winner_name AS player_name,
        (LOWER(TRIM(tourney_name)) || '|' || COALESCE(surface,'Unknown')) AS event_key,
        1 AS win,
        0 AS loss,
        tourney_date
      FROM matches
      {base_where}

      UNION ALL

      SELECT
        loser_name AS player_name,
        (LOWER(TRIM(tourney_name)) || '|' || COALESCE(surface,'Unknown')) AS event_key,
        0 AS win,
        1 AS loss,
        tourney_date
      FROM matches
      {base_where}
    ) x
    GROUP BY player_name, event_key;
    """))

def backfill_h2h_event(conn) -> None:
    """
    H2H by (player pair, event_key).
    event_key = LOWER(TRIM(tourney_name)) || '|' || COALESCE(surface,'Unknown')
    """
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS h2h_event (
      player_lo TEXT NOT NULL,
      player_hi TEXT NOT NULL,
      event_key  TEXT NOT NULL,
      lo_wins    INT  NOT NULL DEFAULT 0,
      hi_wins    INT  NOT NULL DEFAULT 0,
      last_match_date INT NULL,
      PRIMARY KEY (player_lo, player_hi, event_key)
    );
    """))

    conn.execute(text("TRUNCATE h2h_event;"))

    base_where = """
    WHERE winner_name IS NOT NULL
      AND loser_name  IS NOT NULL
      AND COALESCE(score, '') NOT ILIKE '%w/o%'
      AND tourney_name IS NOT NULL
    """

    conn.execute(text(f"""
    INSERT INTO h2h_event (player_lo, player_hi, event_key, lo_wins, hi_wins, last_match_date)
    SELECT
      LEAST(winner_name, loser_name) AS player_lo,
      GREATEST(winner_name, loser_name) AS player_hi,
      (LOWER(TRIM(tourney_name)) || '|' || COALESCE(surface,'Unknown')) AS event_key,
      SUM(CASE WHEN winner_name = LEAST(winner_name, loser_name) THEN 1 ELSE 0 END) AS lo_wins,
      SUM(CASE WHEN winner_name = GREATEST(winner_name, loser_name) THEN 1 ELSE 0 END) AS hi_wins,
      MAX(tourney_date) AS last_match_date
    FROM matches
    {base_where}
    GROUP BY 1,2,3;
    """))

def ensure_and_backfill_rank(conn, *, use_elo_fallback: bool = True) -> dict:
    """
    Ensures public.player_state.rank exists and backfills it from matches winner_rank/loser_rank.
    Optionally fills any remaining NULLs using an Elo-based proxy rank.

    Returns simple stats dict for logging.
    """

    # 1) Ensure rank column exists
    conn.execute(text("""
        ALTER TABLE public.player_state
        ADD COLUMN IF NOT EXISTS rank INT;
    """))

    # 2) Backfill from most recent match rank (winner_rank + loser_rank)
    conn.execute(text("""
        WITH ranked AS (
          SELECT winner_name AS player_name,
                 winner_rank AS rank,
                 tourney_date
          FROM public.matches
          WHERE winner_rank IS NOT NULL

          UNION ALL

          SELECT loser_name AS player_name,
                 loser_rank AS rank,
                 tourney_date
          FROM public.matches
          WHERE loser_rank IS NOT NULL
        ),
        latest AS (
          SELECT DISTINCT ON (player_name)
                 player_name, rank
          FROM ranked
          ORDER BY player_name, tourney_date DESC
        )
        UPDATE public.player_state ps
        SET rank = l.rank
        FROM latest l
        WHERE ps.player_name = l.player_name;
    """))

    # 3) Optional fallback: Elo-based proxy for anyone still NULL
    if use_elo_fallback:
        conn.execute(text("""
            WITH e AS (
              SELECT player_name,
                     ROW_NUMBER() OVER (ORDER BY COALESCE(elo_overall, 1500) DESC) AS rk
              FROM public.player_state
            )
            UPDATE public.player_state ps
            SET rank = e.rk
            FROM e
            WHERE ps.player_name = e.player_name
              AND ps.rank IS NULL;
        """))

    # 4) Stats + stamp into model_state (nice for debugging on your /model page)
    totals = conn.execute(text("""
        SELECT COUNT(*) AS total,
               COUNT(rank) AS non_null_rank,
               COUNT(*) - COUNT(rank) AS null_rank
        FROM public.player_state;
    """)).mappings().one()

    conn.execute(text("""
        INSERT INTO public.model_state(key, value)
        VALUES ('last_rank_backfill_at', :v)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
    """), {"v": datetime.now(timezone.utc).isoformat()})

    return dict(totals)

def set_state(conn, key: str, value: str) -> None:
    conn.execute(text("""
        INSERT INTO model_state (key, value)
        VALUES (:k, :v)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
    """), {"k": key, "v": value})

def get_state(conn, key: str) -> str | None:
    row = conn.execute(text("SELECT value FROM model_state WHERE key=:k;"), {"k": key}).fetchone()
    return row[0] if row else None

def backfill_years(conn, start_year: int, end_year: int) -> None:
    for y in range(start_year, end_year + 1):
        state_key = f"backfilled_{y}"
        if get_state(conn, state_key) == "1":
            continue

        rows = fetch_year_rows(y)
        upsert_matches(conn, rows)
        set_state(conn, state_key, "1")

    set_state(conn, "last_backfill", f"{start_year}-{end_year}")
    set_state(conn, "last_ingest_at", datetime.utcnow().isoformat() + "Z")

# PREDICTION UTILITIES

def log_loss(y_true, p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y_true * math.log(p) + (1 - y_true) * math.log(1 - p))

def canon_pair(a: str, b: str) -> tuple[str, str, bool]:
    a = (a or "").strip()
    b = (b or "").strip()
    if a <= b:
        return a, b, True
    return b, a, False

def smoothed_rate(wins: int, losses: int, prior: float = 0.5, prior_games: int = 6) -> float:
    w = max(0, int(wins or 0))
    l = max(0, int(losses or 0))
    pg = max(1, int(prior_games))
    return (w + prior * pg) / (w + l + pg)

def expected_from_elodiff(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(diff) / 650.0))

def get_elos(conn, player: str, surface: str, base_elo: float = 1500.0) -> tuple[float, float]:
    s = (surface or "Unknown").strip()
    row_s = conn.execute(text("""
        SELECT elo FROM elo_surface WHERE player_name=:p AND surface=:s
    """), {"p": player, "s": s}).fetchone()

    row_o = conn.execute(text("""
        SELECT elo FROM elo_overall WHERE player_name=:p
    """), {"p": player}).fetchone()

    surf_elo = float(row_s[0]) if row_s and row_s[0] is not None else float(base_elo)
    ov_elo   = float(row_o[0]) if row_o and row_o[0] is not None else float(base_elo)
    return surf_elo, ov_elo

def get_h2h(conn, a: str, b: str, surface: str) -> tuple[int, int, int, int]:
    """
    Returns: (a_wins_surface, b_wins_surface, a_wins_all, b_wins_all)
    """
    s = (surface or "Unknown").strip()
    lo, hi, a_is_lo = canon_pair(a, b)

    def fetch(surf_key: str) -> tuple[int, int]:
        r = conn.execute(text("""
            SELECT lo_wins, hi_wins
            FROM h2h
            WHERE player_lo=:lo AND player_hi=:hi AND surface=:s
        """), {"lo": lo, "hi": hi, "s": surf_key}).fetchone()
        if not r:
            return (0, 0)
        lo_w, hi_w = int(r[0] or 0), int(r[1] or 0)
        return (lo_w, hi_w) if a_is_lo else (hi_w, lo_w)

    a_ws, b_ws = fetch(s)
    a_wa, b_wa = fetch("All")
    return a_ws, b_ws, a_wa, b_wa

def get_h2h_event(conn, a: str, b: str, event_key: str) -> tuple[int, int]:
    """
    Returns (a_wins_event, b_wins_event) for this tournament+surface.
    """
    lo, hi, a_is_lo = canon_pair(a, b)
    r = conn.execute(text("""
        SELECT lo_wins, hi_wins
        FROM h2h_event
        WHERE player_lo=:lo AND player_hi=:hi AND event_key=:ek
    """), {"lo": lo, "hi": hi, "ek": event_key}).fetchone()

    if not r:
        return (0, 0)

    lo_w, hi_w = int(r[0] or 0), int(r[1] or 0)
    return (lo_w, hi_w) if a_is_lo else (hi_w, lo_w)

def get_event_record(conn, player: str, event_key: str) -> tuple[int, int]:
    r = conn.execute(text("""
        SELECT wins, losses
        FROM player_event_record
        WHERE player_name=:p AND event_key=:k
    """), {"p": player, "k": event_key}).fetchone()
    if not r:
        return (0, 0)
    return int(r[0] or 0), int(r[1] or 0)

def get_labeled_matches(conn, year: int):
    return conn.execute(text("""
        SELECT
            winner_name,
            loser_name,
            surface,
            tourney_name
        FROM matches
        WHERE tourney_date BETWEEN :y1 AND :y2
          AND score NOT ILIKE '%w/o%'
    """), {
        "y1": year * 10000 + 101,
        "y2": year * 10000 + 1231
    }).fetchall()

def predict_h2h(conn, player_a: str, player_b: str, surface: str, tourney_name: str | None = None) -> dict:
    """
    Uses:
      - blended Elo (75% surface, 25% overall)
      - H2H adj (surface + all)
      - optional event record bonus (if tourney_name passed)
    """
    s = (surface or "Unknown").strip()

    a_s, a_o = get_elos(conn, player_a, s)
    b_s, b_o = get_elos(conn, player_b, s)

    a_blend = 0.75 * a_s + 0.25 * a_o
    b_blend = 0.75 * b_s + 0.25 * b_o

    a_ws, b_ws, a_wa, b_wa = get_h2h(conn, player_a, player_b, s)
    p_h2h_s = smoothed_rate(a_ws, b_ws, prior=0.5, prior_games=4)
    p_h2h_a = smoothed_rate(a_wa, b_wa, prior=0.5, prior_games=8)

    # small Elo-like adjustment
    h2h_edge = (p_h2h_s - 0.5) * 120.0 + (p_h2h_a - 0.5) * 40.0

    event_edge = 0.0
    event_h2h_edge = 0.0
    a_we = 0
    b_we = 0

    if tourney_name:
        ek = build_event_key(tourney_name, s)

        # your existing event record edge
        a_w, a_l = get_event_record(conn, player_a, ek)
        b_w, b_l = get_event_record(conn, player_b, ek)
        a_r = smoothed_rate(a_w, a_l, prior=0.5, prior_games=10)
        b_r = smoothed_rate(b_w, b_l, prior=0.5, prior_games=10)
        event_edge = (a_r - b_r) * 45.0

        # ✅ NEW: event-specific H2H edge (only matches at this tournament+surface)
        a_we, b_we = get_h2h_event(conn, player_a, player_b, ek)
        p_h2h_e = smoothed_rate(a_we, b_we, prior=0.5, prior_games=4)
        event_h2h_edge = (p_h2h_e - 0.5) * 80.0   # tune this weight

    diff = (a_blend - b_blend) + h2h_edge + event_edge + event_h2h_edge
    p_a = expected_from_elodiff(diff)

    return {
        "player_a": player_a,
        "player_b": player_b,
        "surface": s,
        "p_a": float(p_a),
        "p_b": float(1.0 - p_a),
        "breakdown": {
            "a_surface_elo": float(a_s),
            "a_overall_elo": float(a_o),
            "b_surface_elo": float(b_s),
            "b_overall_elo": float(b_o),
            "a_blend": float(a_blend),
            "b_blend": float(b_blend),
            "a_h2h_surface": int(a_ws),
            "b_h2h_surface": int(b_ws),
            "a_h2h_all": int(a_wa),
            "b_h2h_all": int(b_wa),
            "event_h2h_a_wins": int(a_we),
            "event_h2h_b_wins": int(b_we),
            "h2h_edge_elo": float(h2h_edge),
            "event_edge_elo": float(event_edge),
            "event_h2h_edge_elo": float(event_h2h_edge),
            "final_elo_diff": float(diff),
        }
    }

def predict_match_proba(
    conn,
    player_a: str,
    player_b: str,
    surface: str,
    tourney_name: str | None,
    *,
    elo_scale: float,
    surface_weight: float,
    h2h_surface_w: float,
    h2h_all_w: float,
    event_w: float,
) -> float:
    s = surface.strip()

    a_s, a_o = get_elos(conn, player_a, s)
    b_s, b_o = get_elos(conn, player_b, s)

    a_blend = surface_weight * a_s + (1 - surface_weight) * a_o
    b_blend = surface_weight * b_s + (1 - surface_weight) * b_o

    a_ws, b_ws, a_wa, b_wa = get_h2h(conn, player_a, player_b, s)

    p_h2h_s = smoothed_rate(a_ws, b_ws, prior_games=4)
    p_h2h_a = smoothed_rate(a_wa, b_wa, prior_games=8)

    h2h_edge = (
        (p_h2h_s - 0.5) * h2h_surface_w +
        (p_h2h_a - 0.5) * h2h_all_w
    )

    event_edge = 0.0
    if tourney_name:
        ek = build_event_key(tourney_name, s)
        aw, al = get_event_record(conn, player_a, ek)
        bw, bl = get_event_record(conn, player_b, ek)
        ar = smoothed_rate(aw, al, prior_games=10)
        br = smoothed_rate(bw, bl, prior_games=10)
        event_edge = (ar - br) * event_w

    diff = (a_blend - b_blend) + h2h_edge + event_edge

    # 🔑 probability mapping
    return 1.0 / (1.0 + 10 ** (-diff / elo_scale))

def tournament_odds_no_draw(conn, tourney_name: str, surface: str, pool_n: int = 64) -> list[dict]:
    """
    Returns pie-chart-ready probabilities that sum to 1.0.
    Candidate pool = top N by surface Elo on that surface.
    Strength = 70% surface Elo + 30% overall Elo + small event record bonus.
    Uses BATCHED queries to avoid DB timeouts.
    """
    s = (surface or "Unknown").strip()
    ek = build_event_key(tourney_name, s)

    # 1) pick a pool of likely entrants (1 query)
    rows = conn.execute(text("""
        SELECT player_name
        FROM elo_surface
        WHERE surface=:s
        ORDER BY elo DESC
        LIMIT :n
    """), {"s": s, "n": int(pool_n)}).fetchall()
    players = [r[0] for r in rows]
    if not players:
        return []

    # 2) batch fetch overall elos (1 query)
    overall_rows = conn.execute(text("""
        SELECT player_name, elo
        FROM elo_overall
        WHERE player_name = ANY(:players)
    """), {"players": players}).fetchall()
    overall_map = {r[0]: float(r[1]) for r in overall_rows}

    # 3) batch fetch surface elos (1 query)
    surface_rows = conn.execute(text("""
        SELECT player_name, elo
        FROM elo_surface
        WHERE surface=:s AND player_name = ANY(:players)
    """), {"s": s, "players": players}).fetchall()
    surface_map = {r[0]: float(r[1]) for r in surface_rows}

    # 4) batch fetch event records for that event_key (1 query)
    rec_rows = conn.execute(text("""
        SELECT player_name, wins, losses
        FROM player_event_record
        WHERE event_key = :ek AND player_name = ANY(:players)
    """), {"ek": ek, "players": players}).fetchall()
    rec_map = {r[0]: (int(r[1] or 0), int(r[2] or 0)) for r in rec_rows}

    strengths = []
    for p in players:
        p_s = float(surface_map.get(p, 1500.0))
        p_o = float(overall_map.get(p, 1500.0))
        blend = 0.70 * p_s + 0.30 * p_o

        w, l = rec_map.get(p, (0, 0))
        r = smoothed_rate(w, l, prior=0.5, prior_games=10)
        rec_bonus = (r - 0.5) * 200.0

        strength = blend + rec_bonus
        strengths.append((p, float(strength), p_s, p_o, int(w), int(l), float(rec_bonus)))

    # softmax
    max_s = max(x[1] for x in strengths)
    temp = 125.0
    exps = [math.exp((x[1] - max_s) / temp) for x in strengths]
    total = sum(exps) or 1.0

    out = []
    for (p, strength, p_s, p_o, w, l, rec_bonus), e in zip(strengths, exps):
        out.append({
            "player": p,
            "p": float(e / total),
            "surface_elo": float(p_s),
            "overall_elo": float(p_o),
            "event_wins": int(w),
            "event_losses": int(l),
            "event_bonus_elo": float(rec_bonus),
            "strength": float(strength),
        })

    out.sort(key=lambda d: d["p"], reverse=True)
    return out

def add_asof_days_since_last_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pre-match days-since-last-match for winner and loser, computed ONLY from prior matches.
    Requires: tourney_date (yyyymmdd int), winner_name, loser_name.
    Output cols:
      - w_days_since_last, l_days_since_last, days_since_last_diff
      - w_days_missing, l_days_missing
    """
    d = df.sort_values(["tourney_date"]).copy()

    # track last match date (yyyymmdd) seen per player
    last_date = {}

    w_days = []
    l_days = []
    w_miss = []
    l_miss = []

    # convert yyyymmdd -> pandas datetime for accurate day diffs
    def to_dt(x):
        try:
            return pd.to_datetime(str(int(x)), format="%Y%m%d")
        except Exception:
            return pd.NaT

    for _, r in d.iterrows():
        cur_dt = to_dt(r["tourney_date"])
        w = str(r["winner_name"])
        l = str(r["loser_name"])

        w_last = last_date.get(w)
        l_last = last_date.get(l)

        if w_last is None or pd.isna(cur_dt):
            w_days.append(np.nan)
            w_miss.append(1)
        else:
            w_days.append((cur_dt - w_last).days)
            w_miss.append(0)

        if l_last is None or pd.isna(cur_dt):
            l_days.append(np.nan)
            l_miss.append(1)
        else:
            l_days.append((cur_dt - l_last).days)
            l_miss.append(0)

        # update last_date AFTER computing features (no lookahead)
        if not pd.isna(cur_dt):
            last_date[w] = cur_dt
            last_date[l] = cur_dt

    d["w_days_since_last"] = w_days
    d["l_days_since_last"] = l_days
    d["days_since_last_diff"] = d["w_days_since_last"] - d["l_days_since_last"]
    d["w_days_missing"] = w_miss
    d["l_days_missing"] = l_miss

    return d

def update_model(conn, start_year: int, end_year: int) -> None:
    print("[ML] backfill_years start")
    backfill_years(conn, start_year, end_year)

    print("[ML] recompute_elos start")
    recompute_elos(conn, write_per_match_elos=False)

    print("[ML] backfill_h2h start")
    backfill_h2h(conn)

    print("[ML] backfill_event_record start")
    backfill_event_record(conn)
    print("[ML] backfill_event_record done")

    print("[ML] backfill_h2h_event start")
    backfill_h2h_event(conn)

    # ✅ ADD THIS near the end, after player_state exists/updated
    stats = ensure_and_backfill_rank(conn, use_elo_fallback=True)
    print("[ML] rank backfill stats:", stats)

    print("[ML] done — writing state")
    set_state(conn, "last_model_update_at", datetime.utcnow().isoformat() + "Z")