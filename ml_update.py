import csv
import io
import math
import requests
from datetime import datetime
from sqlalchemy import text

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

def make_match_key(row: dict) -> str:
    return f"{row.get('tourney_id','')}|{row.get('tourney_date','')}|{row.get('match_num','')}"

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
            "surface": surface,
            "tourney_date": int(row["tourney_date"]) if row.get("tourney_date") else None,
            "match_num": int(row["match_num"]) if row.get("match_num") else None,
            "round": row.get("round"),
            "winner_name": row.get("winner_name"),
            "loser_name": row.get("loser_name"),
            "score": row.get("score"),
            "minutes": int(row["minutes"]) if row.get("minutes") else None,
        }

        res = conn.execute(text("""
            INSERT INTO matches (
              match_key, tourney_id, tourney_name, surface, tourney_date, match_num,
              round, winner_name, loser_name, score, minutes
            )
            VALUES (
              :match_key, :tourney_id, :tourney_name, :surface, :tourney_date, :match_num,
              :round, :winner_name, :loser_name, :score, :minutes
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
        SELECT match_key, surface, winner_name, loser_name
        FROM matches
        WHERE winner_name IS NOT NULL
          AND loser_name IS NOT NULL
        ORDER BY tourney_date ASC NULLS LAST, match_num ASC NULLS LAST;
    """)).fetchall()

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

    for match_key, surface, winner, loser in rows:
        surface = SURFACE_MAP.get(surface or "Unknown", surface or "Unknown")

        # ----- OVERALL ELO -----
        ra_o = get_overall(winner)
        rb_o = get_overall(loser)
        ea_o = expected_score(ra_o, rb_o)

        k_w_o = k_factor(overall_n.get(winner, 0))
        k_l_o = k_factor(overall_n.get(loser, 0))

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

        k_w_s = k_factor(surf_n.get((winner, surface), 0))
        k_l_s = k_factor(surf_n.get((loser, surface), 0))

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

def update_model(conn, start_year: int, end_year: int) -> None:
    backfill_years(conn, start_year, end_year)
    # write_per_match_elos=False unless you add the columns
    recompute_elos(conn, write_per_match_elos=False)
    set_state(conn, "last_model_update_at", datetime.utcnow().isoformat() + "Z")
