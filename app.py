import os
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from typing import Optional
from urllib.parse import urlencode 
from fastapi.staticfiles import StaticFiles
from pipeline.ml_update import update_model, predict_h2h, tournament_odds_no_draw
from runtime.tennis_model import TennisPredictor, TourneyCtx
from pydantic import BaseModel
import difflib
import json
from datetime import datetime
import re

# --- League config ---
EVENTS_13 = [
    ("AO",  "Australian Open", "slam"),
    ("RG",  "Roland Garros", "slam"),
    ("WIM", "Wimbledon", "slam"),
    ("USO", "US Open", "slam"),
    ("IW",  "Indian Wells", "masters"),
    ("MIA", "Miami", "masters"),
    ("MON", "Monte Carlo", "masters"),
    ("MAD", "Madrid", "masters"),
    ("ROM", "Rome", "masters"),
    ("CAN", "Canada (Toronto/Montreal)", "masters"),
    ("CIN", "Cincinnati", "masters"),
    ("SHA", "Shanghai", "masters"),
    ("PAR", "Paris", "masters"),
]

EVENTS_ORDERED = [
    ("AO", "Australian Open", "slam"),
    ("IW", "Indian Wells", "masters"),
    ("MIA", "Miami", "masters"),
    ("MON", "Monte Carlo", "masters"),
    ("MAD", "Madrid", "masters"),
    ("ROM", "Rome", "masters"),
    ("RG", "Roland Garros", "slam"),
    ("WIM", "Wimbledon", "slam"),
    ("CAN", "Canada", "masters"),
    ("CIN", "Cincinnati", "masters"),
    ("USO", "US Open", "slam"),
    ("SHA", "Shanghai", "masters"),
    ("PAR", "Paris", "masters"),
]

EVENT_META = {
    "AO":  {"tourney_name": "Australian Open", "surface": "Hard"},
    "IW":  {"tourney_name": "Indian Wells Masters", "surface": "Hard"},
    "MIA": {"tourney_name": "Miami Masters", "surface": "Hard"},
    "MON": {"tourney_name": "Monte Carlo Masters", "surface": "Clay"},
    "MAD": {"tourney_name": "Madrid Masters", "surface": "Clay"},
    "ROM": {"tourney_name": "Rome Masters", "surface": "Clay"},
    "RG":  {"tourney_name": "Roland Garros", "surface": "Clay"},
    "WIM": {"tourney_name": "Wimbledon", "surface": "Grass"},
    "CAN": {"tourney_name": "Canada Masters", "surface": "Hard"},
    "CIN": {"tourney_name": "Cincinnati Masters", "surface": "Hard"},
    "USO": {"tourney_name": "US Open", "surface": "Hard"},
    "SHA": {"tourney_name": "Shanghai Masters", "surface": "Hard"},
    "PAR": {"tourney_name": "Paris Masters", "surface": "Hard"},
}

ALLOWED_ROUNDS = ["W", "F", "SF", "QF", "R16", "R32", "R64", "R128"]

# You can tune these later; Slams will use R128 sometimes, Masters usually won't.
POINTS = {"W": 100, "F": 60, "SF": 40, "QF": 25, "R16": 15, "R32": 8, "R64": 4, "R128": 2}

# --- Environment ---
DATABASE_URL = os.environ["DATABASE_URL"]  # Supabase/Render Postgres URL
LEAGUE_YEAR = int(os.environ.get("LEAGUE_YEAR", "2026"))
COMMISSIONER_KEY = os.environ.get("COMMISSIONER_KEY", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
xgb_predictor = TennisPredictor(engine)


def init_db() -> None:
    """Creates tables and seeds events once."""
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS people (
          name TEXT PRIMARY KEY
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,         -- e.g., AO2026
        short_id TEXT NOT NULL,      -- AO
        name TEXT NOT NULL,
        level TEXT NOT NULL,
        sort_order INT NOT NULL,
        year INT NOT NULL
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
          event_id TEXT NOT NULL,
          person_name TEXT NOT NULL,
          player_name TEXT NOT NULL,
          PRIMARY KEY (event_id, person_name),
          FOREIGN KEY (event_id) REFERENCES events(id),
          FOREIGN KEY (person_name) REFERENCES people(name)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS results (
          event_id TEXT NOT NULL,
          player_name TEXT NOT NULL,
          round_reached TEXT NOT NULL,
          PRIMARY KEY (event_id, player_name),
          FOREIGN KEY (event_id) REFERENCES events(id)
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS matches (
        match_key TEXT PRIMARY KEY,
        tourney_id TEXT,
        tourney_name TEXT,
        tourney_level TEXT,
        surface TEXT,
        tourney_date INT,
        match_num INT,
        round TEXT,
        winner_name TEXT,
        loser_name TEXT,
        score TEXT,
        minutes INT,
        winner_rank INT,
        loser_rank INT
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS elo_overall (
        player_name TEXT PRIMARY KEY,
        elo NUMERIC NOT NULL,
        matches_played INT NOT NULL DEFAULT 0,
        last_updated TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS elo_surface (
        player_name TEXT NOT NULL,
        surface TEXT NOT NULL,
        elo NUMERIC NOT NULL,
        matches_played INT NOT NULL DEFAULT 0,
        last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (player_name, surface)
        );
        """))

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

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS bracket_posts (
          id BIGSERIAL PRIMARY KEY,
          year INT NOT NULL,
          event_short TEXT NOT NULL,
          event_id TEXT NOT NULL,
          title TEXT NOT NULL,
          content_json TEXT NOT NULL,  -- full model output as JSON string
          created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS model_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
        );
        """))

        # Seed 13 events for the configured year
        for idx, (short_id, name, level) in enumerate(EVENTS_ORDERED, start=1):
            event_id = f"{short_id}{LEAGUE_YEAR}"
            conn.execute(text("""
                INSERT INTO events (id, short_id, name, level, sort_order, year)
                VALUES (:id, :short_id, :name, :level, :sort_order, :year)
                ON CONFLICT (id) DO UPDATE SET
                    short_id = EXCLUDED.short_id,
                    name = EXCLUDED.name,
                    level = EXCLUDED.level,
                    sort_order = EXCLUDED.sort_order;
            """), {
                "id": event_id,
                "short_id": short_id,
                "name": name,
                "level": level,
                "sort_order": idx,
                "year": LEAGUE_YEAR
            })



@app.on_event("startup")
def startup():
    init_db()



def get_people() -> list[str]:
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT name FROM people ORDER BY name;")).fetchall()
    return [r[0] for r in rows]

def get_player_rank(conn, name: str) -> int:
    row = conn.execute(
        text("SELECT rank FROM public.player_state WHERE player_name=:p LIMIT 1"),
        {"p": name}
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 300

def get_events():
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, short_id, name, level FROM events WHERE year=:y ORDER BY id;"),
            {"y": LEAGUE_YEAR}
        ).fetchall()
    # list of dicts for templates
    return [{"id": r[0], "short_id": r[1], "name": r[2], "level": r[3]} for r in rows]

def get_conn():
    return engine.connect()

def normalize_name(s: str) -> str:
    return " ".join((s or "").strip().split())

def player_exists(conn, name: str) -> bool:
    name = normalize_name(name)
    if not name:
        return False
    r = conn.execute(text("""
        SELECT 1
        FROM (
          SELECT player_name FROM elo_overall
          UNION
          SELECT player_name FROM elo_surface
        ) p
        WHERE p.player_name = :n
        LIMIT 1
    """), {"n": name}).fetchone()
    return bool(r)


_PLACEHOLDER_RE = re.compile(r"^(Q|LL|WC)(\d+)?$", re.IGNORECASE)

def is_placeholder(name: str) -> bool:
    """
    Accept draw placeholders like:
      Q, Q1, Q2...
      LL, LL1...
      WC, WC1...
    """
    s = (name or "").strip()
    if not s:
        return False
    return bool(_PLACEHOLDER_RE.match(s))

def assert_player_exists(conn, name: str):
    if not player_exists(conn, name):
        # 422 = “you sent a value but it’s invalid”
        raise HTTPException(
            status_code=422,
            detail=f"Player '{name}' not found. Check spelling and pick from the dropdown suggestions."
        )

def points_case_sql() -> str:
    # Build a CASE expression from POINTS dict (keeps logic in one place)
    parts = [f"WHEN '{rnd}' THEN {pts}" for rnd, pts in POINTS.items()]
    return "CASE r.round_reached " + " ".join(parts) + " ELSE 0 END"


def calc_totals() -> list[tuple[str, int]]:
    """Total points per person across all events for the year."""
    case_expr = points_case_sql()
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
        SELECT p.person_name AS person,
               COALESCE(SUM({case_expr}), 0) AS total
        FROM predictions p
        JOIN events e ON e.id = p.event_id
        LEFT JOIN results r
          ON r.event_id = p.event_id
         AND r.player_name = p.player_name
        WHERE e.year = :year
        GROUP BY p.person_name
        ORDER BY total DESC, person ASC;
        """), {"year": LEAGUE_YEAR}).fetchall()
    return [(r[0], int(r[1])) for r in rows]

def rank_with_ties(totals: list[tuple[str, int]]) -> list[dict]:
    """
    Input:  [(person, total_points), ...] sorted DESC by points.
    Output: [{person, total, rank, medal}, ...] with tie-aware ranks/medals.

    Example ranks: 1, 1, 3, 4...
    """
    ranked = []
    prev_total = None
    rank = 0          # displayed rank (1-based)
    seen = 0          # number of rows processed

    for person, total in totals:
        seen += 1
        if prev_total is None or total != prev_total:
            rank = seen
            prev_total = total

        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = ""

        ranked.append({
            "person": person,
            "total": total,
            "rank": rank,
            "medal": medal,
        })

    return ranked


def calc_event_breakdown():
    """
    Returns rows for the home page table:
    event -> person -> (player, round, points)
    """
    case_expr = points_case_sql()
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
        SELECT e.id AS event_id, e.short_id, e.name,
               p.person_name, p.player_name,
               COALESCE(r.round_reached, '') AS round_reached,
               {case_expr} AS pts
        FROM predictions p
        JOIN events e ON e.id = p.event_id
        LEFT JOIN results r
          ON r.event_id = p.event_id
         AND r.player_name = p.player_name
        WHERE e.year = :year
        ORDER BY e.sort_order ASC, p.person_name;
        """), {"year": LEAGUE_YEAR}).fetchall()

    # Group into event blocks
    events = {}
    for event_id, short_id, name, person, player, rnd, pts in rows:
        events.setdefault(event_id, {"event_id": event_id, "short_id": short_id, "name": name, "rows": []})
        events[event_id]["rows"].append({
            "person": person,
            "player": player,
            "round": rnd or "—",
            "points": int(pts or 0),
        })
    return list(events.values())

def infer_default_round(n: int) -> str:
    if n == 128: return "R128"
    if n == 64: return "R64"
    if n == 32: return "R32"
    if n == 16: return "R16"
    if n == 8: return "QF"
    if n == 4: return "SF"
    if n == 2: return "F"
    return "R32"

def load_latest_bracket_post(event_short: str):
    with engine.begin() as conn:
        row = conn.execute(text("""
          SELECT title, content_json, params_json, created_at
          FROM bracket_posts
          WHERE year=:y AND event_short=:ev
          ORDER BY created_at DESC
          LIMIT 1
        """), {"y": LEAGUE_YEAR, "ev": event_short}).fetchone()

    if not row:
        return None

    title, content_json, params_json, created_at = row

    # params_json might be NULL for older rows; keep it safe
    params = {}
    if params_json:
        try:
            params = json.loads(params_json) if isinstance(params_json, str) else params_json
        except Exception:
            params = {}

    return {
        "title": title,
        "content": json.loads(content_json),
        "params": params,                 # ✅ THIS fixes your template crash
        "created_at": created_at,
    }


def compute_upset_spots(round1_matches: list[dict], top_n: int = 8):
    spots = []
    with engine.begin() as conn:
        for m in round1_matches:
            a, b = m.get("a",""), m.get("b","")
            if a.upper() == "BYE" or b.upper() == "BYE":
                continue

            p_a = float(m.get("p_a", 0.5))
            fav = a if p_a >= 0.5 else b
            dog = b if p_a >= 0.5 else a
            dog_p = (1 - p_a) if p_a >= 0.5 else p_a

            fav_r = get_player_rank(conn, fav)
            dog_r = get_player_rank(conn, dog)

            # keep only “true underdogs”: worse rank number + lower win chance
            if dog_r <= fav_r:
                continue

            spots.append({
                "match": f"{a} vs {b}",
                "underdog": dog,
                "p": dog_p,
                "underdog_rank": dog_r,
                "favorite_rank": fav_r,
            })

    spots.sort(key=lambda x: -x["p"])
    return spots[:top_n]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    people = get_people()
    events = get_events()
    totals = rank_with_ties(calc_totals())

    return templates.TemplateResponse("home.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "people": people,
        "events": events,
        "totals": totals,
    })

@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return templates.TemplateResponse("help.html", {
        "request": request,
        "year": LEAGUE_YEAR,
    })

@app.post("/add_person")
def add_person(name: str = Form(...)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name cannot be empty.")
    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO people(name) VALUES (:n)
          ON CONFLICT (name) DO NOTHING;
        """), {"n": name})
    return RedirectResponse("/", status_code=303)


@app.get("/picks", response_class=HTMLResponse)
def picks_page(
    request: Request,
    person: str = Query(default=""),
    event_id: str = Query(default=""),
):
    return templates.TemplateResponse("picks.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "people": get_people(),
        "events": get_events(),
        "selected_person": person,
        "selected_event_id": event_id,
    })



@app.post("/picks")
def submit_pick(
    person: str = Form(...),
    event_id: str = Form(...),
    player: str = Form(...),
):
    person = person.strip()
    event_id = event_id.strip()
    player = normalize_name(player)

    if not (person and event_id and player):
        raise HTTPException(400, "Missing fields.")

    with engine.begin() as conn:
        assert_player_exists(conn, player)

        # Ensure person exists (helps avoid “someone forgot to add Mom” errors)
        conn.execute(text("INSERT INTO people(name) VALUES (:n) ON CONFLICT (name) DO NOTHING;"), {"n": person})

        # Upsert prediction (one pick per person per event)
        conn.execute(text("""
          INSERT INTO predictions(event_id, person_name, player_name)
          VALUES (:e, :p, :pl)
          ON CONFLICT (event_id, person_name)
          DO UPDATE SET player_name = excluded.player_name;
        """), {"e": event_id, "p": person, "pl": player})

    qs = urlencode({"person": person, "event_id": event_id})
    return RedirectResponse(f"/picks?{qs}", status_code=303)

@app.get("/breakdown", response_class=HTMLResponse)
def breakdown_page(request: Request, event_id: Optional[str] = Query(default=None)):
    events = get_events()  
    breakdown = calc_event_breakdown()

    # Filter to one event if selected
    if event_id:
        breakdown = [ev for ev in breakdown if ev["event_id"] == event_id]

    return templates.TemplateResponse("breakdown.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": events,                     
        "selected_event_id": event_id or "",  
        "breakdown": breakdown,
    })

@app.get("/model", response_class=HTMLResponse)
def model_page(request: Request):
    with engine.begin() as conn:
        last = conn.execute(text("SELECT value FROM model_state WHERE key='last_model_update_at';")).fetchone()
        last = last[0] if last else "Never"

        backfill = conn.execute(text("SELECT value FROM model_state WHERE key='last_backfill';")).fetchone()
        backfill = backfill[0] if backfill else "Not run"

        match_count = conn.execute(text("SELECT COUNT(*) FROM matches;")).fetchone()[0]
        elo_count = conn.execute(text("SELECT COUNT(*) FROM elo_surface;")).fetchone()[0]

    return templates.TemplateResponse("model.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "last_update": last,
        "backfill_range": backfill,
        "match_count": match_count,
        "elo_count": elo_count,
    })


@app.post("/model/update")
def model_update(commissioner_key: str = Form(...)):
    if commissioner_key != COMMISSIONER_KEY:
        raise HTTPException(403, "Wrong commissioner key.")

    START_YEAR = 2022
    END_YEAR = 2026

    with engine.begin() as conn:
        print("[ML] update button clicked — starting update_model()")
        update_model(conn, START_YEAR, END_YEAR)

    xgb_predictor.clear_cache()

    return RedirectResponse("/model", status_code=303)

@app.get("/model/picks", response_class=HTMLResponse)
def model_picks_page(request: Request, event_short: str = Query(default="AO")):
    key = (event_short or "AO").strip().upper()
    post = load_latest_bracket_post(key)

    return templates.TemplateResponse("model_picks.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": get_events(),
        "selected_event_short": key,
        "post": post,
    })


@app.get("/draw", response_class=HTMLResponse)
def draw_page(request: Request):
    return templates.TemplateResponse("draw.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": get_events(),
    })

class DrawValidateIn(BaseModel):
    players: list[str]

@app.post("/api/draw/validate")
def api_draw_validate(payload: DrawValidateIn):
    players_in = [normalize_name(x) for x in payload.players if normalize_name(x)]
    if not players_in:
        return {"matched": [], "unmatched": []}

    with engine.begin() as conn:
        # pull a “canonical” list once for matching
        rows = conn.execute(text("""
            SELECT player_name
            FROM public.player_state
            ORDER BY elo_overall DESC
        """)).fetchall()
        canon = [r[0] for r in rows]

    canon_lc = {c.lower(): c for c in canon}

    matched = []
    unmatched = []

    for p in players_in:
        key = p.lower()
        if p.strip().upper() == "BYE" or is_placeholder(p):
            matched.append(p)   # accept BYE/Q/LL/WC
            continue
        if key in canon_lc:
            matched.append(canon_lc[key])
        else:
            # fuzzy suggestions
            sug = difflib.get_close_matches(p, canon, n=5, cutoff=0.6)
            unmatched.append({"input": p, "suggestions": sug})

    return {"matched": matched, "unmatched": unmatched}

@app.get("/api/h2h")
def api_h2h(
    player_a: str,
    player_b: str,
    surface: str = "Hard",
    tourney_name: Optional[str] = None,
):
    with engine.begin() as conn:
        a = normalize_name(player_a)
        b = normalize_name(player_b)
        
        # Same player is not accepted
        if a == b:
            raise HTTPException(
            status_code=422,
            detail=f"Player '{a}' cannot be used in both fields."
            )

        assert_player_exists(conn, a)
        assert_player_exists(conn, b)
        return predict_h2h(conn, a, b, surface.strip(), tourney_name=tourney_name)

@app.get("/api/tournament_odds")
def api_tournament_odds(
    event_short: str,
    top_k: int = 10,
    pool_n: int = 64,
):
    key = (event_short or "").strip().upper()
    if key not in EVENT_META:
        raise HTTPException(400, f"Unknown event_short. Use one of: {sorted(EVENT_META.keys())}")

    meta = EVENT_META[key]

    with engine.begin() as conn:
        odds = tournament_odds_no_draw(conn, meta["tourney_name"], meta["surface"], pool_n=int(pool_n))

    # Make pie-chart friendly: top_k + Other
    top_k = max(1, int(top_k))
    head = odds[:top_k]
    tail = odds[top_k:]

    other_p = sum(x["p"] for x in tail)
    pie = [{"label": x["player"], "p": float(x["p"])} for x in head]
    if other_p > 0:
        pie.append({"label": "Other", "p": float(other_p)})

    # normalize
    total = sum(x["p"] for x in pie) or 1.0
    for x in pie:
        x["p"] = float(x["p"] / total)

    return {
        "event_short": key,
        "tourney_name": meta["tourney_name"],
        "surface": meta["surface"],
        "pie": pie,          # PERFECT for a pie chart
        "top_detail": head,  # includes elos + record bonus + etc
    }

@app.get("/api/players")
def api_players(q: str = Query(default="", max_length=50), limit: int = 12):
    q = (q or "").strip()
    if len(q) < 2:
        return []

    with get_conn() as conn:
        rows = conn.execute(text("""
            SELECT player_name
            FROM elo_overall
            WHERE player_name ILIKE :pat
            ORDER BY elo DESC
            LIMIT :lim
        """), {
            "pat": f"%{q}%",
            "lim": int(limit)
        }).fetchall()

    return [r[0] for r in rows]

@app.get("/api/tournaments")
def api_tournaments(q: str = Query("", min_length=1), limit: int = 12):
    q = q.strip()
    if not q:
        return []

    with get_conn() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT tourney_name
            FROM matches
            WHERE tourney_name ILIKE :q
            ORDER BY tourney_name
            LIMIT :limit
        """), {"q": f"%{q}%", "limit": int(limit)}).fetchall()

    return [r[0] for r in rows]

# app.py

@app.post("/api/simulate_draw")
def api_simulate_draw(
    players_text: str = Form(...),
    surface: str = Form("Hard"),
    tourney_level: str = Form("250"),
    best_of: int = Form(3),
    tourney_date: int = Form(20250101),
    n_sims: int = Form(3000),
    seed: int = Form(42),
):
    players = [ln.strip() for ln in (players_text or "").splitlines()]

    # keep blanks OUT, but allow BYE as a real entry if user typed it
    players = [" ".join(p.split()) for p in players if " ".join(p.split())]

    if len(players) < 2:
        raise HTTPException(422, "Need at least 2 entries.")
    if (len(players) & (len(players) - 1)) != 0:
        raise HTTPException(422, "Draw size must be a power of 2 (2,4,8,16,32,64,128).")

    ctx = TourneyCtx(
        tourney_date=int(tourney_date),
        surface=surface.strip(),
        tourney_level=str(tourney_level).strip(),
        round=infer_default_round(len(players)),
        best_of=int(best_of),
    )

    det, odds, round_adv = xgb_predictor.simulate_tournament(
        players_in_order=players,
        base_ctx=ctx,
        n_sims=int(n_sims),
        seed=int(seed),
    )

    top = [{"player": p, "p": float(v)} for p, v in list(odds.items())[:25]]

    return {
        "surface": ctx.surface,
        "tourney_level": ctx.tourney_level,
        "best_of": ctx.best_of,
        "n_sims": int(n_sims),

        "champion_det": det["champion"],
        "rounds_det": det["rounds"],     # ✅ deterministic matchups per round
        "round_adv": round_adv,          # ✅ per-round survival probs

        "title_odds": top,
    }

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": get_events(),          # existing helper you already have
        "event_meta": EVENT_META,        # the dict you added earlier
    })

@app.get("/commissioner/bracket", response_class=HTMLResponse)
def commissioner_bracket_page(request: Request):
    return templates.TemplateResponse("commissioner_bracket.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": get_events(),
    })


@app.post("/commissioner/bracket/generate")
def commissioner_generate_bracket(
    request: Request,
    commissioner_key: str = Form(...),
    event_short: str = Form(...),
    surface: str = Form("Hard"),
    tourney_level: str = Form("G"),
    best_of: int = Form(5),
    tourney_date: int = Form(20260101),
    n_sims: int = Form(5000),
    seed: int = Form(42),

    draw_text: str = Form(...),
):
    if commissioner_key != COMMISSIONER_KEY:
        raise HTTPException(403, "Wrong commissioner key.")

    key = (event_short or "").strip().upper()
    meta = EVENT_META.get(key)
    if not meta:
        raise HTTPException(400, "Unknown event.")

    players = [" ".join((ln or "").split()) for ln in (draw_text or "").splitlines()]
    players = [p for p in players if p]  # keep BYE and placeholders like Q/LL

    if len(players) < 2:
        raise HTTPException(422, "Need at least 2 entries.")
    if (len(players) & (len(players) - 1)) != 0:
        raise HTTPException(422, "Draw size must be a power of 2 (2,4,8,16,32,64,128).")

    ctx = TourneyCtx(
        tourney_date=int(tourney_date),
        surface=surface.strip(),
        tourney_level=str(tourney_level).strip(),
        round=infer_default_round(len(players)),
        best_of=int(best_of),
    )

    det, odds, round_adv = xgb_predictor.simulate_tournament(
        players_in_order=players,
        base_ctx=ctx,
        n_sims=int(n_sims),
        seed=int(seed),
    )

    rounds_list = list(det["rounds"].keys())
    r1 = det["rounds"][rounds_list[0]] if rounds_list else []
    upset_spots = compute_upset_spots(r1, top_n=10)

    top = [{"player": p, "p": float(v)} for p, v in list(odds.items())[:25]]

    payload = {
        "event_short": key,
        "event_id": f"{key}{LEAGUE_YEAR}",
        "surface": ctx.surface,
        "tourney_level": ctx.tourney_level,
        "best_of": ctx.best_of,
        "tourney_date": ctx.tourney_date,
        "n_sims": int(n_sims),
        "seed": int(seed),

        "champion_det": det["champion"],
        "rounds_det": det["rounds"],
        "round_adv": round_adv,
        "title_odds": top,
        "upset_spots": upset_spots,
    }


    params_payload = {
        "event_short": key,
        "event_id": f"{key}{LEAGUE_YEAR}",
        "surface": ctx.surface,
        "tourney_level": ctx.tourney_level,
        "best_of": ctx.best_of,
        "tourney_date": ctx.tourney_date,
        "n_sims": int(n_sims),
        "seed": int(seed),
        "draw_size": len(players),
    }

    title = f"{meta['tourney_name']} — Model Bracket Prediction ({LEAGUE_YEAR})"

    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO bracket_posts(year, event_short, event_id, title, params_json, content_json)
          VALUES (:year, :ev, :eid, :title, :params, :content)
        """), {
            "year": LEAGUE_YEAR,
            "ev": key,
            "eid": f"{key}{LEAGUE_YEAR}",
            "title": title,
            "params": json.dumps(params_payload),
            "content": json.dumps(payload),
        })

    return RedirectResponse(f"/model/picks?event_short={key}", status_code=303)


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request):
    return templates.TemplateResponse("results.html", {
        "request": request,
        "year": LEAGUE_YEAR,
        "events": get_events(),
        "rounds": ALLOWED_ROUNDS,
    })


@app.post("/results")
def submit_result(
    commissioner_key: str = Form(...),
    event_id: str = Form(...),
    player: str = Form(...),
    round_reached: str = Form(...),
):
    if commissioner_key != COMMISSIONER_KEY:
        raise HTTPException(403, "Wrong commissioner key.")

    event_id = event_id.strip()
    player = normalize_name(player)
    round_reached = round_reached.strip().upper()

    if round_reached not in ALLOWED_ROUNDS:
        raise HTTPException(400, f"Invalid round. Use one of: {ALLOWED_ROUNDS}")

    with engine.begin() as conn:
        assert_player_exists(conn, player)
        conn.execute(text("""
          INSERT INTO results(event_id, player_name, round_reached)
          VALUES (:e, :pl, :r)
          ON CONFLICT (event_id, player_name)
          DO UPDATE SET round_reached = excluded.round_reached;
        """), {"e": event_id, "pl": player, "r": round_reached})

@app.get("/debug/sinner")
def debug_sinner():
    ctx = TourneyCtx(
        tourney_date=20260101,
        surface="Hard",
        tourney_level="G",
        round="R128",
        best_of=5,
    )

    # Use your existing predictor instance
    out1 = xgb_predictor.debug_match("Jannik Sinner", "Alex Michelsen", ctx)
    out2 = xgb_predictor.debug_match("Alex Michelsen", "Jannik Sinner", ctx)

    return {"A_vs_B": out1, "B_vs_A": out2, "sum": out1["p_a"] + out2["p_a"]}
