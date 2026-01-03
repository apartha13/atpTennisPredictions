import os
from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from typing import Optional
from urllib.parse import urlencode 
from fastapi.staticfiles import StaticFiles

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


def get_events():
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, short_id, name, level FROM events WHERE year=:y ORDER BY id;"),
            {"y": LEAGUE_YEAR}
        ).fetchall()
    # list of dicts for templates
    return [{"id": r[0], "short_id": r[1], "name": r[2], "level": r[3]} for r in rows]


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
    player = player.strip()

    if not (person and event_id and player):
        raise HTTPException(400, "Missing fields.")

    with engine.begin() as conn:
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
    player = player.strip()
    round_reached = round_reached.strip().upper()

    if round_reached not in ALLOWED_ROUNDS:
        raise HTTPException(400, f"Invalid round. Use one of: {ALLOWED_ROUNDS}")

    with engine.begin() as conn:
        conn.execute(text("""
          INSERT INTO results(event_id, player_name, round_reached)
          VALUES (:e, :pl, :r)
          ON CONFLICT (event_id, player_name)
          DO UPDATE SET round_reached = excluded.round_reached;
        """), {"e": event_id, "pl": player, "r": round_reached})

    return RedirectResponse("/results", status_code=303)
