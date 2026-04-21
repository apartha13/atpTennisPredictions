# 🎾 ATP Predictions — Tennis League

A web application for running a private ATP tennis prediction league.  
Family and friends make **one pick per event**, earn points based on how far their player advances, and track standings throughout the season.

Built with **FastAPI**, **Supabase (Postgres)**, and a clean, modern UI.

---

## ✨ Features

- 🏆 **Live leaderboard** with automatic point calculation
- 🎾 **13 events**:
  - 4 Grand Slams  
  - 9 ATP Masters 1000  
- 👤 **One pick per person per event**
- 🔁 Picks can be updated (overwrite previous pick)
- 🔒 **Commissioner-only results entry**
- 📊 **Per-event breakdown page**
- 🥇 Gold / 🥈 Silver / 🥉 Bronze medals for top 3
- ☁️ Cloud-hosted database (Supabase)
- 🌍 Publicly accessible website (Render)

---

## 🧠 Scoring System

Points are awarded based on the **round reached** by the selected player.

Example scoring (configurable):

| Round | Points |
|------|--------|
| Winner (W) | 100 |
| Final (F) | 60 |
| Semi-final (SF) | 40 |
| Round Robin (RR – ATP Finals) | 20 |
| Quarterfinal (QF) | 25 |
| Round of 16 (R16) | 15 |

---

## 🏗️ Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** Jinja2 templates + custom CSS
- **Database:** Supabase (PostgreSQL)
- **ORM / SQL:** SQLAlchemy
- **Hosting:** Render
- **Server:** Uvicorn (dev), Gunicorn (production)

---

## 🚀 Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/atp-predictions.git
cd atp-predictions
```
### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Set environment variables
```bash
export DATABASE_URL="postgresql+psycopg2://..."
export COMMISSIONER_KEY="your-secret-key"
export LEAGUE_YEAR="2026"
```

### 5️⃣ Run the server
```bash
uvicorn app:app --reload
```
Visit: http://127.0.0.1:8000

### Quick Start (Copy/Paste)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL="postgresql+psycopg2://..."
export COMMISSIONER_KEY="your-secret-key"
export LEAGUE_YEAR="2026"

uvicorn app:app --reload
```

---

## What Each Folder Means

### Root
- `app.py`: FastAPI entrypoint, routes, page rendering, app-level DB setup/seed.
- `requirements.txt`: Python dependencies.
- `README.md`: setup, architecture, and workflow docs.

### `pipeline/`
- Home for data ingestion and model-serving data prep.
- `pipeline/ml_update.py`:
  - Pulls ATP rows from source CSV files.
  - Upserts match history into DB.
  - Recomputes Elo tables.
  - Rebuilds H2H and event-record tables.
  - Exposes helper prediction functions used by app APIs.

### `runtime/`
- Home for prediction runtime logic used by live app endpoints.
- `runtime/tennis_model.py`:
  - Loads XGBoost model artifacts.
  - Builds live match features from `player_state`.
  - Predicts head-to-head probabilities.
  - Simulates full draws using Monte Carlo.
- `runtime/model_runtime.py`:
  - Lightweight legacy wrapper retained for reference.

### `scripts/`
- Standalone scripts for training/backfills/manual maintenance.
- `scripts/tennis_XGBmodel.py`: model training and artifact generation.
- `scripts/player_state_build.py`: standalone player_state table rebuild.
- `scripts/backfill_ranks.py`: rank backfill helper for match rows.
- `scripts/predict_features.py`: legacy feature-builder reference.

### `artifacts/`
- Model outputs consumed by runtime:
  - `xgb_model.json`
  - `feature_columns.json`

### `templates/` and `static/`
- `templates/`: Jinja page templates.
- `static/`: frontend assets (favicon, css/js if added).

---

## What Each Script Does

### App Runtime
- `app.py`
  - Starts FastAPI app.
  - Creates/ensures required league and model tables.
  - Serves league pages (`/`, `/picks`, `/results`, `/breakdown`, `/model`, etc).
  - Serves model APIs (`/api/h2h`, `/api/tournament_odds`, draw simulation, player search).

- `runtime/tennis_model.py`
  - Main runtime predictor class: `TennisPredictor`.
  - Loads XGBoost artifact + feature order.
  - Produces matchup probabilities from current DB player state.
  - Simulates full tournament brackets and round-advancement odds.

### Data + Ratings Pipeline
- `pipeline/ml_update.py`
  - `backfill_years`: ingest match data.
  - `recompute_elos`: rebuild overall and surface Elo tables.
  - `backfill_h2h`, `backfill_h2h_event`, `backfill_event_record`: rebuild historical matchup context tables.
  - `update_model`: orchestration entrypoint called from app Model Update flow.
  - `predict_h2h`, `tournament_odds_no_draw`: lightweight probability APIs for app endpoints.

### Offline/Manual Scripts
- `scripts/tennis_XGBmodel.py`
  - Builds training datasets/features.
  - Trains XGBoost model.
  - Evaluates on validation/test splits.
  - Saves model artifacts to `artifacts/`.

- `scripts/player_state_build.py`
  - Computes player-level state features from historical matches.
  - Upserts to `public.player_state`.

- `scripts/backfill_ranks.py`
  - Targeted rank backfill into `matches` for missing winner/loser ranks.

- `scripts/predict_features.py`
  - Legacy feature function reference (kept for experimentation/history).

---

## Cool Machine Learning Things This App Does

- Surface-aware tennis intelligence:
  - Blends surface-specific and overall strength so Hard/Clay/Grass context matters.

- Dynamic Elo ecosystem:
  - Maintains both overall Elo and per-surface Elo.
  - Uses those for both matchup and tournament-level predictions.

- Historical matchup context:
  - Uses H2H by surface and all-surfaces.
  - Uses event-specific pair history and player event records for tournament context.

- Rich feature engineering for matchups:
  - Rank gaps, Elo gaps, surface match experience, form, and rolling serve profile deltas.

- Bracket simulation engine:
  - Deterministic path (most likely winners by matchup).
  - Monte Carlo simulations for title odds and round-advancement probabilities.

- Artifact-based serving:
  - Offline training outputs (`xgb_model.json`, `feature_columns.json`) are loaded in production runtime.

---

## Common Workflows

### 1) Run the App
```bash
source .venv/bin/activate
export DATABASE_URL="postgresql+psycopg2://..."
export COMMISSIONER_KEY="your-secret-key"
export LEAGUE_YEAR="2026"
uvicorn app:app --reload
```

### 2) Refresh Data + Model Tables (from app)
- Open `/model` page.
- Use the Model Update action (commissioner key required).
- This triggers `pipeline/ml_update.py` update flow.

### 3) Retrain the XGBoost Model Artifacts
```bash
source .venv/bin/activate
export DATABASE_URL="postgresql+psycopg2://..."
python scripts/tennis_XGBmodel.py
```

### 4) Rebuild Player State (manual)
```bash
source .venv/bin/activate
export DATABASE_URL="postgresql+psycopg2://..."
python scripts/player_state_build.py
```

---

## High-Level Data Flow

1. Raw ATP CSV data is ingested into `matches`.
2. Pipeline recomputes Elo + H2H + event context tables.
3. Runtime predictor pulls `player_state` + model artifacts.
4. App endpoints return matchup probabilities, draw simulations, and title odds.
5. League pages combine picks/results with model context for standings and analysis.

## Pipeline Ownership

To avoid overlap, this repository uses the following ownership model:

- Data ingestion/backfill and Elo/history tables: `pipeline/ml_update.py`
- Model training and artifact generation: `scripts/tennis_XGBmodel.py`
- Runtime match prediction and draw simulation: `runtime/tennis_model.py`
- Web API/app orchestration: `app.py`

Legacy/manual scripts retained for reference:

- `scripts/predict_features.py` (legacy feature builder)
- `scripts/player_state_build.py` (standalone player_state builder)
- `scripts/backfill_ranks.py` (standalone rank backfill utility)
- `runtime/model_runtime.py` (lightweight runtime wrapper not used by app routes)

## Project Structure
```arduino
.
├── app.py
├── pipeline/
│   └── ml_update.py
├── runtime/
│   ├── tennis_model.py
│   └── model_runtime.py
├── scripts/
│   ├── tennis_XGBmodel.py
│   ├── player_state_build.py
│   ├── backfill_ranks.py
│   └── predict_features.py
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── picks.html
│   ├── results.html
│   └── breakdown.html
├── static/
│   └── (optional CSS / assets)
└── README.md
```
