# 🎾 ATP Predictions — Family Tennis League

A web application for running a private ATP tennis prediction league.  
Family and friends make **one pick per event**, earn points based on how far their player advances, and track standings throughout the season.

Built with **FastAPI**, **Supabase (Postgres)**, and a clean, modern UI.

---

## ✨ Features

- 🏆 **Live leaderboard** with automatic point calculation
- 🎾 **14 events**:
  - 4 Grand Slams  
  - 9 ATP Masters 1000  
  - Nitto ATP Finals
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
