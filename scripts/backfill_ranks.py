import os
# Legacy standalone utility.
# Rank backfill responsibilities are now part of ml_update.py update flow.
# Keep this script only for targeted/manual repair runs.

from sqlalchemy import create_engine, text
from pipeline.ml_update import fetch_year_rows, make_match_key, parse_int_or_none

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)

def backfill_ranks_for_year(year: int, chunk_size: int = 5000) -> int:
    rows = fetch_year_rows(year)

    # Build payload once
    payload = []
    for r in rows:
        mk = make_match_key(r)
        wr = parse_int_or_none(r.get("winner_rank"))
        lr = parse_int_or_none(r.get("loser_rank"))
        if wr is None and lr is None:
            continue
        payload.append({"mk": mk, "wr": wr, "lr": lr})

    if not payload:
        return 0

    updated_total = 0

    with engine.begin() as conn:
        # Temp staging table lives for this connection
        conn.execute(text("CREATE TEMP TABLE tmp_ranks (mk text primary key, wr int, lr int) ON COMMIT DROP;"))

        # Insert into temp table in chunks (executemany)
        for i in range(0, len(payload), chunk_size):
            batch = payload[i : i + chunk_size]
            conn.execute(
                text("INSERT INTO tmp_ranks (mk, wr, lr) VALUES (:mk, :wr, :lr) ON CONFLICT (mk) DO UPDATE SET wr=EXCLUDED.wr, lr=EXCLUDED.lr;"),
                batch,
            )

        # Single set-based update
        res = conn.execute(text("""
            UPDATE public.matches m
            SET
              winner_rank = COALESCE(m.winner_rank, t.wr),
              loser_rank  = COALESCE(m.loser_rank,  t.lr)
            FROM tmp_ranks t
            WHERE m.match_key = t.mk
              AND (
                (m.winner_rank IS NULL AND t.wr IS NOT NULL) OR
                (m.loser_rank  IS NULL AND t.lr IS NOT NULL)
              );
        """))

        updated_total = int(res.rowcount or 0)

    return updated_total

def main():
    total = 0
    for y in range(2022, 2026):
        n = backfill_ranks_for_year(y)
        total += n
        print(y, "updated", n)
    print("TOTAL updated:", total)

if __name__ == "__main__":
    main()
