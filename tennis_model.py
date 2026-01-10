# tennis_model.py
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

import math

ROUND_NUM = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}

# EXACT columns from your training (fallback only)
MODEL_COLS = [
    "rank_diff", "round_num", "best_of",
    "elo_diff_surface", "elo_diff_overall", "surface_match_count_diff",
    "match_importance",
    "form5_diff", "form10_diff",
    "roll_ace_rate_diff", "roll_df_rate_diff", "roll_first_in_diff",
    "roll_first_won_diff", "roll_second_won_diff", "roll_bp_saved_diff",
    "surface_Clay", "surface_Grass", "surface_Hard", "surface_Unknown",
    "level_250", "level_500", "level_A", "level_D", "level_F", "level_G", "level_M",
]

LEVELS = ["250", "500", "A", "D", "F", "G", "M"]
SURFACES = ["Clay", "Grass", "Hard", "Unknown"]

PLACEHOLDER_RE = re.compile(r"^(Q(\d+)?|QUALIFIER(\s*\d+)?|LL(\d+)?|LUCKY\s*LOSER(\s*\d+)?|WC(\d+)?|WILD\s*CARD(\s*\d+)?)$",
                            re.IGNORECASE)

def _missing_keys(d: dict, keys: list[str]) -> list[str]:
    miss = []
    for k in keys:
        if k not in d or d.get(k) is None:
            miss.append(k)
    return miss

STAT_KEYS = [
    "form5","form10",
    "roll_ace_rate","roll_df_rate","roll_first_in",
    "roll_first_won","roll_second_won","roll_bp_saved",
    "elo_overall","elo_hard","elo_clay","elo_grass",
    "matches_hard","matches_clay","matches_grass",
    "rank",
]

def is_bye(name: str) -> bool:
    return (name or "").strip().upper() == "BYE"

def is_placeholder(name: str) -> bool:
    s = (name or "").strip()
    if not s:
        return False
    return bool(PLACEHOLDER_RE.match(s))


NAME_TAG_RE = re.compile(r"\s*\([^)]*\)\s*$")  # strips "(1)", "(WC)", "(LL)", "(Q)", etc.

def norm_name(name: str) -> str:
    s = " ".join((name or "").strip().split())
    s = NAME_TAG_RE.sub("", s).strip()
    return s

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    else:
        ez = math.exp(z)
        return ez / (1 + ez)

def apply_temperature(p: float, temperature: float) -> float:
    """
    temperature > 1 => more random (pulls probs toward 0.5)
    temperature < 1 => more confident (pushes probs away from 0.5)
    """
    t = max(1e-3, float(temperature))
    return sigmoid(logit(p) / t)

def apply_upset_bias(p: float, upset_bias: float) -> float:
    """
    upset_bias > 0 => more upsets (pull toward 0.5 and slightly beyond)
    upset_bias < 0 => more favorites (push away from 0.5)
    Range suggestion: [-1.0, +1.0]
    """
    b = float(upset_bias)
    # Simple, stable adjustment in logit space
    return sigmoid(logit(p) - b)

@dataclass(frozen=True)
class TourneyCtx:
    tourney_date: int
    surface: str
    tourney_level: str
    round: str
    best_of: int = 3


class TennisPredictor:
    """
    Runtime predictor:
      - Loads trained XGBoost model + feature column order
      - Builds features for A vs B using player_state
      - Simulates bracket:
          * deterministic bracket (p>=0.5)
          * Monte Carlo champ odds
          * Monte Carlo per-round advancement odds (reach/advance)
    """

    def __init__(
        self,
        engine,
        model_path: str = "artifacts/xgb_model.json",
        cols_path: str = "artifacts/feature_columns.json",
    ):
        self.engine = engine

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # Prefer saved training column order
        try:
            with open(cols_path, "r") as f:
                self.feature_cols = json.load(f)
        except Exception:
            self.feature_cols = MODEL_COLS
        
        missing = [c for c in ["rank_diff", "elo_diff_overall", "elo_diff_surface", "round_num"] if c not in self.feature_cols]
        if missing:
            print("[WARN] Missing expected features in feature_cols:", missing)
        
        self._calibrate_proba_direction()


    # ---------- normalization helpers ----------

    def _norm_surface(self, s: str) -> str:
        s = (s or "Unknown").strip()
        return s if s in SURFACES else "Unknown"

    def _norm_level(self, lvl: str) -> str:
        lvl = (lvl or "250").strip()
        return lvl if lvl in LEVELS else "250"

    def _surface_elo_field(self, surface: str) -> str:
        if surface == "Clay":
            return "elo_clay"
        if surface == "Grass":
            return "elo_grass"
        # Unknown uses hard-ish baseline
        return "elo_hard"

    def _surface_matches_field(self, surface: str) -> str:
        if surface == "Clay":
            return "matches_clay"
        if surface == "Grass":
            return "matches_grass"
        return "matches_hard"

    # ---------- DB load ----------

    def load_player_state_bulk(self, players: List[str]) -> Dict[str, dict]:
        players = sorted(set([p.strip() for p in players if p and p.strip()]))
        if not players:
            return {}

        q = text("""
            SELECT *
            FROM public.player_state
            WHERE player_name = ANY(:players)
        """)

        with self.engine.begin() as conn:
            rows = conn.execute(q, {"players": players}).mappings().all()

        return {r["player_name"]: dict(r) for r in rows}

    # ---------- feature building ----------

    @staticmethod
    def _g(d: dict, k: str, default: float) -> float:
        v = d.get(k, default)
        return float(default) if v is None else float(v)

    def build_feature_row(self, a: str, b: str, ctx: TourneyCtx, ps: Dict[str, dict]) -> pd.DataFrame:    
        surface = self._norm_surface(ctx.surface)
        level = self._norm_level(ctx.tourney_level)
        round_num = float(ROUND_NUM.get(ctx.round, 0))

        da = ps.get(a, {}) or {}
        db = ps.get(b, {}) or {}

        # --- Elo overall ---
        a_over = float(da.get("elo_overall", 1500.0))
        b_over = float(db.get("elo_overall", 1500.0))
        elo_diff_overall = a_over - b_over

        # --- Elo surface ---
        surf_field = self._surface_elo_field(surface)
        a_surf = float(da.get(surf_field, 1500.0))
        b_surf = float(db.get(surf_field, 1500.0))
        elo_diff_surface = a_surf - b_surf

        # --- surface match count ---
        match_field = self._surface_matches_field(surface)
        a_cnt = float(da.get(match_field, 0.0))
        b_cnt = float(db.get(match_field, 0.0))
        surface_match_count_diff = a_cnt - b_cnt

        # --- Rank diff (your training convention) ---
        # rank_diff = rank(B) - rank(A)
        ra = da.get("rank", 300) or 300
        rb = db.get("rank", 300) or 300
        rank_diff = float(rb - ra) 

        # --- match_importance ---
        # Keep default=1.0 unless you have the exact formula used in training
        mia = float(da.get("match_importance_override", 1.0) or 1.0)
        mib = float(db.get("match_importance_override", 1.0) or 1.0)
        match_importance = 0.5 * (mia + mib)
        if match_importance <= 0:
            match_importance = 1.0

        row = {
            "rank_diff": rank_diff,
            "round_num": round_num,
            "best_of": float(ctx.best_of),

            "elo_diff_surface": float(elo_diff_surface),
            "elo_diff_overall": float(elo_diff_overall),
            "surface_match_count_diff": float(surface_match_count_diff),

            "match_importance": float(match_importance),

            "form5_diff": self._g(da, "form5", 0.5) - self._g(db, "form5", 0.5),
            "form10_diff": self._g(da, "form10", 0.5) - self._g(db, "form10", 0.5),

            "roll_ace_rate_diff": self._g(da, "roll_ace_rate", 0.0) - self._g(db, "roll_ace_rate", 0.0),
            "roll_df_rate_diff": self._g(da, "roll_df_rate", 0.0) - self._g(db, "roll_df_rate", 0.0),
            "roll_first_in_diff": self._g(da, "roll_first_in", 0.0) - self._g(db, "roll_first_in", 0.0),
            "roll_first_won_diff": self._g(da, "roll_first_won", 0.0) - self._g(db, "roll_first_won", 0.0),
            "roll_second_won_diff": self._g(da, "roll_second_won", 0.0) - self._g(db, "roll_second_won", 0.0),
            "roll_bp_saved_diff": self._g(da, "roll_bp_saved", 0.0) - self._g(db, "roll_bp_saved", 0.0),
        }

        # one-hot columns exactly
        for s in SURFACES:
            row[f"surface_{s}"] = 1.0 if surface == s else 0.0
        for l in LEVELS:
            row[f"level_{l}"] = 1.0 if level == l else 0.0

        return pd.DataFrame([row])

    def align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.feature_cols, fill_value=0.0)

    def predict_match_proba(self, a: str, b: str, ctx: TourneyCtx, ps: Dict[str, dict]) -> float:
        X = self.align_columns(self.build_feature_row(a, b, ctx, ps))
        p = float(self.model.predict_proba(X)[0, 1])
        return p if self.proba_is_a_wins else (1.0 - p)

    def debug_match(self, a: str, b: str, ctx: TourneyCtx, ps: dict | None = None) -> dict:
        if ps is None:
            ps = self.load_player_state_bulk([a, b])

        X_raw = self.build_feature_row(a, b, ctx, ps)
        key_cols = [
        "surface_match_count_diff",
        "form5_diff", "form10_diff",
        "roll_ace_rate_diff", "roll_df_rate_diff", "roll_first_in_diff",
        "roll_first_won_diff", "roll_second_won_diff", "roll_bp_saved_diff",
        "match_importance",
        ]

        vals = {c: float(X_raw[c].iloc[0]) for c in key_cols if c in X_raw.columns}

        nan_cols = [c for c in X_raw.columns if pd.isna(X_raw[c].iloc[0])]

        X = self.align_columns(X_raw)

        p_ab = self._p_a_wins_raw(a, b, ctx, ps)   # P(a wins) from (a,b)
        p_ba = self._p_a_wins_raw(b, a, ctx, ps)   # P(b wins) from (b,a)
        p_sym = 0.5 * (p_ab + (1.0 - p_ba))

        p_raw = float(self.model.predict_proba(X)[0, 1])
        p_a = p_raw if self.proba_is_a_wins else (1.0 - p_raw)
        da = ps.get(a, {})
        db = ps.get(b, {})

        return {
            "a": a, "b": b, "p_a": p_a, "p_raw": p_raw, "p_ab": p_ab, "p_ba": p_ba, "p_sym": p_sym,
            "a_rank": da.get("rank"), "b_rank": db.get("rank"),
            "a_elo_overall": da.get("elo_overall"), "b_elo_overall": db.get("elo_overall"),
            "rank_diff": float(X_raw["rank_diff"].iloc[0]),
            "elo_diff_overall": float(X_raw["elo_diff_overall"].iloc[0]),
            "elo_diff_surface": float(X_raw["elo_diff_surface"].iloc[0]),
            "round_num": float(X_raw["round_num"].iloc[0]),
            "a_missing": _missing_keys(da, STAT_KEYS),
            "b_missing": _missing_keys(db, STAT_KEYS),
             "extra": vals,
            "nan_cols": nan_cols,
    }

    def _calibrate_proba_direction(self) -> None:
        """
        Decide whether predict_proba(X)[:,1] corresponds to P(A wins) or P(B wins).

        We test a "strong favorite vs weak" matchup by Elo (overall), and check if
        proba[:,1] increases when A is stronger. If it decreases, it's flipped.

        This runs once at startup.
        """
        # If we can't calibrate, fall back to assuming class-1 is A-wins.
        self.proba_is_a_wins = True

        try:
            # Pull two players with very different elo_overall from player_state
            with self.engine.begin() as conn:
                rows = conn.execute(text("""
                    SELECT player_name, elo_overall
                    FROM public.player_state
                    WHERE elo_overall IS NOT NULL
                    ORDER BY elo_overall DESC
                    LIMIT 1
                """)).fetchall()

                rows2 = conn.execute(text("""
                    SELECT player_name, elo_overall
                    FROM public.player_state
                    WHERE elo_overall IS NOT NULL
                    ORDER BY elo_overall ASC
                    LIMIT 1
                """)).fetchall()

            if not rows or not rows2:
                print("[WARN] Could not calibrate proba direction (no player_state elo).")
                return

            strong = rows[0][0]
            weak = rows2[0][0]

            ctx = TourneyCtx(tourney_date=20250101, surface="Hard", tourney_level="G", round="R128", best_of=5)

            ps = self.load_player_state_bulk([strong, weak])

            X_sw = self.align_columns(self.build_feature_row(strong, weak, ctx, ps))
            X_ws = self.align_columns(self.build_feature_row(weak, strong, ctx, ps))

            p_sw = float(self.model.predict_proba(X_sw)[0, 1])  # "class 1" prob when strong is A
            p_ws = float(self.model.predict_proba(X_ws)[0, 1])  # "class 1" prob when weak is A

            # interpret class-1 as "A wins" only if strong-as-A looks more likely than weak-as-A
            self.proba_is_a_wins = (p_sw > p_ws)

            # optional: print warning if it's ambiguous
            if abs(p_sw - p_ws) < 0.05:
                print("[CAL] weak calibration signal; difference small:", p_sw, p_ws)
            else:
                self.proba_is_a_wins = True
                print(f"[CAL] proba[:,1] appears to be P(A wins). (p_strong_as_A={p_sw:.3f}, p_weak_as_A={p_ws:.3f})")

        except Exception as e:
            print("[WARN] proba calibration failed:", repr(e))
            self.proba_is_a_wins = True

    def _p_a_wins_raw(self, a: str, b: str, ctx: TourneyCtx, ps: Dict[str, dict]) -> float:
        X = self.align_columns(self.build_feature_row(a, b, ctx, ps))
        p = float(self.model.predict_proba(X)[0, 1])
        return p if self.proba_is_a_wins else (1.0 - p)
    
    def predict_match(self, a: str, b: str, ctx: TourneyCtx, ps: Dict[str, dict] | None = None) -> float:
        if ps is None:
            ps = self.load_player_state_bulk([a, b])

        p_ab = self._p_a_wins_raw(a, b, ctx, ps)   # P(a wins) from (a,b)
        p_ba = self._p_a_wins_raw(b, a, ctx, ps)   # P(b wins) from (b,a)

        p_sym = 0.5 * (p_ab + (1.0 - p_ba))
        return float(clamp(p_sym, 1e-6, 1 - 1e-6))

    def _infer_rounds(self, n: int) -> List[str]:
        if n == 128: return ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
        if n == 64:  return ["R64", "R32", "R16", "QF", "SF", "F"]
        if n == 32:  return ["R32", "R16", "QF", "SF", "F"]
        if n == 16:  return ["R16", "QF", "SF", "F"]
        if n == 8:   return ["QF", "SF", "F"]
        if n == 4:   return ["SF", "F"]
        if n == 2:   return ["F"]
        raise ValueError("Draw size must be a power of 2 (2,4,8,16,32,64,128).")

    def simulate_tournament(
        self,
        players_in_order: List[str],
        base_ctx: TourneyCtx,
        n_sims: int = 3000,
        seed: int = 42,
    ):
        # normalize names but keep placeholders
        players = [norm_name(p) for p in players_in_order if norm_name(p)]
        n = len(players)
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError("Draw must be a power of 2 (2,4,8,16,32,64,128).")

        rounds = self._infer_rounds(n)
        rng = np.random.default_rng(seed)

        # Bulk-load once for everyone (unknowns will just fall back to defaults in build_feature_row)
        ps_all = self.load_player_state_bulk([p for p in players if (not is_bye(p)) and (not is_placeholder(p))])

        # -------------------------
        # Deterministic bracket (with matchups)
        # -------------------------
        det = {"rounds": {}, "champion": None}
        cur = players[:]

        for rnd in rounds:
            ctx = TourneyCtx(**{**base_ctx.__dict__, "round": rnd})
            matches = []
            nxt = []

            for i in range(0, len(cur), 2):
                a, b = cur[i], cur[i + 1]

                # BYE rules
                if is_bye(a) and is_bye(b):
                    # should never happen, but don’t crash
                    winner = "BYE"
                    p = 0.5
                elif is_bye(a):
                    winner = b
                    p = 0.0
                elif is_bye(b):
                    winner = a
                    p = 1.0
                else:
                    p = self.predict_match(a, b, ctx, ps=ps_all)
                    winner = a if p >= 0.5 else b
                matches.append({
                    "a": a,
                    "b": b,
                    "p_a": float(p),      # P(A wins)
                    "winner": winner,
                })
                nxt.append(winner)

            det["rounds"][rnd] = matches
            cur = nxt

        det["champion"] = cur[0]

        # -------------------------
        # Monte Carlo title odds + (optional) per-round advancement
        # -------------------------
        wins: Dict[str, int] = {}
        adv: Dict[str, Dict[str, int]] = {rnd: {} for rnd in rounds}  # counts: who survives each round

        for _ in range(int(n_sims)):
            cur = players[:]
            for rnd in rounds:
                ctx = TourneyCtx(**{**base_ctx.__dict__, "round": rnd})
                nxt = []
                for i in range(0, len(cur), 2):
                    a, b = cur[i], cur[i + 1]

                    if is_bye(a) and is_bye(b):
                        winner = "BYE"
                    elif is_bye(a):
                        winner = b
                    elif is_bye(b):
                        winner = a
                    else:
                        p = self.predict_match(a, b, ctx, ps=ps_all)
                        winner = a if (rng.random() < p) else b

                    nxt.append(winner)

                # record who advanced past this round
                for w in nxt:
                    adv[rnd][w] = adv[rnd].get(w, 0) + 1

                cur = nxt

            champ = cur[0]
            wins[champ] = wins.get(champ, 0) + 1

        odds = {p: c / n_sims for p, c in sorted(wins.items(), key=lambda x: -x[1])}

        # convert adv to probabilities
        round_adv = {
            rnd: {p: c / n_sims for p, c in sorted(m.items(), key=lambda x: -x[1])}
            for rnd, m in adv.items()
        }

        return det, odds, round_adv
