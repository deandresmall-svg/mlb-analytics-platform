from __future__ import annotations

from math import isfinite

import pandas as pd


def _number(row: pd.Series, key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if isfinite(value) else default


def _scale(value: float, low: float, high: float, invert: bool = False) -> float:
    if high <= low:
        return 50.0
    score = 100.0 * (value - low) / (high - low)
    score = max(0.0, min(100.0, score))
    return 100.0 - score if invert else score


def _weighted(parts: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return 50.0
    return sum(score * weight for score, weight in parts) / total_weight


def score_label(score: object) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "Unavailable"
    if value >= 85:
        return "Elite"
    if value >= 70:
        return "Strong"
    if value >= 55:
        return "Favorable"
    if value >= 45:
        return "Neutral"
    if value >= 30:
        return "Weak"
    return "Poor"


def _data_confidence(base_games: float, statcast_games: float) -> str:
    if base_games >= 20 and statcast_games >= 10:
        return "High"
    if base_games >= 10 and statcast_games >= 5:
        return "Medium"
    return "Low"


def hit_score(row: pd.Series) -> dict[str, object]:
    recent = _weighted(
        [
            (_scale(_number(row, "batting_avg_30", 0.245), 0.190, 0.340), 0.35),
            (_scale(_number(row, "hit_rate_10", 0.55), 0.35, 0.85), 0.35),
            (_scale(_number(row, "balls_in_play_rate_30", 0.65), 0.50, 0.82), 0.30),
        ]
    )
    opportunity = _weighted(
        [
            (_scale(_number(row, "projected_pa", 4.0), 3.2, 4.9), 0.65),
            (_scale(_number(row, "batting_order", 5.0), 1.0, 9.0, invert=True), 0.35),
        ]
    )
    contact_quality = _weighted(
        [
            (_scale(_number(row, "sc_batter_xba_30", 0.245), 0.190, 0.340), 0.35),
            (_scale(_number(row, "sc_batter_hard_hit_rate_30", 0.38), 0.24, 0.58), 0.25),
            (_scale(_number(row, "sc_batter_sweet_spot_rate_30", 0.33), 0.20, 0.47), 0.20),
            (_scale(_number(row, "sc_batter_whiff_rate_30", 0.245), 0.15, 0.36, invert=True), 0.20),
        ]
    )
    matchup = _weighted(
        [
            (_scale(_number(row, "sc_opp_xba_allowed_10", 0.245), 0.195, 0.320), 0.35),
            (_scale(_number(row, "sc_opp_hard_hit_rate_allowed_10", 0.38), 0.25, 0.52), 0.25),
            (_scale(_number(row, "sc_opp_whiff_rate_10", 0.245), 0.15, 0.36, invert=True), 0.25),
            (_scale(_number(row, "opponent_sp_whip_5", 1.30), 0.95, 1.65), 0.15),
        ]
    )
    overall = _weighted(
        [(recent, 0.30), (opportunity, 0.20), (contact_quality, 0.30), (matchup, 0.20)]
    )
    score = round(overall, 1)
    return {
        "hit_score": score,
        "hit_score_label": score_label(score),
        "hit_score_confidence": _data_confidence(
            _number(row, "games_prior", 0),
            _number(row, "sc_batter_games_30", 0),
        ),
        "hit_score_recent": round(recent, 1),
        "hit_score_opportunity": round(opportunity, 1),
        "hit_score_contact": round(contact_quality, 1),
        "hit_score_matchup": round(matchup, 1),
    }


def home_run_score(row: pd.Series) -> dict[str, object]:
    production = _weighted(
        [
            (_scale(_number(row, "iso_30", 0.160), 0.070, 0.320), 0.35),
            (_scale(_number(row, "home_runs_per_pa_30", 0.030), 0.005, 0.085), 0.35),
            (_scale(_number(row, "slugging_30", 0.410), 0.300, 0.650), 0.30),
        ]
    )
    contact_quality = _weighted(
        [
            (_scale(_number(row, "sc_batter_barrel_rate_30", 0.07), 0.015, 0.190), 0.30),
            (_scale(_number(row, "sc_batter_hard_hit_rate_30", 0.38), 0.24, 0.60), 0.20),
            (_scale(_number(row, "sc_batter_avg_ev_30", 88.0), 84.0, 94.0), 0.15),
            (_scale(_number(row, "sc_batter_max_ev_30", 103.0), 98.0, 116.0), 0.15),
            (_scale(_number(row, "sc_batter_xslg_30", 0.410), 0.260, 0.700), 0.20),
        ]
    )
    matchup = _weighted(
        [
            (_scale(_number(row, "sc_opp_barrel_rate_allowed_10", 0.07), 0.02, 0.16), 0.35),
            (_scale(_number(row, "sc_opp_xslg_allowed_10", 0.410), 0.280, 0.610), 0.30),
            (_scale(_number(row, "sc_opp_avg_ev_allowed_10", 88.0), 84.5, 93.0), 0.20),
            (_scale(_number(row, "opponent_sp_hr_per_9_5", 1.20), 0.50, 2.10), 0.15),
        ]
    )
    opportunity = _weighted(
        [
            (_scale(_number(row, "projected_pa", 4.0), 3.2, 4.9), 0.60),
            (_scale(_number(row, "batting_order", 5.0), 1.0, 9.0, invert=True), 0.40),
        ]
    )
    overall = _weighted(
        [(production, 0.25), (contact_quality, 0.40), (matchup, 0.25), (opportunity, 0.10)]
    )
    score = round(overall, 1)
    return {
        "home_run_score": score,
        "home_run_score_label": score_label(score),
        "home_run_score_confidence": _data_confidence(
            _number(row, "games_prior", 0),
            _number(row, "sc_batter_games_30", 0),
        ),
        "home_run_score_production": round(production, 1),
        "home_run_score_contact": round(contact_quality, 1),
        "home_run_score_matchup": round(matchup, 1),
        "home_run_score_opportunity": round(opportunity, 1),
    }


def pitcher_k_score(row: pd.Series) -> dict[str, object]:
    strikeout_skill = _weighted(
        [
            (_scale(_number(row, "pitcher_k_per_9_10", 8.5), 5.0, 13.5), 0.40),
            (_scale(_number(row, "pitcher_k_rate_10", 0.23), 0.14, 0.36), 0.35),
            (_scale(_number(row, "pitcher_k_avg_5", 5.0), 2.5, 9.5), 0.25),
        ]
    )
    swing_miss = _weighted(
        [
            (_scale(_number(row, "sc_pitcher_whiff_rate_10", 0.245), 0.15, 0.41), 0.40),
            (_scale(_number(row, "sc_pitcher_chase_rate_10", 0.285), 0.20, 0.41), 0.30),
            (_scale(_number(row, "sc_pitcher_csw_rate_10", 0.275), 0.21, 0.36), 0.30),
        ]
    )
    workload = _weighted(
        [
            (_scale(_number(row, "pitcher_pitches_5", 85.0), 65.0, 105.0), 0.60),
            (_scale(_number(row, "pitcher_ip_5", 5.0), 3.5, 7.0), 0.40),
        ]
    )
    command = _weighted(
        [
            (_scale(_number(row, "pitcher_k_minus_bb_rate_10", 0.145), 0.04, 0.28), 0.60),
            (_scale(_number(row, "sc_pitcher_zone_rate_10", 0.49), 0.38, 0.58), 0.40),
        ]
    )
    overall = _weighted(
        [(strikeout_skill, 0.35), (swing_miss, 0.35), (workload, 0.20), (command, 0.10)]
    )
    score = round(overall, 1)
    return {
        "pitcher_k_score": score,
        "pitcher_k_score_label": score_label(score),
        "pitcher_k_score_confidence": _data_confidence(
            _number(row, "pitcher_games_prior", 0),
            _number(row, "sc_pitcher_games_10", 0),
        ),
        "pitcher_k_score_skill": round(strikeout_skill, 1),
        "pitcher_k_score_swing_miss": round(swing_miss, 1),
        "pitcher_k_score_workload": round(workload, 1),
        "pitcher_k_score_command": round(command, 1),
    }


def add_batter_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for _, row in frame.iterrows():
        rows.append({**hit_score(row), **home_run_score(row)})
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_pitcher_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = [pitcher_k_score(row) for _, row in frame.iterrows()]
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
