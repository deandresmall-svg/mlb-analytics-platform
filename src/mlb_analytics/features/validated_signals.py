from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

from mlb_analytics.config import settings
from mlb_analytics.evaluation.cheatsheet import (
    load_reliability_snapshot,
    recommended_filters,
    score_bucket_evidence,
)


def _number(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if isfinite(value) else default


def _current_snapshot() -> dict[str, Any] | None:
    return load_reliability_snapshot(settings.reliability_snapshot_path)


def batter_signal_columns(
    row: pd.Series,
    thresholds: dict[str, dict[str, Any]] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, object]:
    snapshot = snapshot if snapshot is not None else _current_snapshot()
    thresholds = thresholds or recommended_filters(snapshot)

    hit_rule = thresholds["hits"]
    hit_probability_floor = float(hit_rule.get("probability_at_least", 0.65))
    hit_score_floor = float(hit_rule.get("score_at_least", 55))
    hit_probability = _number(row, "hit_probability")
    hit_score = _number(row, "hit_score", 50.0)
    order = _number(row, "batting_order", 9.0)
    lineup_status = str(row.get("lineup_status", ""))

    if (
        hit_probability >= hit_probability_floor
        and hit_score >= hit_score_floor
        and order <= 6
    ):
        hit_signal = "High-probability hit"
    elif (
        hit_probability >= max(0.40, hit_probability_floor - 0.05)
        and hit_score >= hit_score_floor
    ):
        hit_signal = "Favorable hit profile"
    else:
        hit_signal = "Watch / price dependent"

    hr_rule = thresholds["home_runs"]
    hr_probability_floor = float(hr_rule.get("probability_at_least", 0.15))
    hr_score_floor = float(hr_rule.get("score_at_least", 70))
    hr_probability = _number(row, "home_run_probability")
    hr_score = _number(row, "home_run_score", 50.0)
    if hr_probability >= hr_probability_floor and hr_score >= hr_score_floor:
        hr_signal = "Strong power matchup"
    elif (
        hr_probability >= max(0.03, hr_probability_floor - 0.025)
        and hr_score >= max(55.0, hr_score_floor - 15.0)
    ):
        hr_signal = "Favorable power matchup"
    else:
        hr_signal = "Watch / price dependent"

    lineup_note = (
        "Confirmed lineup"
        if "confirmed" in lineup_status.lower()
        else "Projected lineup"
    )

    return {
        "hit_calibrated_probability": hit_probability,
        "home_run_calibrated_probability": hr_probability,
        "hit_signal": hit_signal,
        "home_run_signal": hr_signal,
        "hit_score_sample_note": score_bucket_evidence(
            snapshot, "hits", row.get("hit_score_label")
        ),
        "home_run_score_sample_note": score_bucket_evidence(
            snapshot, "home_runs", row.get("home_run_score_label")
        ),
        "lineup_evidence": lineup_note,
    }


def pitcher_signal_columns(
    row: pd.Series,
    thresholds: dict[str, dict[str, Any]] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, object]:
    snapshot = snapshot if snapshot is not None else _current_snapshot()
    thresholds = thresholds or recommended_filters(snapshot)
    recommended_score = float(
        thresholds["strikeouts"].get("score_at_least", 55)
    )
    score = _number(row, "pitcher_k_score", 50.0)
    projection = _number(row, "projected_strikeouts", 0.0)
    if score >= max(70.0, recommended_score):
        signal = "Strong strikeout environment"
    elif score >= recommended_score:
        signal = "Favorable strikeout environment"
    elif score >= 45:
        signal = "Neutral strikeout environment"
    else:
        signal = "Weak strikeout environment"
    return {
        "pitcher_k_signal": signal,
        "pitcher_k_score_sample_note": score_bucket_evidence(
            snapshot, "strikeouts", row.get("pitcher_k_score_label")
        ),
        "projection_note": (
            "Use against the sportsbook line; the point estimate is not a prop line."
            if projection > 0
            else "Projection unavailable"
        ),
    }


def add_batter_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    snapshot = _current_snapshot()
    thresholds = recommended_filters(snapshot)
    additions = [
        batter_signal_columns(row, thresholds=thresholds, snapshot=snapshot)
        for _, row in frame.iterrows()
    ]
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(additions)], axis=1
    )


def add_pitcher_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    snapshot = _current_snapshot()
    thresholds = recommended_filters(snapshot)
    additions = [
        pitcher_signal_columns(row, thresholds=thresholds, snapshot=snapshot)
        for _, row in frame.iterrows()
    ]
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(additions)], axis=1
    )
