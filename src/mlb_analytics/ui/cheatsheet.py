from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd
import streamlit as st

from mlb_analytics.evaluation.cheatsheet import (
    DEFAULT_FILTERS,
    recommended_filters,
    snapshot_is_stale,
)


def _percent(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}%}" if isfinite(number) else "N/A"


def _number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if isfinite(number) else "N/A"


def _generated_label(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "No saved evaluation yet"
    value = snapshot.get("generated_at")
    try:
        generated = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return generated.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(value or "Unknown")


def _safe_table(records: list[dict[str, Any]], kind: str) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    percent_columns = [
        "avg_prediction",
        "observed_rate",
        "calibration_gap",
        "observed_ci_low",
        "observed_ci_high",
        "lift_vs_holdout",
        "within_one_rate",
        "actual_5_plus_rate",
        "actual_6_plus_rate",
        "actual_7_plus_rate",
    ]
    for column in percent_columns:
        if column in frame.columns:
            frame[column] = frame[column].map(_percent)
    for column in ["brier"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: _number(value, 4))
    for column in ["avg_score", "avg_projected", "avg_actual", "bias", "mae"]:
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: _number(value, 2))
    frame.columns = [column.replace("_", " ").title() for column in frame.columns]
    return frame


def _status(snapshot: dict[str, Any] | None, model_dir) -> None:
    if not snapshot:
        st.info(
            "This cheat sheet is using baseline defaults. Run the holdout reliability "
            "evaluation on the Calibration page to create an auto-updating version."
        )
        return
    if snapshot_is_stale(snapshot, model_dir):
        st.warning(
            "The saved models are newer than this cheat sheet. Run the Calibration "
            "evaluation again before treating these thresholds as current."
        )
    else:
        st.success(f"Current for the saved models · updated {_generated_label(snapshot)}")


def render_hits_cheat_sheet(
    snapshot: dict[str, Any] | None,
    model_dir,
    compact: bool = False,
) -> None:
    rules = recommended_filters(snapshot)["hits"]
    market = (snapshot or {}).get("markets", {}).get("hits", {})
    summary = market.get("summary", {})
    _status(snapshot, model_dir)
    cols = st.columns(4)
    cols[0].metric("Probability floor", _percent(rules.get("probability_at_least")))
    cols[1].metric("Hit Score floor", int(rules.get("score_at_least", 55)))
    cols[2].metric("Observed rule rate", _percent(rules.get("observed_rate")))
    cols[3].metric("Rule sample", f"{int(rules.get('count') or 0):,}" if rules.get("count") else "Pending")
    st.caption(
        f"{rules.get('label', DEFAULT_FILTERS['hits']['label'])}. "
        f"{rules.get('note', DEFAULT_FILTERS['hits']['note'])} "
        "Prefer confirmed lineups and batting-order positions 1–6."
    )
    if summary:
        st.caption(
            f"Full holdout: {int(summary.get('rows') or 0):,} rows · "
            f"predicted {_percent(summary.get('avg_prediction'))} · "
            f"observed {_percent(summary.get('observed_rate'))} · "
            f"calibration error {_percent(summary.get('calibration_error'))}."
        )
    if not compact:
        p_tab, s_tab = st.tabs(["Probability buckets", "Hit Score buckets"])
        with p_tab:
            table = _safe_table(market.get("probability_buckets", []), "binary")
            if table.empty:
                st.info("Run Calibration to populate probability buckets.")
            else:
                st.dataframe(table, width="stretch", hide_index=True)
        with s_tab:
            table = _safe_table(market.get("score_buckets", []), "binary")
            if table.empty:
                st.info("Run Calibration to populate score buckets.")
            else:
                st.dataframe(table, width="stretch", hide_index=True)


def render_hr_cheat_sheet(
    snapshot: dict[str, Any] | None,
    model_dir,
    compact: bool = False,
) -> None:
    rules = recommended_filters(snapshot)["home_runs"]
    market = (snapshot or {}).get("markets", {}).get("home_runs", {})
    summary = market.get("summary", {})
    _status(snapshot, model_dir)
    cols = st.columns(4)
    cols[0].metric("Probability floor", _percent(rules.get("probability_at_least")))
    cols[1].metric("HR Score floor", int(rules.get("score_at_least", 70)))
    cols[2].metric("Observed rule rate", _percent(rules.get("observed_rate")))
    cols[3].metric("Rule sample", f"{int(rules.get('count') or 0):,}" if rules.get("count") else "Pending")
    st.caption(
        f"{rules.get('label', DEFAULT_FILTERS['home_runs']['label'])}. "
        f"{rules.get('note', DEFAULT_FILTERS['home_runs']['note'])} "
        "Treat the pitch-type score as confirmation, not as a standalone probability."
    )
    if summary:
        st.caption(
            f"Full holdout: {int(summary.get('rows') or 0):,} rows · "
            f"predicted {_percent(summary.get('avg_prediction'))} · "
            f"observed {_percent(summary.get('observed_rate'))} · "
            f"calibration error {_percent(summary.get('calibration_error'))}."
        )
    if not compact:
        p_tab, s_tab = st.tabs(["Probability buckets", "HR Score buckets"])
        with p_tab:
            table = _safe_table(market.get("probability_buckets", []), "binary")
            if table.empty:
                st.info("Run Calibration to populate probability buckets.")
            else:
                st.dataframe(table, width="stretch", hide_index=True)
        with s_tab:
            table = _safe_table(market.get("score_buckets", []), "binary")
            if table.empty:
                st.info("Run Calibration to populate score buckets.")
            else:
                st.dataframe(table, width="stretch", hide_index=True)


def render_k_cheat_sheet(
    snapshot: dict[str, Any] | None,
    model_dir,
    compact: bool = False,
) -> None:
    rules = recommended_filters(snapshot)["strikeouts"]
    market = (snapshot or {}).get("markets", {}).get("strikeouts", {})
    summary = market.get("summary", {})
    _status(snapshot, model_dir)
    cols = st.columns(4)
    cols[0].metric("Minimum K Score", int(rules.get("score_at_least", 55)))
    cols[1].metric("Supported tier", rules.get("score_grade", "Favorable"))
    cols[2].metric("Observed 6+ rate", _percent(rules.get("actual_6_plus_rate")))
    cols[3].metric("Tier sample", f"{int(rules.get('count') or 0):,}" if rules.get("count") else "Pending")
    st.caption(
        f"{rules.get('label', DEFAULT_FILTERS['strikeouts']['label'])}. "
        f"{rules.get('note', DEFAULT_FILTERS['strikeouts']['note'])}"
    )
    if summary:
        st.caption(
            f"Full holdout: {int(summary.get('rows') or 0):,} starts · "
            f"projection {_number(summary.get('avg_projected'))} · "
            f"actual {_number(summary.get('avg_actual'))} · "
            f"MAE {_number(summary.get('mae'), 3)} · "
            f"within one {_percent(summary.get('within_one_rate'))}."
        )
    if not compact:
        table = _safe_table(market.get("score_buckets", []), "count")
        if table.empty:
            st.info("Run Calibration to populate K Score buckets.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)


def render_cheat_sheet(
    snapshot: dict[str, Any] | None,
    model_dir,
    market: str | None = None,
    compact: bool = False,
) -> None:
    if market == "hits":
        render_hits_cheat_sheet(snapshot, model_dir, compact)
        return
    if market == "home_runs":
        render_hr_cheat_sheet(snapshot, model_dir, compact)
        return
    if market == "strikeouts":
        render_k_cheat_sheet(snapshot, model_dir, compact)
        return
    hit_tab, hr_tab, k_tab = st.tabs(["Hits", "Home Runs", "Pitchers"])
    with hit_tab:
        render_hits_cheat_sheet(snapshot, model_dir, compact)
    with hr_tab:
        render_hr_cheat_sheet(snapshot, model_dir, compact)
    with k_tab:
        render_k_cheat_sheet(snapshot, model_dir, compact)
