from __future__ import annotations

from math import isfinite

import pandas as pd
import streamlit as st

from mlb_analytics.config import settings
from mlb_analytics.evaluation.cheatsheet import (
    build_reliability_snapshot,
    load_reliability_snapshot,
    save_reliability_snapshot,
)
from mlb_analytics.evaluation.reliability import (
    binary_holdout_predictions,
    binary_probability_buckets,
    binary_score_buckets,
    binary_summary,
    binary_threshold_grid,
    count_holdout_predictions,
    count_score_buckets,
    count_summary,
    expected_calibration_error,
)
from mlb_analytics.features.player_features import HIT_FEATURES, HR_FEATURES, K_FEATURES
from mlb_analytics.features.scores import add_batter_scores, add_pitcher_scores
from mlb_analytics.services.pipeline import AnalyticsService
from mlb_analytics.ui.cheatsheet import render_cheat_sheet


st.set_page_config(page_title="Calibration & Score Reliability", layout="wide")
st.title("Calibration & Score Reliability")
st.caption(
    "Chronological holdout evaluation using the saved models. The final 20% of "
    "each training dataset is kept out of model fitting, then grouped by model "
    "probability and dashboard score."
)


def _percent(value: object, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number * 100:.{digits}f}%" if isfinite(number) else "N/A"


def _number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if isfinite(number) else "N/A"


def _display_binary_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No populated buckets for this evaluation.")
        return
    output = frame.copy()
    for column in [
        "avg_prediction",
        "observed_rate",
        "calibration_gap",
        "observed_ci_low",
        "observed_ci_high",
        "lift_vs_holdout",
    ]:
        if column in output.columns:
            output[column] = output[column].map(_percent)
    if "brier" in output.columns:
        output["brier"] = output["brier"].map(lambda value: _number(value, 4))
    output.columns = [column.replace("_", " ").title() for column in output.columns]
    st.dataframe(output, width="stretch", hide_index=True)


def _display_count_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No populated score buckets for this evaluation.")
        return
    output = frame.copy()
    for column in ["within_one_rate", "actual_5_plus_rate", "actual_6_plus_rate", "actual_7_plus_rate"]:
        if column in output.columns:
            output[column] = output[column].map(_percent)
    for column in ["avg_score", "avg_projected", "avg_actual", "bias", "mae"]:
        if column in output.columns:
            output[column] = output[column].map(lambda value: _number(value, 2))
    output.columns = [column.replace("_", " ").title() for column in output.columns]
    st.dataframe(output, width="stretch", hide_index=True)


service = AnalyticsService(settings)

st.info(
    "Run this after Train all models. It reads the currently saved Hit, Home Run, "
    "and Strikeout models and evaluates only their chronological holdout rows."
)

if st.button("Run holdout reliability evaluation", type="primary", width="stretch"):
    try:
        with st.spinner("Building point-in-time datasets and evaluating saved models..."):
            _, batter, pitcher = service.datasets()
            scored_batter = add_batter_scores(batter)
            scored_pitcher = add_pitcher_scores(pitcher)

            hit_predictions = binary_holdout_predictions(
                scored_batter,
                settings.model_dir / "hit.joblib",
                HIT_FEATURES,
                "hit",
                "hit_score",
                "hit_score_label",
            )
            hr_predictions = binary_holdout_predictions(
                scored_batter,
                settings.model_dir / "home_run.joblib",
                HR_FEATURES,
                "home_run",
                "home_run_score",
                "home_run_score_label",
            )
            k_predictions = count_holdout_predictions(
                scored_pitcher,
                settings.model_dir / "strikeouts.joblib",
                K_FEATURES,
                "strikeouts",
                "pitcher_k_score",
                "pitcher_k_score_label",
            )

            snapshot = build_reliability_snapshot(
                hit_predictions,
                hr_predictions,
                k_predictions,
                settings.model_dir,
            )
            save_reliability_snapshot(
                snapshot,
                settings.reliability_snapshot_path,
            )
            st.session_state["reliability_results"] = {
                "hit_predictions": hit_predictions,
                "hr_predictions": hr_predictions,
                "k_predictions": k_predictions,
            }
            st.session_state["reliability_snapshot"] = snapshot
        st.success(
            "Holdout reliability evaluation completed and the dashboard cheat sheet was updated."
        )
    except Exception as exc:
        st.error(f"Holdout evaluation failed: {exc}")

snapshot = st.session_state.get("reliability_snapshot") or load_reliability_snapshot(
    settings.reliability_snapshot_path
)
with st.expander("Live dashboard cheat sheet", expanded=False):
    render_cheat_sheet(snapshot, settings.model_dir, compact=True)

results = st.session_state.get("reliability_results")
if not results:
    st.caption(
        "The saved cheat sheet can still be used across the dashboard. Run the evaluation "
        "above to refresh the detailed reliability tables in this browser session."
    )
    st.stop()

hit_predictions = results["hit_predictions"]
hr_predictions = results["hr_predictions"]
k_predictions = results["k_predictions"]

hit_probability = binary_probability_buckets(
    hit_predictions,
    [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.000001],
    ["<45%", "45–50%", "50–55%", "55–60%", "60–65%", "65–70%", "70%+"],
)
hr_probability = binary_probability_buckets(
    hr_predictions,
    [0.0, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 1.000001],
    ["<3%", "3–5%", "5–7.5%", "7.5–10%", "10–12.5%", "12.5–15%", "15%+"],
)

hit_tab, hr_tab, k_tab = st.tabs(["Hits", "Home Runs", "Strikeouts"])

with hit_tab:
    summary = binary_summary(hit_predictions)
    ece = expected_calibration_error(hit_probability)
    cols = st.columns(5)
    cols[0].metric("Holdout rows", f"{summary['rows']:,}")
    cols[1].metric("Average model probability", _percent(summary["avg_prediction"]))
    cols[2].metric("Observed 1+ hit rate", _percent(summary["observed_rate"]))
    cols[3].metric("Brier score", _number(summary["brier"], 4))
    cols[4].metric("Calibration error", _percent(ece))

    st.subheader("Probability reliability")
    _display_binary_table(hit_probability)

    st.subheader("Hit Score reliability")
    hit_scores = binary_score_buckets(hit_predictions, "hit_score_label")
    _display_binary_table(hit_scores)

    st.subheader("Probability + score rule checks")
    st.caption(
        "These are descriptive holdout results, not automatically approved bets. "
        "Pay close attention to sample count and calibration gap."
    )
    hit_grid = binary_threshold_grid(
        hit_predictions,
        "hit_score",
        [0.55, 0.60, 0.65, 0.70],
        [45, 55, 65, 70],
        minimum_rows=25,
    )
    _display_binary_table(hit_grid)

with hr_tab:
    summary = binary_summary(hr_predictions)
    ece = expected_calibration_error(hr_probability)
    cols = st.columns(5)
    cols[0].metric("Holdout rows", f"{summary['rows']:,}")
    cols[1].metric("Average model probability", _percent(summary["avg_prediction"]))
    cols[2].metric("Observed HR rate", _percent(summary["observed_rate"]))
    cols[3].metric("Brier score", _number(summary["brier"], 4))
    cols[4].metric("Calibration error", _percent(ece))

    st.subheader("Probability reliability")
    _display_binary_table(hr_probability)

    st.subheader("Home Run Score reliability")
    hr_scores = binary_score_buckets(hr_predictions, "home_run_score_label")
    _display_binary_table(hr_scores)

    st.subheader("Probability + score rule checks")
    hr_grid = binary_threshold_grid(
        hr_predictions,
        "home_run_score",
        [0.05, 0.075, 0.10, 0.125, 0.15],
        [45, 55, 65, 70],
        minimum_rows=20,
    )
    _display_binary_table(hr_grid)

with k_tab:
    summary = count_summary(k_predictions)
    cols = st.columns(6)
    cols[0].metric("Holdout starts", f"{summary['rows']:,}")
    cols[1].metric("Average projection", _number(summary["avg_projected"], 2))
    cols[2].metric("Average actual Ks", _number(summary["avg_actual"], 2))
    cols[3].metric("Bias", _number(summary["bias"], 2))
    cols[4].metric("MAE", _number(summary["mae"], 3))
    cols[5].metric("Within 1 K", _percent(summary["within_one_rate"]))

    st.subheader("K Score reliability")
    k_scores = count_score_buckets(k_predictions, "pitcher_k_score_label")
    _display_count_table(k_scores)

st.warning(
    "Do not change score weights from a single bucket with a small sample. A useful "
    "score should generally show monotonic improvement from Poor to Elite and remain "
    "stable on future out-of-sample games."
)
