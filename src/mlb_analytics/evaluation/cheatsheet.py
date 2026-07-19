from __future__ import annotations

from datetime import datetime, timezone
import json
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_analytics.evaluation.reliability import (
    binary_probability_buckets,
    binary_score_buckets,
    binary_summary,
    binary_threshold_grid,
    count_score_buckets,
    count_summary,
    expected_calibration_error,
)


SNAPSHOT_VERSION = 1
MODEL_FILES = ("hit.joblib", "home_run.joblib", "strikeouts.joblib")
SCORE_FLOORS = {
    "Poor": 0,
    "Weak": 30,
    "Neutral": 45,
    "Favorable": 55,
    "Strong": 70,
    "Elite": 85,
}

DEFAULT_FILTERS: dict[str, dict[str, Any]] = {
    "hits": {
        "probability_at_least": 0.65,
        "score_at_least": 55,
        "label": "Baseline validated rule",
        "note": "Run Calibration to refresh this rule from the current saved model.",
    },
    "home_runs": {
        "probability_at_least": 0.15,
        "score_at_least": 70,
        "label": "Baseline validated rule",
        "note": "Run Calibration to refresh this rule from the current saved model.",
    },
    "strikeouts": {
        "score_at_least": 55,
        "score_grade": "Favorable",
        "label": "Baseline validated tier",
        "note": "Use the projection against the sportsbook line; no fixed line cushion is assumed.",
    },
}

HIT_PROBABILITY_BINS = [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 1.000001]
HIT_PROBABILITY_LABELS = [
    "<45%",
    "45–50%",
    "50–55%",
    "55–60%",
    "60–65%",
    "65–70%",
    "70%+",
]
HR_PROBABILITY_BINS = [0.0, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 1.000001]
HR_PROBABILITY_LABELS = [
    "<3%",
    "3–5%",
    "5–7.5%",
    "7.5–10%",
    "10–12.5%",
    "12.5–15%",
    "15%+",
]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def model_signature(model_dir: Path) -> dict[str, dict[str, Any]]:
    signature: dict[str, dict[str, Any]] = {}
    for filename in MODEL_FILES:
        path = Path(model_dir) / filename
        if not path.exists():
            signature[filename] = {"exists": False}
            continue
        stat = path.stat()
        signature[filename] = {
            "exists": True,
            "modified_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return signature


def _choose_binary_rule(
    grid: pd.DataFrame,
    total_rows: int,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if grid.empty:
        return dict(fallback)

    data = grid.copy()
    numeric_columns = [
        "probability_at_least",
        "score_at_least",
        "count",
        "avg_prediction",
        "observed_rate",
        "calibration_gap",
        "lift_vs_holdout",
        "brier",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")

    minimum_count = max(100, ceil(total_rows * 0.05))
    candidates = data.loc[
        (data["count"] >= minimum_count)
        & (data["calibration_gap"].abs() <= 0.035)
        & (data["lift_vs_holdout"] > 0)
    ].copy()
    if candidates.empty:
        candidates = data.loc[data["count"] >= max(50, ceil(total_rows * 0.025))].copy()
    if candidates.empty:
        return dict(fallback)

    candidates["support"] = np.log1p(candidates["count"]) / 100.0
    candidates["selection_score"] = (
        candidates["observed_rate"]
        - 0.50 * candidates["calibration_gap"].abs()
        + candidates["support"]
    )
    selected = candidates.sort_values(
        ["selection_score", "count"], ascending=[False, False]
    ).iloc[0]
    return {
        "probability_at_least": float(selected["probability_at_least"]),
        "score_at_least": int(round(float(selected["score_at_least"]))),
        "count": int(selected["count"]),
        "avg_prediction": float(selected["avg_prediction"]),
        "observed_rate": float(selected["observed_rate"]),
        "calibration_gap": float(selected["calibration_gap"]),
        "lift_vs_holdout": float(selected["lift_vs_holdout"]),
        "brier": float(selected["brier"]),
        "label": "Best supported current holdout rule",
        "note": (
            f"Selected from combinations with at least {minimum_count:,} holdout rows, "
            "positive lift, and a calibration gap within 3.5 percentage points."
        ),
    }


def _choose_k_rule(table: pd.DataFrame, total_rows: int) -> dict[str, Any]:
    fallback = dict(DEFAULT_FILTERS["strikeouts"])
    if table.empty:
        return fallback
    data = table.copy()
    for column in [
        "count",
        "avg_score",
        "avg_projected",
        "avg_actual",
        "bias",
        "mae",
        "within_one_rate",
        "actual_5_plus_rate",
        "actual_6_plus_rate",
        "actual_7_plus_rate",
    ]:
        data[column] = pd.to_numeric(data.get(column), errors="coerce")

    minimum_count = max(50, ceil(total_rows * 0.10))
    order = ["Poor", "Weak", "Neutral", "Favorable", "Strong", "Elite"]
    qualifying = data.loc[
        (data["count"] >= minimum_count)
        & (data["avg_actual"] >= 5.5)
        & (data["actual_6_plus_rate"] >= 0.50)
    ].copy()
    if not qualifying.empty:
        qualifying["rank"] = qualifying["bucket"].map(
            {label: index for index, label in enumerate(order)}
        )
        selected = qualifying.sort_values(["rank", "count"], ascending=[True, False]).iloc[0]
    else:
        supported = data.loc[data["count"] >= max(20, ceil(total_rows * 0.05))].copy()
        if supported.empty:
            return fallback
        selected = supported.sort_values(
            ["actual_6_plus_rate", "count"], ascending=[False, False]
        ).iloc[0]

    grade = str(selected["bucket"])
    return {
        "score_at_least": SCORE_FLOORS.get(grade, 55),
        "score_grade": grade,
        "count": int(selected["count"]),
        "avg_score": float(selected["avg_score"]),
        "avg_projected": float(selected["avg_projected"]),
        "avg_actual": float(selected["avg_actual"]),
        "bias": float(selected["bias"]),
        "mae": float(selected["mae"]),
        "within_one_rate": float(selected["within_one_rate"]),
        "actual_5_plus_rate": float(selected["actual_5_plus_rate"]),
        "actual_6_plus_rate": float(selected["actual_6_plus_rate"]),
        "actual_7_plus_rate": float(selected["actual_7_plus_rate"]),
        "label": "Lowest currently supported high-K tier",
        "note": (
            f"Requires at least {minimum_count:,} holdout starts, 5.5+ average actual Ks, "
            "and a 50%+ observed 6-plus rate."
        ),
    }


def build_reliability_snapshot(
    hit_predictions: pd.DataFrame,
    hr_predictions: pd.DataFrame,
    k_predictions: pd.DataFrame,
    model_dir: Path,
) -> dict[str, Any]:
    hit_probability = binary_probability_buckets(
        hit_predictions, HIT_PROBABILITY_BINS, HIT_PROBABILITY_LABELS
    )
    hr_probability = binary_probability_buckets(
        hr_predictions, HR_PROBABILITY_BINS, HR_PROBABILITY_LABELS
    )
    hit_scores = binary_score_buckets(hit_predictions, "hit_score_label")
    hr_scores = binary_score_buckets(hr_predictions, "home_run_score_label")
    k_scores = count_score_buckets(k_predictions, "pitcher_k_score_label")
    hit_grid = binary_threshold_grid(
        hit_predictions,
        "hit_score",
        [0.55, 0.60, 0.65, 0.70],
        [45, 55, 65, 70],
        minimum_rows=25,
    )
    hr_grid = binary_threshold_grid(
        hr_predictions,
        "home_run_score",
        [0.05, 0.075, 0.10, 0.125, 0.15],
        [45, 55, 65, 70],
        minimum_rows=20,
    )
    hit_summary = binary_summary(hit_predictions)
    hr_summary = binary_summary(hr_predictions)
    k_summary = count_summary(k_predictions)

    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_signature": model_signature(model_dir),
        "markets": {
            "hits": {
                "summary": {
                    **{key: _json_value(value) for key, value in hit_summary.items()},
                    "calibration_error": _json_value(
                        expected_calibration_error(hit_probability)
                    ),
                },
                "probability_buckets": dataframe_records(hit_probability),
                "score_buckets": dataframe_records(hit_scores),
                "rule_grid": dataframe_records(hit_grid),
                "recommended_rule": _choose_binary_rule(
                    hit_grid, len(hit_predictions), DEFAULT_FILTERS["hits"]
                ),
            },
            "home_runs": {
                "summary": {
                    **{key: _json_value(value) for key, value in hr_summary.items()},
                    "calibration_error": _json_value(
                        expected_calibration_error(hr_probability)
                    ),
                },
                "probability_buckets": dataframe_records(hr_probability),
                "score_buckets": dataframe_records(hr_scores),
                "rule_grid": dataframe_records(hr_grid),
                "recommended_rule": _choose_binary_rule(
                    hr_grid, len(hr_predictions), DEFAULT_FILTERS["home_runs"]
                ),
            },
            "strikeouts": {
                "summary": {
                    **{key: _json_value(value) for key, value in k_summary.items()},
                },
                "score_buckets": dataframe_records(k_scores),
                "recommended_rule": _choose_k_rule(k_scores, len(k_predictions)),
            },
        },
    }


def save_reliability_snapshot(snapshot: dict[str, Any], path: Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def load_reliability_snapshot(path: Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != SNAPSHOT_VERSION:
        return None
    return data


def snapshot_is_stale(snapshot: dict[str, Any] | None, model_dir: Path) -> bool:
    if not snapshot:
        return True
    return snapshot.get("model_signature") != model_signature(model_dir)


def recommended_filters(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    output = {market: dict(values) for market, values in DEFAULT_FILTERS.items()}
    if not snapshot:
        return output
    markets = snapshot.get("markets", {})
    for market in output:
        recommendation = markets.get(market, {}).get("recommended_rule")
        if isinstance(recommendation, dict):
            output[market].update(recommendation)
    return output


def score_bucket_evidence(
    snapshot: dict[str, Any] | None,
    market: str,
    grade: object,
    minimum_validated_rows: int = 100,
) -> str:
    label = str(grade or "Unavailable")
    if not snapshot:
        if label == "Elite":
            return "Small-sample tier — combine with Strong until revalidated"
        return "Run Calibration to refresh tier evidence"
    rows = snapshot.get("markets", {}).get(market, {}).get("score_buckets", [])
    for row in rows:
        if str(row.get("bucket")) != label:
            continue
        count = int(row.get("count") or 0)
        if count < minimum_validated_rows:
            return f"Small sample: {count:,} holdout rows"
        if market == "strikeouts":
            rate = row.get("actual_6_plus_rate")
            return (
                f"Validated: {count:,} starts · 6+ rate {float(rate):.1%}"
                if rate is not None
                else f"Validated: {count:,} starts"
            )
        observed = row.get("observed_rate")
        return (
            f"Validated: {count:,} rows · observed {float(observed):.1%}"
            if observed is not None
            else f"Validated: {count:,} rows"
        )
    return "Tier not populated in the latest holdout"
