from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from mlb_analytics.models.base import BinaryTimeModel, CountTimeModel


SCORE_ORDER = ["Poor", "Weak", "Neutral", "Favorable", "Strong", "Elite"]


def _chronological_test(dataframe: pd.DataFrame, label: str) -> pd.DataFrame:
    data = dataframe.dropna(subset=[label]).copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data.dropna(subset=["game_date"]).sort_values(
        ["game_date"] + [
            column
            for column in ("game_pk", "player_id")
            if column in data.columns
        ]
    )
    cut = max(int(len(data) * 0.8), 1)
    test = data.iloc[cut:].copy()
    if len(test) < 10:
        raise ValueError("Not enough chronological holdout rows")
    return test


def binary_holdout_predictions(
    dataframe: pd.DataFrame,
    model_path: Path,
    fallback_features: list[str],
    label: str,
    score_column: str,
    score_label_column: str,
) -> pd.DataFrame:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    test = _chronological_test(dataframe, label)
    model = BinaryTimeModel.load(Path(model_path))
    probability = model.predict_frame(test, fallback_features)

    output_columns = [
        column
        for column in (
            "game_date",
            "game_pk",
            "player_id",
            score_column,
            score_label_column,
        )
        if column in test.columns
    ]
    output = test[output_columns].copy()
    output["predicted_probability"] = np.clip(probability, 0.0, 1.0)
    output["actual"] = pd.to_numeric(test[label], errors="coerce").to_numpy()
    output["squared_error"] = (
        output["predicted_probability"] - output["actual"]
    ) ** 2
    return output.reset_index(drop=True)


def count_holdout_predictions(
    dataframe: pd.DataFrame,
    model_path: Path,
    fallback_features: list[str],
    label: str,
    score_column: str,
    score_label_column: str,
) -> pd.DataFrame:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    test = _chronological_test(dataframe, label)
    model = CountTimeModel.load(Path(model_path))
    prediction = model.predict_frame(test, fallback_features)

    output_columns = [
        column
        for column in (
            "game_date",
            "game_pk",
            "player_id",
            score_column,
            score_label_column,
        )
        if column in test.columns
    ]
    output = test[output_columns].copy()
    output["projected"] = np.maximum(prediction, 0.0)
    output["actual"] = pd.to_numeric(test[label], errors="coerce").to_numpy()
    output["error"] = output["projected"] - output["actual"]
    output["absolute_error"] = output["error"].abs()
    return output.reset_index(drop=True)


def _wilson_interval(successes: float, count: int, z: float = 1.96) -> tuple[float, float]:
    if count <= 0:
        return np.nan, np.nan
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    margin = (
        z
        * sqrt((proportion * (1 - proportion) + z * z / (4 * count)) / count)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _binary_group_summary(grouped) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket, group in grouped:
        if group.empty:
            continue
        count = len(group)
        actual_sum = float(group["actual"].sum())
        observed = actual_sum / count
        low, high = _wilson_interval(actual_sum, count)
        predicted = float(group["predicted_probability"].mean())
        rows.append(
            {
                "bucket": str(bucket),
                "count": count,
                "avg_prediction": predicted,
                "observed_rate": observed,
                "calibration_gap": observed - predicted,
                "brier": float(group["squared_error"].mean()),
                "observed_ci_low": low,
                "observed_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def binary_probability_buckets(
    predictions: pd.DataFrame,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    data = predictions.copy()
    data["probability_bucket"] = pd.cut(
        data["predicted_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return _binary_group_summary(
        data.groupby("probability_bucket", observed=False, sort=True)
    )


def binary_score_buckets(
    predictions: pd.DataFrame,
    score_label_column: str,
) -> pd.DataFrame:
    data = predictions.copy()
    data[score_label_column] = pd.Categorical(
        data[score_label_column],
        categories=SCORE_ORDER,
        ordered=True,
    )
    return _binary_group_summary(
        data.groupby(score_label_column, observed=False, sort=True)
    )


def binary_threshold_grid(
    predictions: pd.DataFrame,
    score_column: str,
    probability_thresholds: list[float],
    score_thresholds: list[float],
    minimum_rows: int = 20,
) -> pd.DataFrame:
    baseline = float(predictions["actual"].mean()) if not predictions.empty else np.nan
    rows: list[dict[str, object]] = []
    for probability_threshold in probability_thresholds:
        for score_threshold in score_thresholds:
            sample = predictions.loc[
                (predictions["predicted_probability"] >= probability_threshold)
                & (pd.to_numeric(predictions[score_column], errors="coerce") >= score_threshold)
            ]
            if len(sample) < minimum_rows:
                continue
            observed = float(sample["actual"].mean())
            predicted = float(sample["predicted_probability"].mean())
            rows.append(
                {
                    "probability_at_least": probability_threshold,
                    "score_at_least": score_threshold,
                    "count": len(sample),
                    "avg_prediction": predicted,
                    "observed_rate": observed,
                    "calibration_gap": observed - predicted,
                    "lift_vs_holdout": observed - baseline,
                    "brier": float(sample["squared_error"].mean()),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["probability_at_least", "score_at_least"],
        ascending=[True, True],
    )


def count_score_buckets(
    predictions: pd.DataFrame,
    score_label_column: str,
) -> pd.DataFrame:
    data = predictions.copy()
    data[score_label_column] = pd.Categorical(
        data[score_label_column],
        categories=SCORE_ORDER,
        ordered=True,
    )
    rows: list[dict[str, object]] = []
    for bucket, group in data.groupby(score_label_column, observed=False, sort=True):
        if group.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "count": len(group),
                "avg_score": float(
                    pd.to_numeric(
                        group.get("pitcher_k_score"), errors="coerce"
                    ).mean()
                ),
                "avg_projected": float(group["projected"].mean()),
                "avg_actual": float(group["actual"].mean()),
                "bias": float(group["error"].mean()),
                "mae": float(group["absolute_error"].mean()),
                "within_one_rate": float((group["absolute_error"] <= 1.0).mean()),
                "actual_5_plus_rate": float((group["actual"] >= 5).mean()),
                "actual_6_plus_rate": float((group["actual"] >= 6).mean()),
                "actual_7_plus_rate": float((group["actual"] >= 7).mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    bucket_table: pd.DataFrame,
) -> float:
    if bucket_table.empty or bucket_table["count"].sum() <= 0:
        return float("nan")
    weights = bucket_table["count"] / bucket_table["count"].sum()
    gaps = (bucket_table["observed_rate"] - bucket_table["avg_prediction"]).abs()
    return float((weights * gaps).sum())


def binary_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": len(predictions),
        "avg_prediction": float(predictions["predicted_probability"].mean()),
        "observed_rate": float(predictions["actual"].mean()),
        "brier": float(brier_score_loss(
            predictions["actual"], predictions["predicted_probability"]
        )),
    }


def count_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": len(predictions),
        "avg_projected": float(predictions["projected"].mean()),
        "avg_actual": float(predictions["actual"].mean()),
        "bias": float(predictions["error"].mean()),
        "mae": float(mean_absolute_error(
            predictions["actual"], predictions["projected"]
        )),
        "within_one_rate": float((predictions["absolute_error"] <= 1.0).mean()),
    }
