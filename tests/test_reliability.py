from pathlib import Path

import numpy as np
import pandas as pd

from mlb_analytics.evaluation.reliability import (
    binary_probability_buckets,
    binary_score_buckets,
    binary_threshold_grid,
    count_score_buckets,
    expected_calibration_error,
)


def test_binary_probability_buckets_and_ece():
    frame = pd.DataFrame(
        {
            "predicted_probability": [0.42, 0.48, 0.52, 0.58, 0.66, 0.72],
            "actual": [0, 1, 0, 1, 1, 1],
            "squared_error": [
                (0.42 - 0) ** 2,
                (0.48 - 1) ** 2,
                (0.52 - 0) ** 2,
                (0.58 - 1) ** 2,
                (0.66 - 1) ** 2,
                (0.72 - 1) ** 2,
            ],
        }
    )
    table = binary_probability_buckets(
        frame,
        [0.0, 0.5, 0.6, 1.0001],
        ["low", "middle", "high"],
    )
    assert table["count"].sum() == 6
    assert np.isfinite(expected_calibration_error(table))


def test_binary_score_buckets_order_and_threshold_grid():
    frame = pd.DataFrame(
        {
            "predicted_probability": [0.55, 0.61, 0.66, 0.72] * 10,
            "actual": [0, 1, 1, 1] * 10,
            "squared_error": [0.30, 0.15, 0.12, 0.08] * 10,
            "hit_score": [45, 55, 65, 75] * 10,
            "hit_score_label": ["Neutral", "Favorable", "Favorable", "Strong"] * 10,
        }
    )
    scores = binary_score_buckets(frame, "hit_score_label")
    assert scores["count"].sum() == 40
    grid = binary_threshold_grid(
        frame,
        "hit_score",
        [0.60, 0.65],
        [55, 65],
        minimum_rows=5,
    )
    assert not grid.empty
    assert (grid["count"] >= 5).all()


def test_count_score_buckets():
    frame = pd.DataFrame(
        {
            "pitcher_k_score": [40, 50, 60, 75],
            "pitcher_k_score_label": ["Weak", "Neutral", "Favorable", "Strong"],
            "projected": [4.0, 5.0, 6.0, 7.0],
            "actual": [3, 5, 7, 8],
            "error": [1, 0, -1, -1],
            "absolute_error": [1, 0, 1, 1],
        }
    )
    table = count_score_buckets(frame, "pitcher_k_score_label")
    assert table["count"].sum() == 4
    assert "actual_6_plus_rate" in table.columns
