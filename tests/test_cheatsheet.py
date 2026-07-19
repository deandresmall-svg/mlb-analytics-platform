from pathlib import Path

import pandas as pd

from mlb_analytics.evaluation.cheatsheet import (
    build_reliability_snapshot,
    load_reliability_snapshot,
    recommended_filters,
    save_reliability_snapshot,
    score_bucket_evidence,
    snapshot_is_stale,
)


def _binary_predictions(label: str, score_column: str, score_label_column: str) -> pd.DataFrame:
    rows = []
    for index in range(500):
        probability = 0.42 + (index % 31) / 100
        actual = int((index % 10) < round(probability * 10))
        score = 35 + (index % 46)
        grade = (
            "Strong" if score >= 70 else
            "Favorable" if score >= 55 else
            "Neutral" if score >= 45 else
            "Weak"
        )
        rows.append(
            {
                "game_date": "2026-07-01",
                "predicted_probability": probability,
                "actual": actual,
                "squared_error": (probability - actual) ** 2,
                score_column: score,
                score_label_column: grade,
            }
        )
    return pd.DataFrame(rows)


def _k_predictions() -> pd.DataFrame:
    rows = []
    for index in range(200):
        score = 25 + index % 60
        grade = (
            "Strong" if score >= 70 else
            "Favorable" if score >= 55 else
            "Neutral" if score >= 45 else
            "Weak"
        )
        projected = 3.5 + score / 20
        actual = round(projected + ((index % 3) - 1))
        rows.append(
            {
                "game_date": "2026-07-01",
                "pitcher_k_score": score,
                "pitcher_k_score_label": grade,
                "projected": projected,
                "actual": actual,
                "error": projected - actual,
                "absolute_error": abs(projected - actual),
            }
        )
    return pd.DataFrame(rows)


def test_snapshot_round_trip_and_filters(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ["hit.joblib", "home_run.joblib", "strikeouts.joblib"]:
        (model_dir / name).write_bytes(b"model")

    snapshot = build_reliability_snapshot(
        _binary_predictions("hit", "hit_score", "hit_score_label"),
        _binary_predictions("home_run", "home_run_score", "home_run_score_label"),
        _k_predictions(),
        model_dir,
    )
    path = tmp_path / "reliability.json"
    save_reliability_snapshot(snapshot, path)
    loaded = load_reliability_snapshot(path)
    assert loaded is not None
    filters = recommended_filters(loaded)
    assert filters["hits"]["probability_at_least"] >= 0.55
    assert filters["home_runs"]["score_at_least"] >= 45
    assert filters["strikeouts"]["score_at_least"] >= 45
    assert not snapshot_is_stale(loaded, model_dir)


def test_snapshot_becomes_stale_when_model_changes(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ["hit.joblib", "home_run.joblib", "strikeouts.joblib"]:
        (model_dir / name).write_bytes(b"model")
    snapshot = build_reliability_snapshot(
        _binary_predictions("hit", "hit_score", "hit_score_label"),
        _binary_predictions("home_run", "home_run_score", "home_run_score_label"),
        _k_predictions(),
        model_dir,
    )
    (model_dir / "hit.joblib").write_bytes(b"new-model-data")
    assert snapshot_is_stale(snapshot, model_dir)


def test_score_bucket_evidence_fallback():
    assert "Small-sample" in score_bucket_evidence(None, "hits", "Elite")
