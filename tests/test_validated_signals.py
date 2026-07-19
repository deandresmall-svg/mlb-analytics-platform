from __future__ import annotations

import pandas as pd

from mlb_analytics.features.validated_signals import (
    batter_signal_columns,
    pitcher_signal_columns,
)


def test_validated_hit_and_hr_signals():
    output = batter_signal_columns(
        pd.Series(
            {
                "hit_probability": 0.67,
                "hit_score": 60,
                "hit_score_label": "Favorable",
                "home_run_probability": 0.17,
                "home_run_score": 73,
                "home_run_score_label": "Strong",
                "batting_order": 3,
                "lineup_status": "Confirmed lineup",
            }
        )
    )
    assert output["hit_signal"] == "High-probability hit"
    assert output["home_run_signal"] == "Strong power matchup"
    assert output["lineup_evidence"] == "Confirmed lineup"


def test_elite_tier_gets_small_sample_warning():
    output = pitcher_signal_columns(
        pd.Series(
            {
                "pitcher_k_score": 88,
                "pitcher_k_score_label": "Elite",
                "projected_strikeouts": 7.1,
            }
        )
    )
    assert output["pitcher_k_signal"] == "Strong strikeout environment"
    assert "Small-sample" in output["pitcher_k_score_sample_note"]
