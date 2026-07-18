import pandas as pd

from mlb_analytics.features.scores import (
    hit_score,
    home_run_score,
    pitcher_k_score,
    score_label,
)


def test_scores_are_bounded_and_labeled():
    row = pd.Series(
        {
            "games_prior": 30,
            "sc_batter_games_30": 25,
            "batting_avg_30": 0.310,
            "hit_rate_10": 0.80,
            "balls_in_play_rate_30": 0.74,
            "projected_pa": 4.7,
            "batting_order": 2,
            "sc_batter_xba_30": 0.300,
            "sc_batter_hard_hit_rate_30": 0.50,
            "sc_batter_sweet_spot_rate_30": 0.40,
            "sc_batter_whiff_rate_30": 0.18,
            "sc_opp_xba_allowed_10": 0.275,
            "sc_opp_hard_hit_rate_allowed_10": 0.44,
            "sc_opp_whiff_rate_10": 0.20,
            "opponent_sp_whip_5": 1.45,
            "iso_30": 0.240,
            "home_runs_per_pa_30": 0.060,
            "slugging_30": 0.560,
            "sc_batter_barrel_rate_30": 0.14,
            "sc_batter_avg_ev_30": 91.5,
            "sc_batter_max_ev_30": 112,
            "sc_batter_xslg_30": 0.590,
            "sc_opp_barrel_rate_allowed_10": 0.11,
            "sc_opp_xslg_allowed_10": 0.500,
            "sc_opp_avg_ev_allowed_10": 90.0,
            "opponent_sp_hr_per_9_5": 1.6,
        }
    )
    hit = hit_score(row)
    hr = home_run_score(row)
    assert 0 <= hit["hit_score"] <= 100
    assert 0 <= hr["home_run_score"] <= 100
    assert hit["hit_score_confidence"] == "High"
    assert hr["home_run_score_confidence"] == "High"
    assert score_label(90) == "Elite"


def test_pitcher_score_is_bounded():
    row = pd.Series(
        {
            "pitcher_games_prior": 20,
            "sc_pitcher_games_10": 10,
            "pitcher_k_per_9_10": 11.5,
            "pitcher_k_rate_10": 0.31,
            "pitcher_k_avg_5": 7.2,
            "sc_pitcher_whiff_rate_10": 0.34,
            "sc_pitcher_chase_rate_10": 0.34,
            "sc_pitcher_csw_rate_10": 0.31,
            "pitcher_pitches_5": 96,
            "pitcher_ip_5": 6.1,
            "pitcher_k_minus_bb_rate_10": 0.22,
            "sc_pitcher_zone_rate_10": 0.51,
        }
    )
    score = pitcher_k_score(row)
    assert 0 <= score["pitcher_k_score"] <= 100
    assert score["pitcher_k_score_confidence"] == "High"
