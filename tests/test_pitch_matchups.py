from __future__ import annotations

import pandas as pd

from mlb_analytics.features.pitch_matchups import (
    batter_vs_pitch_mix_features,
    index_pitch_type_games,
    pitcher_vs_lineup_pitch_mix_features,
)


def _row(
    player_id: int,
    game_pk: int,
    game_date: str,
    pitch_type: str,
    pitches: int,
    opponent_hand: str,
    player_hand: str,
    xba: float,
    xslg: float,
    xwoba: float,
    whiff_rate: float,
) -> dict:
    swings = max(int(pitches * 0.5), 1)
    bbe = max(int(pitches * 0.18), 1)
    return {
        "player_id": player_id,
        "game_pk": game_pk,
        "game_date": game_date,
        "pitch_type": pitch_type,
        "opponent_hand": opponent_hand,
        "player_hand": player_hand,
        "pitches": pitches,
        "swings": swings,
        "whiffs": int(swings * whiff_rate),
        "out_zone_pitches": max(int(pitches * 0.5), 1),
        "chases": int(pitches * 0.14),
        "bbe": bbe,
        "hard_hits": int(bbe * 0.45),
        "barrels": int(bbe * 0.10),
        "fly_balls": int(bbe * 0.28),
        "xba_sum": xba * bbe,
        "xba_count": bbe,
        "xslg_sum": xslg * bbe,
        "xslg_count": bbe,
        "xwoba_sum": xwoba * bbe,
        "xwoba_count": bbe,
    }


def test_batter_vs_pitch_mix_uses_pitcher_usage_and_no_future_rows():
    batter = pd.DataFrame(
        [
            _row(10, 1, "2026-06-01", "FF", 100, "R", "L", .300, .520, .380, .18),
            _row(10, 2, "2026-06-02", "SL", 80, "R", "L", .210, .320, .270, .34),
            _row(10, 99, "2026-08-01", "FF", 500, "R", "L", .500, .900, .600, .05),
        ]
    )
    pitcher = pd.DataFrame(
        [
            _row(20, 3, "2026-06-01", "FF", 240, "L", "R", .270, .460, .340, .24),
            _row(20, 4, "2026-06-02", "SL", 60, "L", "R", .230, .360, .290, .30),
        ]
    )
    output = batter_vs_pitch_mix_features(
        index_pitch_type_games(batter),
        index_pitch_type_games(pitcher),
        10,
        20,
        "2026-07-01",
    )
    assert output["pt_primary_pitch"] == "FF"
    assert output["pt_primary_pitch_usage"] > 0.75
    assert 0.24 < output["pt_matchup_xba"] < 0.34
    assert output["pt_pitch_mix_coverage"] > 0.70


def test_pitcher_lineup_features_use_likely_hitters():
    batter_pitch_types = []
    batting = []
    for player_id in range(1, 10):
        for game in range(1, 5):
            batting.append(
                {
                    "player_id": player_id,
                    "team_id": 100,
                    "game_pk": game,
                    "game_date": f"2026-06-0{game}",
                    "plate_appearances": 4,
                }
            )
        batter_pitch_types.append(
            _row(player_id, 10 + player_id, "2026-06-05", "FF", 60, "R", "R", .240, .390, .310, .30)
        )
    pitcher_pitch_types = pd.DataFrame(
        [_row(99, 50, "2026-06-05", "FF", 300, "R", "R", .240, .390, .310, .28)]
    )
    output = pitcher_vs_lineup_pitch_mix_features(
        index_pitch_type_games(pd.DataFrame(batter_pitch_types)),
        index_pitch_type_games(pitcher_pitch_types),
        pd.DataFrame(batting),
        99,
        100,
        "2026-07-01",
    )
    assert output["pt_lineup_hitters"] == 9
    assert output["pt_primary_pitch"] == "FF"
    assert output["pt_lineup_whiff_rate"] > 0.24
