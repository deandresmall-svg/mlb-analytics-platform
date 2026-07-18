from pathlib import Path

import pandas as pd

from mlb_analytics.data.repository import Repository
from mlb_analytics.data.statcast import aggregate_statcast, prepare_statcast


def sample_pitches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 1,
                "game_date": "2026-07-01",
                "game_type": "R",
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 100,
                "pitcher": 200,
                "stand": "L",
                "p_throws": "R",
                "description": "called_strike",
                "pitch_type": "FF",
                "zone": 5,
                "release_speed": 95.0,
                "release_spin_rate": 2300,
            },
            {
                "game_pk": 1,
                "game_date": "2026-07-01",
                "game_type": "R",
                "at_bat_number": 1,
                "pitch_number": 2,
                "batter": 100,
                "pitcher": 200,
                "stand": "L",
                "p_throws": "R",
                "description": "ball",
                "pitch_type": "SL",
                "zone": 11,
                "release_speed": 86.0,
                "release_spin_rate": 2500,
            },
            {
                "game_pk": 1,
                "game_date": "2026-07-01",
                "game_type": "R",
                "at_bat_number": 1,
                "pitch_number": 3,
                "batter": 100,
                "pitcher": 200,
                "stand": "L",
                "p_throws": "R",
                "description": "swinging_strike",
                "pitch_type": "SL",
                "zone": 11,
                "release_speed": 87.0,
                "release_spin_rate": 2550,
            },
            {
                "game_pk": 1,
                "game_date": "2026-07-01",
                "game_type": "R",
                "at_bat_number": 1,
                "pitch_number": 4,
                "batter": 100,
                "pitcher": 200,
                "stand": "L",
                "p_throws": "R",
                "description": "hit_into_play",
                "events": "home_run",
                "pitch_type": "FF",
                "zone": 5,
                "release_speed": 96.0,
                "release_spin_rate": 2350,
                "launch_speed": 100.0,
                "launch_angle": 20.0,
                "launch_speed_angle": 6,
                "bb_type": "fly_ball",
                "estimated_ba_using_speedangle": 0.800,
                "estimated_slg_using_speedangle": 2.000,
                "estimated_woba_using_speedangle": 0.900,
            },
        ]
    )


def test_prepare_and_aggregate_statcast():
    pitches = prepare_statcast(sample_pitches())
    assert pitches["is_swing"].sum() == 2
    assert pitches["is_whiff"].sum() == 1
    assert pitches["is_chase"].sum() == 1
    assert pitches["is_hard_hit"].sum() == 1
    assert pitches["is_barrel"].sum() == 1
    assert pitches["is_sweet_spot"].sum() == 1

    aggregates = aggregate_statcast(pitches)
    batter = aggregates["batters"].iloc[0]
    pitcher = aggregates["pitchers"].iloc[0]

    assert batter["pitches"] == 4
    assert batter["plate_appearances"] == 1
    assert batter["swings"] == 2
    assert batter["whiffs"] == 1
    assert batter["out_zone_pitches"] == 2
    assert batter["chases"] == 1
    assert batter["bbe"] == 1
    assert batter["barrels"] == 1
    assert pitcher["hard_hits"] == 1
    assert len(aggregates["batter_pitch_types"]) == 2
    assert len(aggregates["pitcher_pitch_types"]) == 2


def test_repository_statcast_upserts(tmp_path: Path):
    repo = Repository(f"sqlite:///{tmp_path / 'analytics.db'}")
    repo.initialize()
    pitches = prepare_statcast(sample_pitches())
    aggregates = aggregate_statcast(pitches)

    assert repo.upsert_statcast_pitches(pitches) == 4
    assert repo.upsert_statcast_batters(aggregates["batters"]) == 1
    assert repo.upsert_statcast_pitchers(aggregates["pitchers"]) == 1
    assert repo.upsert_statcast_batter_pitch_types(
        aggregates["batter_pitch_types"]
    ) == 2
    assert repo.upsert_statcast_pitcher_pitch_types(
        aggregates["pitcher_pitch_types"]
    ) == 2

    coverage = repo.coverage().iloc[0]
    assert coverage["statcast_pitches"] == 4
    assert coverage["statcast_batter_games"] == 1
    assert coverage["statcast_pitcher_games"] == 1


def test_prepare_statcast_handles_missing_nullable_measurements():
    frame = pd.DataFrame(
        [
            {
                "game_pk": 2,
                "game_date": "2026-06-10",
                "at_bat_number": 1,
                "pitch_number": 1,
                "batter": 101,
                "pitcher": 201,
                "description": "ball",
                "zone": pd.NA,
                "launch_speed": pd.NA,
                "launch_angle": pd.NA,
                "launch_speed_angle": pd.NA,
            }
        ]
    )

    pitches = prepare_statcast(frame)

    assert len(pitches) == 1
    assert pitches.loc[0, "is_in_zone"] == 0
    assert pitches.loc[0, "is_out_zone"] == 0
    assert pitches.loc[0, "is_hard_hit"] == 0
    assert pitches.loc[0, "is_barrel"] == 0
    assert pitches.loc[0, "is_sweet_spot"] == 0
