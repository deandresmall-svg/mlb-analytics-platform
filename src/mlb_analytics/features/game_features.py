from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [
    # Team results
    "home_win_pct_7",
    "away_win_pct_7",
    "home_win_pct_14",
    "away_win_pct_14",
    "home_win_pct_30",
    "away_win_pct_30",

    # Run production
    "home_runs_pg_7",
    "away_runs_pg_7",
    "home_runs_pg_14",
    "away_runs_pg_14",
    "home_runs_pg_30",
    "away_runs_pg_30",

    # Run prevention
    "home_runs_allowed_pg_7",
    "away_runs_allowed_pg_7",
    "home_runs_allowed_pg_14",
    "away_runs_allowed_pg_14",
    "home_runs_allowed_pg_30",
    "away_runs_allowed_pg_30",

    # Run differential
    "home_run_diff_7",
    "away_run_diff_7",
    "home_run_diff_14",
    "away_run_diff_14",
    "home_run_diff_30",
    "away_run_diff_30",

    # Team hitting
    "home_batting_avg_14",
    "away_batting_avg_14",
    "home_obp_14",
    "away_obp_14",
    "home_slg_14",
    "away_slg_14",
    "home_ops_14",
    "away_ops_14",
    "home_iso_14",
    "away_iso_14",
    "home_bb_rate_14",
    "away_bb_rate_14",
    "home_k_rate_14",
    "away_k_rate_14",
    "home_hits_pg_14",
    "away_hits_pg_14",
    "home_hr_pg_14",
    "away_hr_pg_14",

    # Starting pitcher
    "home_sp_era_5",
    "away_sp_era_5",
    "home_sp_whip_5",
    "away_sp_whip_5",
    "home_sp_k_per_9_5",
    "away_sp_k_per_9_5",
    "home_sp_bb_per_9_5",
    "away_sp_bb_per_9_5",
    "home_sp_kbb_5",
    "away_sp_kbb_5",
    "home_sp_hr_per_9_5",
    "away_sp_hr_per_9_5",
    "home_sp_pitches_5",
    "away_sp_pitches_5",
    "home_sp_innings_5",
    "away_sp_innings_5",

    # Bullpen workload
    "home_bullpen_pitches_1",
    "away_bullpen_pitches_1",
    "home_bullpen_pitches_2",
    "away_bullpen_pitches_2",
    "home_bullpen_pitches_3",
    "away_bullpen_pitches_3",
    "home_bullpen_innings_3",
    "away_bullpen_innings_3",

    # Rest and environment
    "rest_diff",
    "park_factor",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "precipitation",
]


def _prior(df: pd.DataFrame, day, number: int) -> pd.DataFrame:
    """Return only games played before the current game date."""
    if df.empty or "game_date" not in df.columns:
        return df.iloc[0:0].copy()

    dates = pd.to_datetime(df["game_date"], errors="coerce")
    current_day = pd.Timestamp(day)

    return (
        df.loc[dates < current_day]
        .assign(_sort_date=dates[dates < current_day])
        .sort_values("_sort_date")
        .drop(columns="_sort_date")
        .tail(number)
    )


def _safe_sum(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0

    return float(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())


def _safe_mean(
    df: pd.DataFrame,
    column: str,
    default: float,
) -> float:
    if df.empty or column not in df.columns:
        return default

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return default

    return float(values.mean())


def _rate(
    numerator: float,
    denominator: float,
    default: float,
) -> float:
    if denominator <= 0:
        return default

    return float(numerator / denominator)


def _team_wins(
    history: pd.DataFrame,
    all_team_stats: pd.DataFrame,
) -> list[float]:
    wins: list[float] = []

    if history.empty:
        return wins

    for _, row in history.iterrows():
        opponent = all_team_stats[
            (all_team_stats["game_pk"] == row["game_pk"])
            & (all_team_stats["team_id"] == row["opponent_id"])
        ]

        if opponent.empty:
            continue

        team_runs = pd.to_numeric(
            pd.Series([row.get("runs")]),
            errors="coerce",
        ).iloc[0]

        opponent_runs = pd.to_numeric(
            opponent["runs"],
            errors="coerce",
        ).iloc[0]

        if pd.isna(team_runs) or pd.isna(opponent_runs):
            continue

        wins.append(float(team_runs > opponent_runs))

    return wins


def _runs_allowed(
    history: pd.DataFrame,
    all_team_stats: pd.DataFrame,
    default: float = 4.3,
) -> float:
    """Calculate opponent runs using the opposing team row from each game."""
    if history.empty:
        return default

    allowed: list[float] = []

    for _, row in history.iterrows():
        opponent = all_team_stats[
            (all_team_stats["game_pk"] == row["game_pk"])
            & (all_team_stats["team_id"] == row["opponent_id"])
        ]

        if opponent.empty:
            continue

        runs = pd.to_numeric(
            opponent["runs"],
            errors="coerce",
        ).iloc[0]

        if pd.notna(runs):
            allowed.append(float(runs))

    return float(np.mean(allowed)) if allowed else default


def _starter_features(
    pitcher_stats: pd.DataFrame,
    pitcher_id,
    day,
) -> dict[str, float]:
    defaults = {
        "sp_era_5": 4.30,
        "sp_whip_5": 1.30,
        "sp_k_per_9_5": 8.50,
        "sp_bb_per_9_5": 3.20,
        "sp_kbb_5": 2.50,
        "sp_hr_per_9_5": 1.20,
        "sp_pitches_5": 85.0,
        "sp_innings_5": 5.0,
    }

    if pd.isna(pitcher_id) or pitcher_stats.empty:
        return defaults

    player_ids = pd.to_numeric(
        pitcher_stats["player_id"],
        errors="coerce",
    )

    games_started = pd.to_numeric(
        pitcher_stats["games_started"],
        errors="coerce",
    ).fillna(0)

    history = _prior(
        pitcher_stats[
            (player_ids == float(pitcher_id))
            & (games_started > 0)
        ],
        day,
        5,
    )

    if history.empty:
        return defaults

    innings = _safe_sum(history, "innings_pitched")
    earned_runs = _safe_sum(history, "earned_runs")
    hits = _safe_sum(history, "hits")
    walks = _safe_sum(history, "walks")
    strikeouts = _safe_sum(history, "strikeouts")
    home_runs = _safe_sum(history, "home_runs")

    starts = max(len(history), 1)

    return {
        "sp_era_5": _rate(earned_runs * 9, innings, 4.30),
        "sp_whip_5": _rate(hits + walks, innings, 1.30),
        "sp_k_per_9_5": _rate(strikeouts * 9, innings, 8.50),
        "sp_bb_per_9_5": _rate(walks * 9, innings, 3.20),
        "sp_kbb_5": _rate(strikeouts, walks, 2.50),
        "sp_hr_per_9_5": _rate(home_runs * 9, innings, 1.20),
        "sp_pitches_5": _rate(
            _safe_sum(history, "pitches_thrown"),
            starts,
            85.0,
        ),
        "sp_innings_5": _rate(innings, starts, 5.0),
    }


def _weather_value(game: pd.Series, column: str, default: float) -> float:
    value = game.get(column, default)

    if pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_game_features(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    if games.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "label", *FEATURES])

    ts = team_stats.copy()
    ps = pitcher_stats.copy()

    if not ts.empty:
        ts["game_date"] = pd.to_datetime(
            ts["game_date"],
            errors="coerce",
        )

    if not ps.empty:
        ps["game_date"] = pd.to_datetime(
            ps["game_date"],
            errors="coerce",
        )

    sorted_games = games.copy()
    sorted_games["game_date"] = pd.to_datetime(
        sorted_games["game_date"],
        errors="coerce",
    )
    sorted_games = sorted_games.sort_values("game_date")

    for _, game in sorted_games.iterrows():
        day = game["game_date"]

        home_score = pd.to_numeric(
            pd.Series([game.get("home_score")]),
            errors="coerce",
        ).iloc[0]

        away_score = pd.to_numeric(
            pd.Series([game.get("away_score")]),
            errors="coerce",
        ).iloc[0]

        label = (
            float(home_score > away_score)
            if pd.notna(home_score) and pd.notna(away_score)
            else np.nan
        )

        row = {
            "game_pk": game["game_pk"],
            "game_date": day,
            "label": label,
        }

        for side in ("home", "away"):
            team_id = game.get(f"{side}_team_id")

            team_ids = (
                pd.to_numeric(ts["team_id"], errors="coerce")
                if not ts.empty
                else pd.Series(dtype=float)
            )

            history = _prior(
                ts[team_ids == float(team_id)]
                if pd.notna(team_id) and not ts.empty
                else ts.iloc[0:0],
                day,
                30,
            )

            history_7 = history.tail(7)
            history_14 = history.tail(14)
            history_30 = history.tail(30)

            wins_7 = _team_wins(history_7, ts)
            wins_14 = _team_wins(history_14, ts)
            wins_30 = _team_wins(history_30, ts)

            row[f"{side}_win_pct_7"] = (
                float(np.mean(wins_7)) if wins_7 else 0.50
            )
            row[f"{side}_win_pct_14"] = (
                float(np.mean(wins_14)) if wins_14 else 0.50
            )
            row[f"{side}_win_pct_30"] = (
                float(np.mean(wins_30)) if wins_30 else 0.50
            )

            runs_pg_7 = _safe_mean(history_7, "runs", 4.3)
            runs_pg_14 = _safe_mean(history_14, "runs", 4.3)
            runs_pg_30 = _safe_mean(history_30, "runs", 4.3)

            runs_allowed_7 = _runs_allowed(history_7, ts)
            runs_allowed_14 = _runs_allowed(history_14, ts)
            runs_allowed_30 = _runs_allowed(history_30, ts)

            row[f"{side}_runs_pg_7"] = runs_pg_7
            row[f"{side}_runs_pg_14"] = runs_pg_14
            row[f"{side}_runs_pg_30"] = runs_pg_30

            row[f"{side}_runs_allowed_pg_7"] = runs_allowed_7
            row[f"{side}_runs_allowed_pg_14"] = runs_allowed_14
            row[f"{side}_runs_allowed_pg_30"] = runs_allowed_30

            row[f"{side}_run_diff_7"] = runs_pg_7 - runs_allowed_7
            row[f"{side}_run_diff_14"] = runs_pg_14 - runs_allowed_14
            row[f"{side}_run_diff_30"] = runs_pg_30 - runs_allowed_30

            hits = _safe_sum(history_14, "hits")
            walks = _safe_sum(history_14, "walks")
            strikeouts = _safe_sum(history_14, "strikeouts")
            at_bats = _safe_sum(history_14, "at_bats")
            total_bases = _safe_sum(history_14, "total_bases")
            home_runs = _safe_sum(history_14, "home_runs")

            estimated_plate_appearances = at_bats + walks

            batting_avg = _rate(hits, at_bats, 0.245)
            on_base_pct = _rate(
                hits + walks,
                estimated_plate_appearances,
                0.320,
            )
            slugging = _rate(total_bases, at_bats, 0.400)
            isolated_power = slugging - batting_avg

            row[f"{side}_batting_avg_14"] = batting_avg
            row[f"{side}_obp_14"] = on_base_pct
            row[f"{side}_slg_14"] = slugging
            row[f"{side}_ops_14"] = on_base_pct + slugging
            row[f"{side}_iso_14"] = isolated_power
            row[f"{side}_bb_rate_14"] = _rate(
                walks,
                estimated_plate_appearances,
                0.08,
            )
            row[f"{side}_k_rate_14"] = _rate(
                strikeouts,
                estimated_plate_appearances,
                0.22,
            )
            row[f"{side}_hits_pg_14"] = _safe_mean(
                history_14,
                "hits",
                8.3,
            )
            row[f"{side}_hr_pg_14"] = _safe_mean(
                history_14,
                "home_runs",
                1.1,
            )

            history_1 = history.tail(1)
            history_2 = history.tail(2)
            history_3 = history.tail(3)

            row[f"{side}_bullpen_pitches_1"] = _safe_sum(
                history_1,
                "bullpen_pitches",
            )
            row[f"{side}_bullpen_pitches_2"] = _safe_sum(
                history_2,
                "bullpen_pitches",
            )
            row[f"{side}_bullpen_pitches_3"] = _safe_sum(
                history_3,
                "bullpen_pitches",
            )
            row[f"{side}_bullpen_innings_3"] = _safe_sum(
                history_3,
                "bullpen_innings",
            )

            pitcher_id = game.get(f"{side}_probable_pitcher_id")
            starter = _starter_features(ps, pitcher_id, day)

            for feature_name, value in starter.items():
                row[f"{side}_{feature_name}"] = value

            if history.empty:
                row[f"{side}_rest"] = 3.0
            else:
                last_game = pd.to_datetime(
                    history["game_date"],
                    errors="coerce",
                ).max()

                if pd.isna(last_game):
                    row[f"{side}_rest"] = 3.0
                else:
                    days_rest = (pd.Timestamp(day) - last_game).days
                    row[f"{side}_rest"] = float(
                        min(max(days_rest, 0), 7)
                    )

        row["rest_diff"] = row["home_rest"] - row["away_rest"]

        row["park_factor"] = _weather_value(
            game,
            "park_factor",
            1.0,
        )
        row["temperature"] = _weather_value(
            game,
            "temperature",
            72.0,
        )
        row["humidity"] = _weather_value(
            game,
            "humidity",
            50.0,
        )
        row["pressure"] = _weather_value(
            game,
            "pressure",
            1013.0,
        )
        row["wind_speed"] = _weather_value(
            game,
            "wind_speed",
            7.0,
        )
        row["precipitation"] = _weather_value(
            game,
            "precipitation",
            0.0,
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    for feature in FEATURES:
        if feature not in result.columns:
            result[feature] = 0.0

        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce",
        )

    result[FEATURES] = result[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result