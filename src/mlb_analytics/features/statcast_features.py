from __future__ import annotations

import numpy as np
import pandas as pd


BATTER_HIT_STATCAST_FEATURES = [
    "sc_batter_games_30",
    "sc_batter_bbe_30",
    "sc_batter_avg_ev_10",
    "sc_batter_avg_ev_30",
    "sc_batter_hard_hit_rate_10",
    "sc_batter_hard_hit_rate_30",
    "sc_batter_sweet_spot_rate_10",
    "sc_batter_sweet_spot_rate_30",
    "sc_batter_xba_10",
    "sc_batter_xba_30",
    "sc_batter_xwoba_30",
    "sc_batter_whiff_rate_10",
    "sc_batter_whiff_rate_30",
    "sc_batter_chase_rate_30",
    "sc_batter_line_drive_rate_30",
    "sc_opp_games_10",
    "sc_opp_xba_allowed_10",
    "sc_opp_hard_hit_rate_allowed_10",
    "sc_opp_whiff_rate_10",
    "sc_opp_chase_rate_10",
]

BATTER_HR_STATCAST_FEATURES = [
    "sc_batter_games_30",
    "sc_batter_bbe_30",
    "sc_batter_avg_ev_10",
    "sc_batter_avg_ev_30",
    "sc_batter_max_ev_30",
    "sc_batter_hard_hit_rate_10",
    "sc_batter_hard_hit_rate_30",
    "sc_batter_barrel_rate_10",
    "sc_batter_barrel_rate_30",
    "sc_batter_sweet_spot_rate_30",
    "sc_batter_xslg_10",
    "sc_batter_xslg_30",
    "sc_batter_xwoba_30",
    "sc_batter_fly_ball_rate_30",
    "sc_opp_games_10",
    "sc_opp_avg_ev_allowed_10",
    "sc_opp_hard_hit_rate_allowed_10",
    "sc_opp_barrel_rate_allowed_10",
    "sc_opp_xslg_allowed_10",
    "sc_opp_xwoba_allowed_10",
]

PITCHER_K_STATCAST_FEATURES = [
    "sc_pitcher_games_10",
    "sc_pitcher_pitches_5",
    "sc_pitcher_pitches_10",
    "sc_pitcher_whiff_rate_5",
    "sc_pitcher_whiff_rate_10",
    "sc_pitcher_chase_rate_5",
    "sc_pitcher_chase_rate_10",
    "sc_pitcher_csw_rate_5",
    "sc_pitcher_csw_rate_10",
    "sc_pitcher_zone_rate_10",
    "sc_pitcher_avg_velocity_5",
    "sc_pitcher_avg_velocity_10",
    "sc_pitcher_max_velocity_5",
    "sc_pitcher_spin_rate_10",
    "sc_pitcher_xwoba_allowed_10",
    "sc_pitcher_hard_hit_rate_allowed_10",
    "sc_pitcher_barrel_rate_allowed_10",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _sum(frame: pd.DataFrame, column: str) -> float:
    values = _num(frame, column)
    return float(values.fillna(0).sum()) if not values.empty else 0.0


def _mean(frame: pd.DataFrame, column: str, default: float) -> float:
    values = _num(frame, column).dropna()
    return float(values.mean()) if not values.empty else default


def _max(frame: pd.DataFrame, column: str, default: float) -> float:
    values = _num(frame, column).dropna()
    return float(values.max()) if not values.empty else default


def _rate(numerator: float, denominator: float, default: float) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def index_statcast_games(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Index Statcast game aggregates by MLB player id for fast rolling lookups."""
    if frame.empty or "player_id" not in frame.columns:
        return {}
    data = frame.copy()
    data["game_date"] = pd.to_datetime(data.get("game_date"), errors="coerce")
    data["player_id"] = pd.to_numeric(data["player_id"], errors="coerce")
    data = data.dropna(subset=["player_id", "game_date"])
    output: dict[int, pd.DataFrame] = {}
    for player_id, group in data.groupby("player_id", sort=False):
        output[int(player_id)] = group.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    return output


def _player_history(
    frame: pd.DataFrame | dict[int, pd.DataFrame],
    player_id,
    cutoff,
    max_games: int,
) -> pd.DataFrame:
    if pd.isna(player_id):
        return pd.DataFrame()

    if isinstance(frame, dict):
        history = frame.get(int(float(player_id)), pd.DataFrame()).copy()
        if history.empty:
            return history
        dates = pd.to_datetime(history["game_date"], errors="coerce")
        history = history.loc[dates < pd.Timestamp(cutoff)].copy()
        if history.empty:
            return history
        history["_date"] = dates.loc[history.index]
        return history.sort_values(["_date", "game_pk"]).tail(max_games)

    if frame.empty or "player_id" not in frame.columns or "game_date" not in frame.columns:
        return pd.DataFrame()
    ids = pd.to_numeric(frame["player_id"], errors="coerce")
    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    history = frame.loc[
        (ids == float(player_id)) & (dates < pd.Timestamp(cutoff))
    ].copy()
    if history.empty:
        return history
    history["_date"] = dates.loc[history.index]
    return history.sort_values(["_date", "game_pk"]).tail(max_games)


def _contact_metrics(frame: pd.DataFrame) -> dict[str, float]:
    bbe = _sum(frame, "bbe")
    swings = _sum(frame, "swings")
    out_zone = _sum(frame, "out_zone_pitches")
    launch_count = _sum(frame, "launch_speed_count")
    xba_count = _sum(frame, "xba_count")
    xslg_count = _sum(frame, "xslg_count")
    xwoba_count = _sum(frame, "xwoba_count")
    return {
        "games": float(len(frame)),
        "pitches": _sum(frame, "pitches"),
        "bbe": bbe,
        "avg_ev": _rate(_sum(frame, "launch_speed_sum"), launch_count, 88.0),
        "max_ev": _max(frame, "max_exit_velocity", 103.0),
        "avg_launch_angle": _rate(
            _sum(frame, "launch_angle_sum"),
            _sum(frame, "launch_angle_count"),
            12.0,
        ),
        "hard_hit_rate": _rate(_sum(frame, "hard_hits"), bbe, 0.38),
        "barrel_rate": _rate(_sum(frame, "barrels"), bbe, 0.07),
        "sweet_spot_rate": _rate(_sum(frame, "sweet_spot"), bbe, 0.33),
        "ground_ball_rate": _rate(_sum(frame, "ground_balls"), bbe, 0.43),
        "fly_ball_rate": _rate(_sum(frame, "fly_balls"), bbe, 0.24),
        "line_drive_rate": _rate(_sum(frame, "line_drives"), bbe, 0.24),
        "popup_rate": _rate(_sum(frame, "popups"), bbe, 0.09),
        "xba": _rate(_sum(frame, "xba_sum"), xba_count, 0.245),
        "xslg": _rate(_sum(frame, "xslg_sum"), xslg_count, 0.410),
        "xwoba": _rate(_sum(frame, "xwoba_sum"), xwoba_count, 0.320),
        "whiff_rate": _rate(_sum(frame, "whiffs"), swings, 0.245),
        "contact_rate": 1.0 - _rate(_sum(frame, "whiffs"), swings, 0.245),
        "chase_rate": _rate(_sum(frame, "chases"), out_zone, 0.285),
        "csw_rate": _rate(
            _sum(frame, "whiffs") + _sum(frame, "called_strikes"),
            _sum(frame, "pitches"),
            0.275,
        ),
        "zone_rate": _rate(
            _sum(frame, "in_zone_pitches"),
            _sum(frame, "pitches"),
            0.49,
        ),
        "avg_velocity": _rate(
            _sum(frame, "release_speed_sum"),
            _sum(frame, "release_speed_count"),
            92.5,
        ),
        "max_velocity": _max(frame, "max_release_speed", 96.0),
        "spin_rate": _rate(
            _sum(frame, "release_spin_sum"),
            _sum(frame, "release_spin_count"),
            2250.0,
        ),
    }


def batter_statcast_features(
    batter_games: pd.DataFrame | dict[int, pd.DataFrame],
    player_id,
    cutoff,
) -> dict[str, float]:
    history = _player_history(batter_games, player_id, cutoff, 30)
    metrics_10 = _contact_metrics(history.tail(10))
    metrics_30 = _contact_metrics(history.tail(30))
    return {
        "sc_batter_games_30": metrics_30["games"],
        "sc_batter_bbe_30": metrics_30["bbe"],
        "sc_batter_avg_ev_10": metrics_10["avg_ev"],
        "sc_batter_avg_ev_30": metrics_30["avg_ev"],
        "sc_batter_max_ev_30": metrics_30["max_ev"],
        "sc_batter_hard_hit_rate_10": metrics_10["hard_hit_rate"],
        "sc_batter_hard_hit_rate_30": metrics_30["hard_hit_rate"],
        "sc_batter_barrel_rate_10": metrics_10["barrel_rate"],
        "sc_batter_barrel_rate_30": metrics_30["barrel_rate"],
        "sc_batter_sweet_spot_rate_10": metrics_10["sweet_spot_rate"],
        "sc_batter_sweet_spot_rate_30": metrics_30["sweet_spot_rate"],
        "sc_batter_xba_10": metrics_10["xba"],
        "sc_batter_xba_30": metrics_30["xba"],
        "sc_batter_xslg_10": metrics_10["xslg"],
        "sc_batter_xslg_30": metrics_30["xslg"],
        "sc_batter_xwoba_30": metrics_30["xwoba"],
        "sc_batter_whiff_rate_10": metrics_10["whiff_rate"],
        "sc_batter_whiff_rate_30": metrics_30["whiff_rate"],
        "sc_batter_chase_rate_30": metrics_30["chase_rate"],
        "sc_batter_contact_rate_10": metrics_10["contact_rate"],
        "sc_batter_contact_rate_30": metrics_30["contact_rate"],
        "sc_batter_ground_ball_rate_30": metrics_30["ground_ball_rate"],
        "sc_batter_fly_ball_rate_30": metrics_30["fly_ball_rate"],
        "sc_batter_line_drive_rate_30": metrics_30["line_drive_rate"],
    }


def opponent_pitcher_statcast_features(
    pitcher_games: pd.DataFrame | dict[int, pd.DataFrame],
    player_id,
    cutoff,
) -> dict[str, float]:
    history = _player_history(pitcher_games, player_id, cutoff, 10)
    metrics_5 = _contact_metrics(history.tail(5))
    metrics_10 = _contact_metrics(history.tail(10))
    return {
        "sc_opp_games_10": metrics_10["games"],
        "sc_opp_bbe_10": metrics_10["bbe"],
        "sc_opp_avg_ev_allowed_5": metrics_5["avg_ev"],
        "sc_opp_avg_ev_allowed_10": metrics_10["avg_ev"],
        "sc_opp_hard_hit_rate_allowed_5": metrics_5["hard_hit_rate"],
        "sc_opp_hard_hit_rate_allowed_10": metrics_10["hard_hit_rate"],
        "sc_opp_barrel_rate_allowed_5": metrics_5["barrel_rate"],
        "sc_opp_barrel_rate_allowed_10": metrics_10["barrel_rate"],
        "sc_opp_xba_allowed_5": metrics_5["xba"],
        "sc_opp_xba_allowed_10": metrics_10["xba"],
        "sc_opp_xslg_allowed_5": metrics_5["xslg"],
        "sc_opp_xslg_allowed_10": metrics_10["xslg"],
        "sc_opp_xwoba_allowed_5": metrics_5["xwoba"],
        "sc_opp_xwoba_allowed_10": metrics_10["xwoba"],
        "sc_opp_whiff_rate_5": metrics_5["whiff_rate"],
        "sc_opp_whiff_rate_10": metrics_10["whiff_rate"],
        "sc_opp_chase_rate_5": metrics_5["chase_rate"],
        "sc_opp_chase_rate_10": metrics_10["chase_rate"],
        "sc_opp_csw_rate_5": metrics_5["csw_rate"],
        "sc_opp_csw_rate_10": metrics_10["csw_rate"],
        "sc_opp_avg_velocity_5": metrics_5["avg_velocity"],
        "sc_opp_avg_velocity_10": metrics_10["avg_velocity"],
    }


def pitcher_statcast_features(
    pitcher_games: pd.DataFrame | dict[int, pd.DataFrame],
    player_id,
    cutoff,
) -> dict[str, float]:
    history = _player_history(pitcher_games, player_id, cutoff, 10)
    metrics_5 = _contact_metrics(history.tail(5))
    metrics_10 = _contact_metrics(history.tail(10))
    starts_5 = max(metrics_5["games"], 1.0)
    starts_10 = max(metrics_10["games"], 1.0)
    return {
        "sc_pitcher_games_10": metrics_10["games"],
        "sc_pitcher_pitches_5": metrics_5["pitches"] / starts_5,
        "sc_pitcher_pitches_10": metrics_10["pitches"] / starts_10,
        "sc_pitcher_whiff_rate_5": metrics_5["whiff_rate"],
        "sc_pitcher_whiff_rate_10": metrics_10["whiff_rate"],
        "sc_pitcher_chase_rate_5": metrics_5["chase_rate"],
        "sc_pitcher_chase_rate_10": metrics_10["chase_rate"],
        "sc_pitcher_csw_rate_5": metrics_5["csw_rate"],
        "sc_pitcher_csw_rate_10": metrics_10["csw_rate"],
        "sc_pitcher_zone_rate_10": metrics_10["zone_rate"],
        "sc_pitcher_avg_velocity_5": metrics_5["avg_velocity"],
        "sc_pitcher_avg_velocity_10": metrics_10["avg_velocity"],
        "sc_pitcher_max_velocity_5": metrics_5["max_velocity"],
        "sc_pitcher_spin_rate_10": metrics_10["spin_rate"],
        "sc_pitcher_xwoba_allowed_10": metrics_10["xwoba"],
        "sc_pitcher_hard_hit_rate_allowed_10": metrics_10["hard_hit_rate"],
        "sc_pitcher_barrel_rate_allowed_10": metrics_10["barrel_rate"],
    }


def ensure_numeric_features(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    for feature in features:
        if feature not in output.columns:
            output[feature] = np.nan
        output[feature] = pd.to_numeric(output[feature], errors="coerce")
    output[features] = output[features].replace([np.inf, -np.inf], np.nan)
    return output
