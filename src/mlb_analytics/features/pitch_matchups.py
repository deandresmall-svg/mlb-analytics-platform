from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd


BATTER_HIT_PITCH_MATCHUP_FEATURES = [
    "pt_matchup_xba",
    "pt_matchup_xwoba",
    "pt_matchup_whiff_rate",
    "pt_matchup_contact_rate",
    "pt_matchup_chase_rate",
    "pt_matchup_hard_hit_rate",
    "pt_pitch_mix_coverage",
    "pt_primary_pitch_usage",
    "pt_batter_sample_pitches",
    "pt_pitcher_sample_pitches",
]

BATTER_HR_PITCH_MATCHUP_FEATURES = [
    "pt_matchup_xslg",
    "pt_matchup_xwoba",
    "pt_matchup_hard_hit_rate",
    "pt_matchup_barrel_rate",
    "pt_matchup_fly_ball_rate",
    "pt_matchup_whiff_rate",
    "pt_pitch_mix_coverage",
    "pt_primary_pitch_usage",
    "pt_batter_sample_pitches",
    "pt_pitcher_sample_pitches",
]

PITCHER_K_LINEUP_MATCHUP_FEATURES = [
    "pt_lineup_whiff_rate",
    "pt_lineup_contact_rate",
    "pt_lineup_chase_rate",
    "pt_lineup_xwoba",
    "pt_lineup_coverage",
    "pt_lineup_hitters",
    "pt_primary_pitch_usage",
    "pt_pitcher_mix_pitches",
]


_BASELINES = {
    "xba": 0.245,
    "xslg": 0.410,
    "xwoba": 0.320,
    "whiff_rate": 0.245,
    "contact_rate": 0.755,
    "chase_rate": 0.285,
    "hard_hit_rate": 0.380,
    "barrel_rate": 0.070,
    "fly_ball_rate": 0.240,
}


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _sum(frame: pd.DataFrame, column: str) -> float:
    values = _num(frame, column)
    return float(values.fillna(0).sum()) if not values.empty else 0.0


def _rate(numerator: float, denominator: float, default: float) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def _mode(frame: pd.DataFrame, column: str, default: str = "U") -> str:
    if frame.empty or column not in frame.columns:
        return default
    values = frame[column].dropna().astype(str)
    values = values[~values.isin(["", "nan", "None", "U"])]
    if values.empty:
        return default
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else default


def _scale(value: float, low: float, high: float, invert: bool = False) -> float:
    if not isfinite(value) or high <= low:
        return 50.0
    score = max(0.0, min(100.0, 100.0 * (value - low) / (high - low)))
    return 100.0 - score if invert else score




class BattingTeamIndex(dict[int, pd.DataFrame]):
    def __init__(self) -> None:
        super().__init__()
        self.lineup_cache: dict[tuple[int, str, int], list[int]] = {}


def index_batting_teams(frame: pd.DataFrame) -> BattingTeamIndex:
    output = BattingTeamIndex()
    if frame.empty or "team_id" not in frame.columns:
        return output
    data = frame.copy()
    data["team_id"] = pd.to_numeric(data["team_id"], errors="coerce")
    data["game_date"] = pd.to_datetime(data.get("game_date"), errors="coerce")
    data = data.dropna(subset=["team_id", "game_date", "player_id"])
    for team_id, group in data.groupby("team_id", sort=False):
        output[int(team_id)] = group.sort_values(
            ["game_date", "game_pk"]
        ).reset_index(drop=True)
    return output

def index_pitch_type_games(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Index pitch-type game rows by MLB player id."""
    if frame.empty or "player_id" not in frame.columns:
        return {}
    data = frame.copy()
    data["player_id"] = pd.to_numeric(data["player_id"], errors="coerce")
    data["game_date"] = pd.to_datetime(data.get("game_date"), errors="coerce")
    data = data.dropna(subset=["player_id", "game_date"])
    output: dict[int, pd.DataFrame] = {}
    for player_id, group in data.groupby("player_id", sort=False):
        output[int(player_id)] = group.sort_values(
            ["game_date", "game_pk", "pitch_type"]
        ).reset_index(drop=True)
    return output


def _history(
    index: dict[int, pd.DataFrame],
    player_id,
    cutoff,
    max_games: int,
) -> pd.DataFrame:
    if pd.isna(player_id):
        return pd.DataFrame()
    try:
        history = index.get(int(float(player_id)), pd.DataFrame()).copy()
    except (TypeError, ValueError):
        return pd.DataFrame()
    if history.empty:
        return history
    dates = pd.to_datetime(history["game_date"], errors="coerce")
    history = history.loc[dates < pd.Timestamp(cutoff)].copy()
    if history.empty:
        return history
    history["_date"] = dates.loc[history.index]
    game_order = (
        history[["game_pk", "_date"]]
        .drop_duplicates()
        .sort_values(["_date", "game_pk"])
        .tail(max_games)["game_pk"]
    )
    return history[history["game_pk"].isin(game_order)].copy()


def _aggregate_pitch_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    for pitch_type, group in frame.groupby("pitch_type", dropna=False):
        pitches = _sum(group, "pitches")
        swings = _sum(group, "swings")
        bbe = _sum(group, "bbe")
        out_zone = _sum(group, "out_zone_pitches")
        xba_count = _sum(group, "xba_count")
        xslg_count = _sum(group, "xslg_count")
        xwoba_count = _sum(group, "xwoba_count")
        rows.append(
            {
                "pitch_type": str(pitch_type or "UN"),
                "pitches": pitches,
                "swings": swings,
                "bbe": bbe,
                "xba": _rate(_sum(group, "xba_sum"), xba_count, _BASELINES["xba"]),
                "xslg": _rate(
                    _sum(group, "xslg_sum"), xslg_count, _BASELINES["xslg"]
                ),
                "xwoba": _rate(
                    _sum(group, "xwoba_sum"), xwoba_count, _BASELINES["xwoba"]
                ),
                "whiff_rate": _rate(
                    _sum(group, "whiffs"), swings, _BASELINES["whiff_rate"]
                ),
                "contact_rate": 1.0
                - _rate(_sum(group, "whiffs"), swings, _BASELINES["whiff_rate"]),
                "chase_rate": _rate(
                    _sum(group, "chases"), out_zone, _BASELINES["chase_rate"]
                ),
                "hard_hit_rate": _rate(
                    _sum(group, "hard_hits"), bbe, _BASELINES["hard_hit_rate"]
                ),
                "barrel_rate": _rate(
                    _sum(group, "barrels"), bbe, _BASELINES["barrel_rate"]
                ),
                "fly_ball_rate": _rate(
                    _sum(group, "fly_balls"), bbe, _BASELINES["fly_ball_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _shrink(value: float, sample: float, baseline: float, full_sample: float) -> float:
    weight = max(0.0, min(1.0, sample / full_sample))
    return weight * value + (1.0 - weight) * baseline


def _blend_metric(
    batter_row: pd.Series | None,
    pitcher_row: pd.Series | None,
    metric: str,
) -> float:
    baseline = _BASELINES[metric]
    if batter_row is None:
        batter_value = baseline
        batter_sample = 0.0
    else:
        batter_sample = float(batter_row.get("pitches", 0.0) or 0.0)
        batter_value = _shrink(
            float(batter_row.get(metric, baseline)), batter_sample, baseline, 120.0
        )
    if pitcher_row is None:
        pitcher_value = baseline
        pitcher_sample = 0.0
    else:
        pitcher_sample = float(pitcher_row.get("pitches", 0.0) or 0.0)
        pitcher_value = _shrink(
            float(pitcher_row.get(metric, baseline)), pitcher_sample, baseline, 160.0
        )

    total_sample = batter_sample + pitcher_sample
    if total_sample <= 0:
        return baseline
    batter_weight = 0.55 if batter_sample > 0 else 0.0
    pitcher_weight = 0.45 if pitcher_sample > 0 else 0.0
    total_weight = batter_weight + pitcher_weight
    return (
        batter_value * batter_weight + pitcher_value * pitcher_weight
    ) / total_weight


def batter_vs_pitch_mix_features(
    batter_index: dict[int, pd.DataFrame],
    pitcher_index: dict[int, pd.DataFrame],
    batter_id,
    pitcher_id,
    cutoff,
) -> dict[str, float | str]:
    """Build point-in-time batter-vs-arsenal features with sample shrinkage."""
    defaults: dict[str, float | str] = {
        "pt_matchup_xba": _BASELINES["xba"],
        "pt_matchup_xslg": _BASELINES["xslg"],
        "pt_matchup_xwoba": _BASELINES["xwoba"],
        "pt_matchup_whiff_rate": _BASELINES["whiff_rate"],
        "pt_matchup_contact_rate": _BASELINES["contact_rate"],
        "pt_matchup_chase_rate": _BASELINES["chase_rate"],
        "pt_matchup_hard_hit_rate": _BASELINES["hard_hit_rate"],
        "pt_matchup_barrel_rate": _BASELINES["barrel_rate"],
        "pt_matchup_fly_ball_rate": _BASELINES["fly_ball_rate"],
        "pt_pitch_mix_coverage": 0.0,
        "pt_primary_pitch_usage": 0.0,
        "pt_batter_sample_pitches": 0.0,
        "pt_pitcher_sample_pitches": 0.0,
        "pt_primary_pitch": "N/A",
        "pt_pitcher_hand": "U",
        "pt_batter_hand": "U",
        "pt_matchup_score": 50.0,
    }

    batter_history = _history(batter_index, batter_id, cutoff, 30)
    pitcher_history = _history(pitcher_index, pitcher_id, cutoff, 10)
    if pitcher_history.empty:
        return defaults

    batter_hand = _mode(batter_history, "player_hand")
    pitcher_hand = _mode(pitcher_history, "player_hand")

    pitcher_split = pitcher_history
    if batter_hand != "U" and "opponent_hand" in pitcher_history.columns:
        split = pitcher_history[
            pitcher_history["opponent_hand"].astype(str).eq(batter_hand)
        ]
        if _sum(split, "pitches") >= 80:
            pitcher_split = split

    batter_split = batter_history
    if pitcher_hand != "U" and "opponent_hand" in batter_history.columns:
        split = batter_history[
            batter_history["opponent_hand"].astype(str).eq(pitcher_hand)
        ]
        if _sum(split, "pitches") >= 60:
            batter_split = split

    pitcher_by_type = _aggregate_pitch_types(pitcher_split)
    batter_by_type = _aggregate_pitch_types(batter_split)
    if pitcher_by_type.empty:
        return defaults

    total_pitcher_pitches = float(pitcher_by_type["pitches"].sum())
    if total_pitcher_pitches <= 0:
        return defaults

    batter_lookup = {
        str(row["pitch_type"]): row for _, row in batter_by_type.iterrows()
    }
    pitcher_lookup = {
        str(row["pitch_type"]): row for _, row in pitcher_by_type.iterrows()
    }

    weighted = {metric: 0.0 for metric in _BASELINES}
    coverage = 0.0
    weighted_batter_sample = 0.0
    primary_row = pitcher_by_type.sort_values("pitches", ascending=False).iloc[0]

    for _, pitcher_row in pitcher_by_type.iterrows():
        pitch_type = str(pitcher_row["pitch_type"])
        usage = float(pitcher_row["pitches"]) / total_pitcher_pitches
        batter_row = batter_lookup.get(pitch_type)
        for metric in weighted:
            weighted[metric] += usage * _blend_metric(
                batter_row, pitcher_row, metric
            )
        batter_sample = (
            float(batter_row.get("pitches", 0.0)) if batter_row is not None else 0.0
        )
        weighted_batter_sample += usage * batter_sample
        if batter_sample >= 30 and float(pitcher_row.get("pitches", 0.0)) >= 40:
            coverage += usage

    score = (
        _scale(weighted["xba"], 0.190, 0.330) * 0.24
        + _scale(weighted["xslg"], 0.280, 0.620) * 0.22
        + _scale(weighted["xwoba"], 0.250, 0.410) * 0.18
        + _scale(weighted["hard_hit_rate"], 0.25, 0.55) * 0.14
        + _scale(weighted["barrel_rate"], 0.02, 0.15) * 0.10
        + _scale(weighted["whiff_rate"], 0.15, 0.36, invert=True) * 0.12
    )

    return {
        "pt_matchup_xba": weighted["xba"],
        "pt_matchup_xslg": weighted["xslg"],
        "pt_matchup_xwoba": weighted["xwoba"],
        "pt_matchup_whiff_rate": weighted["whiff_rate"],
        "pt_matchup_contact_rate": weighted["contact_rate"],
        "pt_matchup_chase_rate": weighted["chase_rate"],
        "pt_matchup_hard_hit_rate": weighted["hard_hit_rate"],
        "pt_matchup_barrel_rate": weighted["barrel_rate"],
        "pt_matchup_fly_ball_rate": weighted["fly_ball_rate"],
        "pt_pitch_mix_coverage": min(max(coverage, 0.0), 1.0),
        "pt_primary_pitch_usage": float(primary_row["pitches"])
        / total_pitcher_pitches,
        "pt_batter_sample_pitches": weighted_batter_sample,
        "pt_pitcher_sample_pitches": total_pitcher_pitches,
        "pt_primary_pitch": str(primary_row["pitch_type"]),
        "pt_pitcher_hand": pitcher_hand,
        "pt_batter_hand": batter_hand,
        "pt_matchup_score": round(score, 1),
    }


def likely_team_hitters(
    batting: pd.DataFrame | BattingTeamIndex,
    team_id,
    cutoff,
    max_players: int = 9,
) -> list[int]:
    if pd.isna(team_id):
        return []
    try:
        team_key = int(float(team_id))
    except (TypeError, ValueError):
        return []
    cutoff_key = pd.Timestamp(cutoff).date().isoformat()
    if isinstance(batting, BattingTeamIndex):
        cache_key = (team_key, cutoff_key, max_players)
        if cache_key in batting.lineup_cache:
            return list(batting.lineup_cache[cache_key])
        data = batting.get(team_key, pd.DataFrame())
    else:
        if batting.empty:
            return []
        data = batting.copy()
        data["game_date"] = pd.to_datetime(data.get("game_date"), errors="coerce")
        ids = pd.to_numeric(data.get("team_id"), errors="coerce")
        data = data[ids == float(team_key)]
    if data.empty:
        return []
    dates = pd.to_datetime(data.get("game_date"), errors="coerce")
    history = data[dates < pd.Timestamp(cutoff)]
    if history.empty:
        return []
    candidates: list[tuple[pd.Timestamp, float, int]] = []
    for player_id, group in history.groupby("player_id"):
        group = group.sort_values(["game_date", "game_pk"]).tail(10)
        if len(group) < 3:
            continue
        last_game = pd.Timestamp(group["game_date"].max())
        recent_pa = _sum(group, "plate_appearances")
        candidates.append((last_game, recent_pa, int(float(player_id))))
    candidates.sort(reverse=True)
    result = [player_id for _, _, player_id in candidates[:max_players]]
    if isinstance(batting, BattingTeamIndex):
        batting.lineup_cache[(team_key, cutoff_key, max_players)] = list(result)
    return result

def pitcher_vs_lineup_pitch_mix_features(
    batter_index: dict[int, pd.DataFrame],
    pitcher_index: dict[int, pd.DataFrame],
    batting: pd.DataFrame | BattingTeamIndex,
    pitcher_id,
    opponent_team_id,
    cutoff,
) -> dict[str, float | str]:
    defaults: dict[str, float | str] = {
        "pt_lineup_whiff_rate": _BASELINES["whiff_rate"],
        "pt_lineup_contact_rate": _BASELINES["contact_rate"],
        "pt_lineup_chase_rate": _BASELINES["chase_rate"],
        "pt_lineup_xwoba": _BASELINES["xwoba"],
        "pt_lineup_coverage": 0.0,
        "pt_lineup_hitters": 0.0,
        "pt_primary_pitch_usage": 0.0,
        "pt_pitcher_mix_pitches": 0.0,
        "pt_primary_pitch": "N/A",
    }
    try:
        pitcher_key = int(float(pitcher_id))
    except (TypeError, ValueError):
        return defaults
    if pitcher_key not in pitcher_index or pitcher_index[pitcher_key].empty:
        return defaults
    hitter_ids = likely_team_hitters(batting, opponent_team_id, cutoff)
    if not hitter_ids:
        return defaults

    matchups = [
        batter_vs_pitch_mix_features(
            batter_index, pitcher_index, hitter_id, pitcher_id, cutoff
        )
        for hitter_id in hitter_ids
    ]
    if not matchups:
        return defaults

    frame = pd.DataFrame(matchups)
    weights = pd.to_numeric(frame["pt_pitch_mix_coverage"], errors="coerce").fillna(0.0)
    weights = 0.35 + 0.65 * weights
    total_weight = float(weights.sum())

    def weighted_mean(column: str, default: float) -> float:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(default)
        return float((values * weights).sum() / total_weight) if total_weight else default

    primary = frame["pt_primary_pitch"].mode()
    return {
        "pt_lineup_whiff_rate": weighted_mean(
            "pt_matchup_whiff_rate", _BASELINES["whiff_rate"]
        ),
        "pt_lineup_contact_rate": weighted_mean(
            "pt_matchup_contact_rate", _BASELINES["contact_rate"]
        ),
        "pt_lineup_chase_rate": weighted_mean(
            "pt_matchup_chase_rate", _BASELINES["chase_rate"]
        ),
        "pt_lineup_xwoba": weighted_mean("pt_matchup_xwoba", _BASELINES["xwoba"]),
        "pt_lineup_coverage": float(
            pd.to_numeric(frame["pt_pitch_mix_coverage"], errors="coerce")
            .fillna(0.0)
            .mean()
        ),
        "pt_lineup_hitters": float(len(frame)),
        "pt_primary_pitch_usage": weighted_mean("pt_primary_pitch_usage", 0.0),
        "pt_pitcher_mix_pitches": weighted_mean("pt_pitcher_sample_pitches", 0.0),
        "pt_primary_pitch": str(primary.iloc[0]) if not primary.empty else "N/A",
    }
