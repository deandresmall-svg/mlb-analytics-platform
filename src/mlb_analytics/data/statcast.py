from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests


RAW_COLUMNS = [
    "game_pk",
    "game_date",
    "game_type",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "player_name",
    "stand",
    "p_throws",
    "events",
    "description",
    "type",
    "pitch_type",
    "pitch_name",
    "zone",
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "bb_type",
    "hc_x",
    "hc_y",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "launch_speed_angle",
]

INTEGER_COLUMNS = [
    "game_pk",
    "inning",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "zone",
    "launch_speed_angle",
]

NUMERIC_COLUMNS = [
    "plate_x",
    "plate_z",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
]

SWING_DESCRIPTIONS = {
    "foul",
    "foul_bunt",
    "foul_pitchout",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
    "swinging_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
}
WHIFF_DESCRIPTIONS = {
    "missed_bunt",
    "swinging_pitchout",
    "swinging_strike",
    "swinging_strike_blocked",
}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}


class StatcastError(RuntimeError):
    """Raised when a Baseball Savant Statcast download fails."""


@dataclass(frozen=True)
class StatcastChunk:
    start: date
    end: date
    rows: int


class StatcastClient:
    """Download pitch-level Statcast CSV data from Baseball Savant.

    The public CSV search can be slow for long ranges, so callers should request
    small chunks and persist each successful chunk before continuing.
    """

    def __init__(
        self,
        base_url: str = "https://baseballsavant.mlb.com/statcast_search/csv",
        timeout: float = 90.0,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.retries = max(int(retries), 1)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "text/csv,*/*;q=0.8",
                "User-Agent": "mlb-github-streamlit-analytics/2.0",
            }
        )

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        if end < start:
            raise ValueError("Statcast end date cannot be before start date")

        params = {
            "all": "true",
            "type": "details",
            "player_type": "batter",
            "game_date_gt": start.isoformat(),
            "game_date_lt": end.isoformat(),
            "hfGT": "R|",
            "min_pitches": "0",
            "min_results": "0",
            "min_pas": "0",
            "sort_col": "pitches",
            "player_event_sort": "api_p_release_speed",
            "sort_order": "desc",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                text = response.text.strip()
                if not text:
                    return pd.DataFrame(columns=RAW_COLUMNS)
                if text.lstrip().startswith("<"):
                    raise StatcastError(
                        "Baseball Savant returned HTML instead of CSV. "
                        "Retry with a smaller date range."
                    )
                frame = pd.read_csv(StringIO(text), low_memory=False)
                return prepare_statcast(frame)
            except (requests.RequestException, pd.errors.ParserError, StatcastError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))

        raise StatcastError(
            f"Statcast download failed for {start} through {end}: {last_error}"
        )


def _series(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _mode_or_default(series: pd.Series, default: str = "U") -> str:
    values = series.dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return default
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else str(values.iloc[-1])



def _flag(values: pd.Series) -> pd.Series:
    """Convert a nullable boolean series into a compact 0/1 integer flag."""
    return values.astype("boolean").fillna(False).astype("int8")


def recompute_statcast_flags(data: pd.DataFrame) -> pd.DataFrame:
    """Rebuild all reusable pitch flags from the raw Savant columns.

    Baseball Savant can publish exit velocity on foul contact. Those rows are
    not official batted-ball events and must not enter BBE, hard-hit, barrel,
    sweet-spot, or batted-ball-type denominators. A measured BBE therefore
    requires both ``type == "X"`` (ball put into play) and a non-null launch
    speed.

    This function is also used when rebuilding aggregates from stored pitches,
    so historical rows downloaded before this fix are corrected without having
    to download them again.
    """
    output = data.copy()

    descriptions = output["description"].fillna("").astype(str).str.lower()
    pitch_result = output["type"].fillna("").astype(str).str.upper()
    zone = pd.to_numeric(output["zone"], errors="coerce")
    launch_speed = pd.to_numeric(output["launch_speed"], errors="coerce")
    launch_angle = pd.to_numeric(output["launch_angle"], errors="coerce")
    launch_speed_angle = pd.to_numeric(
        output["launch_speed_angle"], errors="coerce"
    )
    bb_type = output["bb_type"].fillna("").astype(str).str.lower()

    is_swing = descriptions.isin(SWING_DESCRIPTIONS)
    is_whiff = descriptions.isin(WHIFF_DESCRIPTIONS)
    is_called_strike = descriptions.isin(CALLED_STRIKE_DESCRIPTIONS)
    is_in_zone = zone.between(1, 9, inclusive="both")
    is_out_zone = zone.notna() & ~is_in_zone

    # Statcast's pitch-result code X means the ball was put into play. Requiring
    # a measured launch speed keeps the denominator aligned with the contact
    # metrics used by Baseball Savant and excludes tracked foul-ball contact.
    is_bbe = pitch_result.eq("X") & launch_speed.notna()

    output["is_swing"] = _flag(is_swing)
    output["is_whiff"] = _flag(is_whiff)
    output["is_called_strike"] = _flag(is_called_strike)
    output["is_in_zone"] = _flag(is_in_zone)
    output["is_out_zone"] = _flag(is_out_zone)
    output["is_chase"] = _flag(is_swing & is_out_zone)
    output["is_bbe"] = _flag(is_bbe)
    output["is_hard_hit"] = _flag(is_bbe & launch_speed.ge(95.0))
    output["is_barrel"] = _flag(is_bbe & launch_speed_angle.eq(6))
    output["is_sweet_spot"] = _flag(
        is_bbe & launch_angle.between(8.0, 32.0, inclusive="both")
    )
    output["is_ground_ball"] = _flag(is_bbe & bb_type.eq("ground_ball"))
    output["is_fly_ball"] = _flag(is_bbe & bb_type.eq("fly_ball"))
    output["is_line_drive"] = _flag(is_bbe & bb_type.eq("line_drive"))
    output["is_popup"] = _flag(is_bbe & bb_type.eq("popup"))

    return output

def prepare_statcast(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Baseball Savant CSV and add reusable pitch flags."""
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    data = frame.copy()
    if "hit_distance_sc" not in data.columns and "hit_distance" in data.columns:
        data["hit_distance_sc"] = data["hit_distance"]
    if "release_spin_rate" not in data.columns and "release_spin" in data.columns:
        data["release_spin_rate"] = data["release_spin"]

    for column in RAW_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[RAW_COLUMNS].copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce").dt.date.astype("string")

    for column in INTEGER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in [
        "game_type",
        "home_team",
        "away_team",
        "inning_topbot",
        "player_name",
        "stand",
        "p_throws",
        "events",
        "description",
        "type",
        "pitch_type",
        "pitch_name",
        "bb_type",
    ]:
        data[column] = data[column].astype("string")

    data = data.dropna(
        subset=["game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher"]
    ).copy()

    data = recompute_statcast_flags(data)

    for column in ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]:
        data[column] = data[column].astype(int)

    data["pitch_type"] = data["pitch_type"].fillna("UN")
    data["stand"] = data["stand"].fillna("U")
    data["p_throws"] = data["p_throws"].fillna("U")

    return data.reset_index(drop=True)


def _safe_sum(group: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(group[column], errors="coerce").fillna(0).sum())


def _safe_count(group: pd.DataFrame, column: str) -> int:
    return int(pd.to_numeric(group[column], errors="coerce").notna().sum())


def _safe_max(group: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _aggregate_common(group: pd.DataFrame) -> dict[str, float | int | None]:
    # Contact-quality and expected-stat fields are meaningful only for measured
    # balls put into play. Restricting these calculations to BBE rows prevents
    # foul contact with tracked exit velocity from inflating denominators.
    bbe_group = group.loc[pd.to_numeric(group["is_bbe"], errors="coerce").fillna(0).eq(1)]

    return {
        "pitches": int(len(group)),
        "plate_appearances": int(group["at_bat_number"].nunique()),
        "swings": int(group["is_swing"].sum()),
        "whiffs": int(group["is_whiff"].sum()),
        "called_strikes": int(group["is_called_strike"].sum()),
        "in_zone_pitches": int(group["is_in_zone"].sum()),
        "out_zone_pitches": int(group["is_out_zone"].sum()),
        "chases": int(group["is_chase"].sum()),
        "bbe": int(group["is_bbe"].sum()),
        "launch_speed_sum": _safe_sum(bbe_group, "launch_speed"),
        "launch_speed_count": _safe_count(bbe_group, "launch_speed"),
        "max_exit_velocity": _safe_max(bbe_group, "launch_speed"),
        "launch_angle_sum": _safe_sum(bbe_group, "launch_angle"),
        "launch_angle_count": _safe_count(bbe_group, "launch_angle"),
        "hard_hits": int(group["is_hard_hit"].sum()),
        "barrels": int(group["is_barrel"].sum()),
        "sweet_spot": int(group["is_sweet_spot"].sum()),
        "ground_balls": int(group["is_ground_ball"].sum()),
        "fly_balls": int(group["is_fly_ball"].sum()),
        "line_drives": int(group["is_line_drive"].sum()),
        "popups": int(group["is_popup"].sum()),
        "xba_sum": _safe_sum(bbe_group, "estimated_ba_using_speedangle"),
        "xba_count": _safe_count(bbe_group, "estimated_ba_using_speedangle"),
        "xslg_sum": _safe_sum(bbe_group, "estimated_slg_using_speedangle"),
        "xslg_count": _safe_count(bbe_group, "estimated_slg_using_speedangle"),
        "xwoba_sum": _safe_sum(bbe_group, "estimated_woba_using_speedangle"),
        "xwoba_count": _safe_count(bbe_group, "estimated_woba_using_speedangle"),
        "release_speed_sum": _safe_sum(group, "release_speed"),
        "release_speed_count": _safe_count(group, "release_speed"),
        "max_release_speed": _safe_max(group, "release_speed"),
        "release_spin_sum": _safe_sum(group, "release_spin_rate"),
        "release_spin_count": _safe_count(group, "release_spin_rate"),
        "pfx_x_sum": _safe_sum(group, "pfx_x"),
        "pfx_z_sum": _safe_sum(group, "pfx_z"),
        "movement_count": min(_safe_count(group, "pfx_x"), _safe_count(group, "pfx_z")),
    }


def _aggregate_groups(
    data: pd.DataFrame,
    keys: list[str],
    id_column: str,
    hand_column: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for key_values, group in data.groupby(keys, dropna=False, sort=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(keys, key_values))
        row["player_id"] = int(row.pop(id_column))
        row["player_hand"] = _mode_or_default(group[hand_column])
        row.update(_aggregate_common(group))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_statcast(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build daily player and pitch-type aggregates from normalized pitches."""
    if data.empty:
        return {
            "batters": pd.DataFrame(),
            "pitchers": pd.DataFrame(),
            "batter_pitch_types": pd.DataFrame(),
            "pitcher_pitch_types": pd.DataFrame(),
        }

    # Recompute flags here as well so aggregate rebuilds correct historical
    # rows that were stored before the BBE definition was tightened.
    data = recompute_statcast_flags(data)

    batters = _aggregate_groups(
        data,
        ["game_pk", "game_date", "batter"],
        id_column="batter",
        hand_column="stand",
    )
    pitchers = _aggregate_groups(
        data,
        ["game_pk", "game_date", "pitcher"],
        id_column="pitcher",
        hand_column="p_throws",
    )

    batter_pitch_types = _aggregate_groups(
        data,
        ["game_pk", "game_date", "batter", "pitch_type", "p_throws"],
        id_column="batter",
        hand_column="stand",
    ).rename(columns={"p_throws": "opponent_hand"})

    pitcher_pitch_types = _aggregate_groups(
        data,
        ["game_pk", "game_date", "pitcher", "pitch_type", "stand"],
        id_column="pitcher",
        hand_column="p_throws",
    ).rename(columns={"stand": "opponent_hand"})

    for frame in [batters, pitchers, batter_pitch_types, pitcher_pitch_types]:
        if not frame.empty:
            frame["game_date"] = frame["game_date"].astype(str)

    return {
        "batters": batters,
        "pitchers": pitchers,
        "batter_pitch_types": batter_pitch_types,
        "pitcher_pitch_types": pitcher_pitch_types,
    }


def chunk_ranges(start: date, end: date, days: int) -> list[tuple[date, date]]:
    from datetime import timedelta

    days = max(int(days), 1)
    ranges: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return ranges
