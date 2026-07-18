from __future__ import annotations

import numpy as np
import pandas as pd


HIT_FEATURES = [
    "hit_rate_5",
    "hit_rate_10",
    "hit_rate_20",
    "hit_rate_30",
    "multi_hit_rate_10",
    "multi_hit_rate_30",
    "batting_avg_5",
    "batting_avg_10",
    "batting_avg_30",
    "obp_10",
    "obp_30",
    "slg_10",
    "slg_30",
    "ops_10",
    "ops_30",
    "iso_30",
    "pa_avg_5",
    "pa_avg_10",
    "pa_avg_30",
    "ab_avg_10",
    "games_with_4_pa_rate_10",
    "strikeout_rate_10",
    "strikeout_rate_30",
    "walk_rate_10",
    "walk_rate_30",
    "balls_in_play_rate_10",
    "balls_in_play_rate_30",
    "hits_per_game_5",
    "hits_per_game_10",
    "hits_per_game_30",
    "total_bases_per_game_10",
    "total_bases_per_game_30",
    "hit_rate_trend_5_vs_30",
    "batting_avg_trend_10_vs_30",
    "pa_trend_5_vs_30",
    "games_prior",

    # New lineup and matchup features
    "batting_order",
    "top_4_lineup",
    "projected_pa",
    "is_home",
    "opponent_sp_era_5",
    "opponent_sp_whip_5",
    "opponent_sp_k_per_9_5",
    "opponent_sp_bb_per_9_5",
    "opponent_sp_k_minus_bb_rate_5",
    "opponent_sp_hr_per_9_5",
    "opponent_sp_hits_per_9_5",
    "opponent_sp_pitches_5",
    "opponent_sp_innings_5",
]


HR_FEATURES = [
    "home_run_rate_5",
    "home_run_rate_10",
    "home_run_rate_20",
    "home_run_rate_30",
    "home_runs_per_pa_10",
    "home_runs_per_pa_30",
    "slugging_10",
    "slugging_30",
    "iso_10",
    "iso_30",
    "total_bases_per_game_10",
    "total_bases_per_game_30",
    "extra_base_hit_rate_10",
    "extra_base_hit_rate_30",
    "batting_avg_10",
    "batting_avg_30",
    "pa_avg_5",
    "pa_avg_10",
    "pa_avg_30",
    "games_with_4_pa_rate_10",
    "strikeout_rate_10",
    "strikeout_rate_30",
    "balls_in_play_rate_10",
    "balls_in_play_rate_30",
    "home_run_rate_trend_5_vs_30",
    "iso_trend_10_vs_30",
    "slugging_trend_10_vs_30",
    "games_since_last_hr",
    "games_prior",

    # New lineup and matchup features
    "batting_order",
    "top_4_lineup",
    "projected_pa",
    "is_home",
    "opponent_sp_era_5",
    "opponent_sp_whip_5",
    "opponent_sp_k_per_9_5",
    "opponent_sp_bb_per_9_5",
    "opponent_sp_k_minus_bb_rate_5",
    "opponent_sp_hr_per_9_5",
    "opponent_sp_hits_per_9_5",
    "opponent_sp_pitches_5",
    "opponent_sp_innings_5",
]


K_FEATURES = [
    "pitcher_k_per_ip_3",
    "pitcher_k_per_ip_5",
    "pitcher_k_per_ip_10",
    "pitcher_k_per_9_5",
    "pitcher_k_per_9_10",
    "pitcher_k_rate_5",
    "pitcher_k_rate_10",
    "pitcher_pitches_3",
    "pitcher_pitches_5",
    "pitcher_pitches_10",
    "pitcher_ip_3",
    "pitcher_ip_5",
    "pitcher_ip_10",
    "pitcher_batters_faced_5",
    "pitcher_batters_faced_10",
    "pitcher_bb_per_9_5",
    "pitcher_bb_per_9_10",
    "pitcher_k_minus_bb_rate_5",
    "pitcher_k_minus_bb_rate_10",
    "pitcher_pitches_per_ip_5",
    "pitcher_pitches_per_batter_5",
    "pitcher_k_avg_3",
    "pitcher_k_avg_5",
    "pitcher_k_avg_10",
    "pitcher_k_trend_3_vs_10",
    "pitcher_ip_trend_3_vs_10",
    "pitcher_games_prior",
]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(_numeric(frame[column]).sum())


def _mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default


def _rate(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def _positive_game_rate(
    frame: pd.DataFrame,
    column: str,
    threshold: float = 0.0,
) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float((_numeric(frame[column]) > threshold).mean())


def _batting_features(prior: pd.DataFrame) -> dict[str, float]:
    p5 = prior.tail(5)
    p10 = prior.tail(10)
    p20 = prior.tail(20)
    p30 = prior.tail(30)

    def batting_average(frame: pd.DataFrame) -> float:
        return _rate(_sum(frame, "hits"), _sum(frame, "at_bats"), 0.245)

    def on_base_percentage(frame: pd.DataFrame) -> float:
        return _rate(
            _sum(frame, "hits") + _sum(frame, "walks"),
            _sum(frame, "plate_appearances"),
            0.320,
        )

    def slugging(frame: pd.DataFrame) -> float:
        return _rate(_sum(frame, "total_bases"), _sum(frame, "at_bats"), 0.400)

    def strikeout_rate(frame: pd.DataFrame) -> float:
        return _rate(
            _sum(frame, "strikeouts"),
            _sum(frame, "plate_appearances"),
            0.220,
        )

    def walk_rate(frame: pd.DataFrame) -> float:
        return _rate(
            _sum(frame, "walks"),
            _sum(frame, "plate_appearances"),
            0.080,
        )

    def balls_in_play_rate(frame: pd.DataFrame) -> float:
        pa = _sum(frame, "plate_appearances")
        bip = max(
            pa
            - _sum(frame, "strikeouts")
            - _sum(frame, "walks")
            - _sum(frame, "home_runs"),
            0.0,
        )
        return _rate(bip, pa, 0.650)

    def hr_per_pa(frame: pd.DataFrame) -> float:
        return _rate(
            _sum(frame, "home_runs"),
            _sum(frame, "plate_appearances"),
            0.030,
        )

    avg_5 = batting_average(p5)
    avg_10 = batting_average(p10)
    avg_30 = batting_average(p30)
    obp_10 = on_base_percentage(p10)
    obp_30 = on_base_percentage(p30)
    slg_10 = slugging(p10)
    slg_30 = slugging(p30)
    iso_10 = slg_10 - avg_10
    iso_30 = slg_30 - avg_30

    hit_rate_5 = _positive_game_rate(p5, "hits")
    hit_rate_10 = _positive_game_rate(p10, "hits")
    hit_rate_20 = _positive_game_rate(p20, "hits")
    hit_rate_30 = _positive_game_rate(p30, "hits")

    hr_rate_5 = _positive_game_rate(p5, "home_runs")
    hr_rate_10 = _positive_game_rate(p10, "home_runs")
    hr_rate_20 = _positive_game_rate(p20, "home_runs")
    hr_rate_30 = _positive_game_rate(p30, "home_runs")

    games_since_last_hr = 30.0
    if not prior.empty and "home_runs" in prior.columns:
        reversed_hr = _numeric(prior["home_runs"]).iloc[::-1].reset_index(drop=True)
        matches = reversed_hr[reversed_hr > 0]
        if not matches.empty:
            games_since_last_hr = float(matches.index[0])

    return {
        "hit_rate_5": hit_rate_5,
        "hit_rate_10": hit_rate_10,
        "hit_rate_20": hit_rate_20,
        "hit_rate_30": hit_rate_30,
        "multi_hit_rate_10": _positive_game_rate(p10, "hits", 1.0),
        "multi_hit_rate_30": _positive_game_rate(p30, "hits", 1.0),
        "batting_avg_5": avg_5,
        "batting_avg_10": avg_10,
        "batting_avg_30": avg_30,
        "obp_10": obp_10,
        "obp_30": obp_30,
        "slg_10": slg_10,
        "slg_30": slg_30,
        "ops_10": obp_10 + slg_10,
        "ops_30": obp_30 + slg_30,
        "iso_10": iso_10,
        "iso_30": iso_30,
        "pa_avg_5": _mean(p5, "plate_appearances", 4.0),
        "pa_avg_10": _mean(p10, "plate_appearances", 4.0),
        "pa_avg_30": _mean(p30, "plate_appearances", 4.0),
        "ab_avg_10": _mean(p10, "at_bats", 3.6),
        "games_with_4_pa_rate_10": _positive_game_rate(
            p10, "plate_appearances", 3.0
        ),
        "strikeout_rate_10": strikeout_rate(p10),
        "strikeout_rate_30": strikeout_rate(p30),
        "walk_rate_10": walk_rate(p10),
        "walk_rate_30": walk_rate(p30),
        "balls_in_play_rate_10": balls_in_play_rate(p10),
        "balls_in_play_rate_30": balls_in_play_rate(p30),
        "hits_per_game_5": _mean(p5, "hits", 1.0),
        "hits_per_game_10": _mean(p10, "hits", 1.0),
        "hits_per_game_30": _mean(p30, "hits", 1.0),
        "total_bases_per_game_10": _mean(p10, "total_bases", 1.5),
        "total_bases_per_game_30": _mean(p30, "total_bases", 1.5),
        "home_run_rate_5": hr_rate_5,
        "home_run_rate_10": hr_rate_10,
        "home_run_rate_20": hr_rate_20,
        "home_run_rate_30": hr_rate_30,
        "home_runs_per_pa_10": hr_per_pa(p10),
        "home_runs_per_pa_30": hr_per_pa(p30),
        "slugging_10": slg_10,
        "slugging_30": slg_30,
        "extra_base_hit_rate_10": _rate(
            _sum(p10, "total_bases") - _sum(p10, "hits"),
            _sum(p10, "plate_appearances"),
            0.100,
        ),
        "extra_base_hit_rate_30": _rate(
            _sum(p30, "total_bases") - _sum(p30, "hits"),
            _sum(p30, "plate_appearances"),
            0.100,
        ),
        "hit_rate_trend_5_vs_30": hit_rate_5 - hit_rate_30,
        "batting_avg_trend_10_vs_30": avg_10 - avg_30,
        "pa_trend_5_vs_30": (
            _mean(p5, "plate_appearances", 4.0)
            - _mean(p30, "plate_appearances", 4.0)
        ),
        "home_run_rate_trend_5_vs_30": hr_rate_5 - hr_rate_30,
        "iso_trend_10_vs_30": iso_10 - iso_30,
        "slugging_trend_10_vs_30": slg_10 - slg_30,
        "games_since_last_hr": min(games_since_last_hr, 30.0),
        "games_prior": float(len(prior)),
    }


def _opponent_starter_features(
    pitching: pd.DataFrame,
    pitcher_id,
    game_date,
) -> dict[str, float]:
    defaults = {
        "opponent_sp_era_5": 4.30,
        "opponent_sp_whip_5": 1.30,
        "opponent_sp_k_per_9_5": 8.50,
        "opponent_sp_bb_per_9_5": 3.20,
        "opponent_sp_k_minus_bb_rate_5": 0.145,
        "opponent_sp_hr_per_9_5": 1.20,
        "opponent_sp_hits_per_9_5": 8.50,
        "opponent_sp_pitches_5": 85.0,
        "opponent_sp_innings_5": 5.0,
    }

    if (
        pitching.empty
        or pd.isna(pitcher_id)
        or "player_id" not in pitching.columns
        or "game_date" not in pitching.columns
    ):
        return defaults

    ids = pd.to_numeric(pitching["player_id"], errors="coerce")
    dates = pd.to_datetime(pitching["game_date"], errors="coerce")
    starts = (
        pd.to_numeric(pitching["games_started"], errors="coerce").fillna(0)
        if "games_started" in pitching.columns
        else pd.Series(1, index=pitching.index)
    )

    history = pitching.loc[
        (ids == float(pitcher_id))
        & (dates < pd.Timestamp(game_date))
        & (starts > 0)
    ].copy()

    history["_date"] = dates.loc[history.index]
    history = history.sort_values("_date").tail(5)

    if history.empty:
        return defaults

    innings = _sum(history, "innings_pitched")
    earned_runs = _sum(history, "earned_runs")
    hits = _sum(history, "hits")
    walks = _sum(history, "walks")
    strikeouts = _sum(history, "strikeouts")
    home_runs = _sum(history, "home_runs")
    batters_faced = _sum(history, "batters_faced")

    if batters_faced <= 0:
        batters_faced = max(innings * 3 + hits + walks, 1.0)

    k_rate = _rate(strikeouts, batters_faced, 0.230)
    bb_rate = _rate(walks, batters_faced, 0.085)

    return {
        "opponent_sp_era_5": _rate(earned_runs * 9, innings, 4.30),
        "opponent_sp_whip_5": _rate(hits + walks, innings, 1.30),
        "opponent_sp_k_per_9_5": _rate(strikeouts * 9, innings, 8.50),
        "opponent_sp_bb_per_9_5": _rate(walks * 9, innings, 3.20),
        "opponent_sp_k_minus_bb_rate_5": k_rate - bb_rate,
        "opponent_sp_hr_per_9_5": _rate(home_runs * 9, innings, 1.20),
        "opponent_sp_hits_per_9_5": _rate(hits * 9, innings, 8.50),
        "opponent_sp_pitches_5": _mean(history, "pitches_thrown", 85.0),
        "opponent_sp_innings_5": _mean(history, "innings_pitched", 5.0),
    }


def _lineup_features(current_game: pd.Series, prior: pd.DataFrame) -> dict[str, float]:
    order_value = pd.to_numeric(
        pd.Series([current_game.get("batting_order")]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(order_value) or not 1 <= float(order_value) <= 9:
        historical_order = pd.to_numeric(
            prior.get("batting_order", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        order_value = float(historical_order.tail(10).median()) if not historical_order.empty else 5.0

    order_value = float(min(max(order_value, 1.0), 9.0))

    historical_pa = _mean(prior.tail(10), "plate_appearances", 4.0)
    lineup_pa_adjustment = {
        1: 0.35,
        2: 0.30,
        3: 0.20,
        4: 0.15,
        5: 0.05,
        6: -0.05,
        7: -0.15,
        8: -0.25,
        9: -0.35,
    }

    projected_pa = historical_pa + lineup_pa_adjustment[int(round(order_value))]
    projected_pa = float(min(max(projected_pa, 2.5), 5.2))

    side = str(current_game.get("side", "")).lower()

    return {
        "batting_order": order_value,
        "top_4_lineup": float(order_value <= 4),
        "projected_pa": projected_pa,
        "is_home": float(side == "home"),
    }


def build_batter_training(
    batting: pd.DataFrame,
    pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    pitching = pitching.copy() if pitching is not None else pd.DataFrame()

    if batting.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "player_id",
                "game_date",
                "hit",
                "home_run",
                *sorted(set(HIT_FEATURES + HR_FEATURES)),
            ]
        )

    data = batting.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data.sort_values(["player_id", "game_date", "game_pk"])

    if not pitching.empty:
        pitching["game_date"] = pd.to_datetime(
            pitching["game_date"],
            errors="coerce",
        )

    for player_id, player_games in data.groupby("player_id"):
        player_games = player_games.reset_index(drop=True)

        for index, current_game in player_games.iterrows():
            prior = player_games.iloc[max(0, index - 30):index]
            if len(prior) < 3:
                continue

            opponent_pitcher_id = current_game.get("opponent_pitcher_id")

            current_hits = pd.to_numeric(
                pd.Series([current_game.get("hits")]),
                errors="coerce",
            ).fillna(0).iloc[0]

            current_home_runs = pd.to_numeric(
                pd.Series([current_game.get("home_runs")]),
                errors="coerce",
            ).fillna(0).iloc[0]

            rows.append(
                {
                    "game_pk": current_game["game_pk"],
                    "player_id": player_id,
                    "game_date": current_game["game_date"],
                    "hit": float(current_hits > 0),
                    "home_run": float(current_home_runs > 0),
                    **_batting_features(prior),
                    **_lineup_features(current_game, prior),
                    **_opponent_starter_features(
                        pitching,
                        opponent_pitcher_id,
                        current_game["game_date"],
                    ),
                }
            )

    result = pd.DataFrame(rows)
    all_features = sorted(set(HIT_FEATURES + HR_FEATURES))

    for feature in all_features:
        if feature not in result.columns:
            result[feature] = 0.0
        result[feature] = pd.to_numeric(result[feature], errors="coerce")

    result[all_features] = result[all_features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result



def build_batter_prediction_rows(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    team_id: int,
    opponent_pitcher_id,
    game_date,
    side: str,
    max_players: int = 9,
) -> pd.DataFrame:
    """Build point-in-time feature rows for likely active hitters.

    Until confirmed lineups are available, likely hitters are selected from the
    most recent team appearances and their batting order is estimated from the
    median of their last ten recorded lineup spots.
    """
    if batting.empty:
        return pd.DataFrame()

    data = batting.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    cutoff = pd.Timestamp(game_date)
    team_history = data.loc[
        (pd.to_numeric(data["team_id"], errors="coerce") == int(team_id))
        & (data["game_date"] < cutoff)
    ].copy()

    if team_history.empty:
        return pd.DataFrame()

    candidates = []
    for player_id, history in team_history.groupby("player_id"):
        history = history.sort_values(["game_date", "game_pk"]).tail(30)
        if len(history) < 3:
            continue
        recent = history.tail(10)
        candidates.append(
            {
                "player_id": player_id,
                "player_name": history["player_name"].dropna().iloc[-1]
                if history["player_name"].notna().any()
                else str(player_id),
                "last_game": history["game_date"].max(),
                "recent_pa": _sum(recent, "plate_appearances"),
                "history": history,
            }
        )

    if not candidates:
        return pd.DataFrame()

    candidates.sort(
        key=lambda row: (row["last_game"], row["recent_pa"]),
        reverse=True,
    )
    candidates = candidates[:max_players]

    rows = []
    for candidate in candidates:
        prior = candidate["history"]
        current = pd.Series(
            {
                "batting_order": np.nan,
                "side": side,
            }
        )
        lineup = _lineup_features(current, prior)
        rows.append(
            {
                "player_id": candidate["player_id"],
                "player_name": candidate["player_name"],
                "team_id": int(team_id),
                "game_date": cutoff,
                "opponent_pitcher_id": opponent_pitcher_id,
                "lineup_status": "Projected from recent games",
                **_batting_features(prior),
                **lineup,
                **_opponent_starter_features(
                    pitching, opponent_pitcher_id, cutoff
                ),
            }
        )

    result = pd.DataFrame(rows)
    all_features = sorted(set(HIT_FEATURES + HR_FEATURES))
    for feature in all_features:
        if feature not in result.columns:
            result[feature] = np.nan
        result[feature] = pd.to_numeric(result[feature], errors="coerce")

    return result.sort_values(
        ["batting_order", "projected_pa"], ascending=[True, False]
    ).reset_index(drop=True)


def build_pitcher_prediction_row(
    pitching: pd.DataFrame,
    pitcher_id,
    game_date,
) -> pd.DataFrame:
    """Build a point-in-time strikeout feature row for a probable starter."""
    if pitching.empty or pd.isna(pitcher_id):
        return pd.DataFrame()

    data = pitching.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    ids = pd.to_numeric(data["player_id"], errors="coerce")
    starts = pd.to_numeric(data.get("games_started", 1), errors="coerce").fillna(0)
    prior = data.loc[
        (ids == float(pitcher_id))
        & (data["game_date"] < pd.Timestamp(game_date))
        & (starts > 0)
    ].sort_values(["game_date", "game_pk"]).tail(10)

    if len(prior) < 2:
        return pd.DataFrame()

    names = prior.get("player_name", pd.Series(dtype=object)).dropna()
    row = {
        "player_id": int(float(pitcher_id)),
        "player_name": names.iloc[-1] if not names.empty else str(pitcher_id),
        "game_date": pd.Timestamp(game_date),
        **_pitcher_features(prior),
    }
    return pd.DataFrame([row])

def _pitcher_features(prior: pd.DataFrame) -> dict[str, float]:
    p3 = prior.tail(3)
    p5 = prior.tail(5)
    p10 = prior.tail(10)

    def innings(frame: pd.DataFrame) -> float:
        return _sum(frame, "innings_pitched")

    def strikeouts(frame: pd.DataFrame) -> float:
        return _sum(frame, "strikeouts")

    def walks(frame: pd.DataFrame) -> float:
        return _sum(frame, "walks")

    def batters_faced(frame: pd.DataFrame) -> float:
        value = _sum(frame, "batters_faced")
        if value > 0:
            return value
        return max(innings(frame) * 3 + _sum(frame, "hits") + walks(frame), 1.0)

    def k_per_ip(frame: pd.DataFrame) -> float:
        return _rate(strikeouts(frame), innings(frame), 0.95)

    def k_per_9(frame: pd.DataFrame) -> float:
        return _rate(strikeouts(frame) * 9, innings(frame), 8.5)

    def bb_per_9(frame: pd.DataFrame) -> float:
        return _rate(walks(frame) * 9, innings(frame), 3.2)

    def k_rate(frame: pd.DataFrame) -> float:
        return _rate(strikeouts(frame), batters_faced(frame), 0.230)

    def bb_rate(frame: pd.DataFrame) -> float:
        return _rate(walks(frame), batters_faced(frame), 0.085)

    k_avg_3 = _mean(p3, "strikeouts", 5.0)
    k_avg_5 = _mean(p5, "strikeouts", 5.0)
    k_avg_10 = _mean(p10, "strikeouts", 5.0)
    ip_avg_3 = _mean(p3, "innings_pitched", 5.0)
    ip_avg_5 = _mean(p5, "innings_pitched", 5.0)
    ip_avg_10 = _mean(p10, "innings_pitched", 5.0)
    bf_5 = batters_faced(p5)
    bf_10 = batters_faced(p10)
    pitches_5 = _sum(p5, "pitches_thrown")

    return {
        "pitcher_k_per_ip_3": k_per_ip(p3),
        "pitcher_k_per_ip_5": k_per_ip(p5),
        "pitcher_k_per_ip_10": k_per_ip(p10),
        "pitcher_k_per_9_5": k_per_9(p5),
        "pitcher_k_per_9_10": k_per_9(p10),
        "pitcher_k_rate_5": k_rate(p5),
        "pitcher_k_rate_10": k_rate(p10),
        "pitcher_pitches_3": _mean(p3, "pitches_thrown", 85.0),
        "pitcher_pitches_5": _mean(p5, "pitches_thrown", 85.0),
        "pitcher_pitches_10": _mean(p10, "pitches_thrown", 85.0),
        "pitcher_ip_3": ip_avg_3,
        "pitcher_ip_5": ip_avg_5,
        "pitcher_ip_10": ip_avg_10,
        "pitcher_batters_faced_5": _rate(bf_5, max(len(p5), 1), 22.0),
        "pitcher_batters_faced_10": _rate(bf_10, max(len(p10), 1), 22.0),
        "pitcher_bb_per_9_5": bb_per_9(p5),
        "pitcher_bb_per_9_10": bb_per_9(p10),
        "pitcher_k_minus_bb_rate_5": k_rate(p5) - bb_rate(p5),
        "pitcher_k_minus_bb_rate_10": k_rate(p10) - bb_rate(p10),
        "pitcher_pitches_per_ip_5": _rate(pitches_5, innings(p5), 16.5),
        "pitcher_pitches_per_batter_5": _rate(pitches_5, bf_5, 3.9),
        "pitcher_k_avg_3": k_avg_3,
        "pitcher_k_avg_5": k_avg_5,
        "pitcher_k_avg_10": k_avg_10,
        "pitcher_k_trend_3_vs_10": k_avg_3 - k_avg_10,
        "pitcher_ip_trend_3_vs_10": ip_avg_3 - ip_avg_10,
        "pitcher_games_prior": float(len(prior)),
    }


def build_pitcher_k_training(pitching: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    if pitching.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "player_id",
                "game_date",
                "strikeouts",
                *K_FEATURES,
            ]
        )

    data = pitching.copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")

    if "games_started" in data.columns:
        games_started = pd.to_numeric(
            data["games_started"],
            errors="coerce",
        ).fillna(0)
        data = data[games_started > 0]

    data = data.sort_values(["player_id", "game_date", "game_pk"])

    for player_id, pitcher_games in data.groupby("player_id"):
        pitcher_games = pitcher_games.reset_index(drop=True)

        for index, current_game in pitcher_games.iterrows():
            prior = pitcher_games.iloc[max(0, index - 10):index]
            if len(prior) < 2:
                continue

            current_strikeouts = pd.to_numeric(
                pd.Series([current_game.get("strikeouts")]),
                errors="coerce",
            ).fillna(0).iloc[0]

            rows.append(
                {
                    "game_pk": current_game["game_pk"],
                    "player_id": player_id,
                    "game_date": current_game["game_date"],
                    "strikeouts": float(current_strikeouts),
                    **_pitcher_features(prior),
                }
            )

    result = pd.DataFrame(rows)

    for feature in K_FEATURES:
        if feature not in result.columns:
            result[feature] = 0.0
        result[feature] = pd.to_numeric(result[feature], errors="coerce")

    result[K_FEATURES] = result[K_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result
