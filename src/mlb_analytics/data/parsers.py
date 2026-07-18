from __future__ import annotations

import pandas as pd


def parse_schedule(payload: dict) -> pd.DataFrame:
    rows: list[dict] = []

    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})

            away_pitcher = away.get("probablePitcher", {}) or {}
            home_pitcher = home.get("probablePitcher", {}) or {}

            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game.get("officialDate"),
                    "game_time": game.get("gameDate"),
                    "status": game.get("status", {}).get(
                        "detailedState"
                    ),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team": away.get("team", {}).get("name"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team": home.get("team", {}).get("name"),
                    "away_score": away.get("score"),
                    "home_score": home.get("score"),
                    "venue": game.get("venue", {}).get("name"),
                    "away_probable_pitcher_id": away_pitcher.get(
                        "id"
                    ),
                    "away_probable_pitcher": away_pitcher.get(
                        "fullName"
                    ),
                    "home_probable_pitcher_id": home_pitcher.get(
                        "id"
                    ),
                    "home_probable_pitcher": home_pitcher.get(
                        "fullName"
                    ),
                }
            )

    return pd.DataFrame(rows)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _batting_order(value) -> float | None:
    numeric_value = _num(value)

    if numeric_value is None:
        return None

    order = int(numeric_value)

    # MLB box scores commonly store lineup order as:
    # 100, 200, 300 ... 900.
    if order >= 100:
        order //= 100

    if 1 <= order <= 9:
        return float(order)

    return None


def _probable_pitcher_id(
    games_row,
    opponent_side: str,
):
    value = games_row.get(
        f"{opponent_side}_probable_pitcher_id"
    )

    numeric_value = _num(value)

    if numeric_value is None:
        return None

    return int(numeric_value)


def parse_boxscore(
    game_pk,
    payload: dict,
    games_row,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    team_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    player_rows: list[dict] = []

    teams = payload.get("teams", {})

    for side in ("away", "home"):
        opponent_side = "home" if side == "away" else "away"

        block = teams.get(side, {})
        stats = block.get("teamStats", {})
        batting_stats = stats.get("batting", {})
        pitching_stats = stats.get("pitching", {})

        team_id = games_row[f"{side}_team_id"]
        opponent_id = games_row[f"{opponent_side}_team_id"]

        opponent_pitcher_id = _probable_pitcher_id(
            games_row,
            opponent_side,
        )

        team_rows.append(
            {
                "game_pk": game_pk,
                "game_date": games_row["game_date"],
                "team_id": team_id,
                "side": side,
                "opponent_id": opponent_id,
                "runs": games_row.get(f"{side}_score"),
                "hits": _num(batting_stats.get("hits")),
                "home_runs": _num(
                    batting_stats.get("homeRuns")
                ),
                "walks": _num(
                    batting_stats.get("baseOnBalls")
                ),
                "strikeouts": _num(
                    batting_stats.get("strikeOuts")
                ),
                "at_bats": _num(
                    batting_stats.get("atBats")
                ),
                "total_bases": _num(
                    batting_stats.get("totalBases")
                ),
                "bullpen_innings": 0.0,
                "bullpen_pitches": 0.0,
            }
        )

        for player in block.get("players", {}).values():
            person = player.get("person", {})
            player_stats = player.get("stats", {})
            player_pitching = player_stats.get("pitching", {})
            player_batting = player_stats.get("batting", {})

            if player_pitching:
                pitcher_rows.append(
                    {
                        "game_pk": game_pk,
                        "game_date": games_row["game_date"],
                        "team_id": team_id,
                        "player_id": person.get("id"),
                        "player_name": person.get("fullName"),
                        "side": side,
                        "games_started": (
                            _num(
                                player_pitching.get(
                                    "gamesStarted"
                                )
                            )
                            or 0.0
                        ),
                        "innings_pitched": (
                            _num(
                                player_pitching.get(
                                    "inningsPitched"
                                )
                            )
                            or 0.0
                        ),
                        "pitches_thrown": (
                            _num(
                                player_pitching.get(
                                    "numberOfPitches"
                                )
                            )
                            or 0.0
                        ),
                        "hits": (
                            _num(player_pitching.get("hits"))
                            or 0.0
                        ),
                        "runs": (
                            _num(player_pitching.get("runs"))
                            or 0.0
                        ),
                        "earned_runs": (
                            _num(
                                player_pitching.get(
                                    "earnedRuns"
                                )
                            )
                            or 0.0
                        ),
                        "walks": (
                            _num(
                                player_pitching.get(
                                    "baseOnBalls"
                                )
                            )
                            or 0.0
                        ),
                        "strikeouts": (
                            _num(
                                player_pitching.get(
                                    "strikeOuts"
                                )
                            )
                            or 0.0
                        ),
                        "home_runs": (
                            _num(
                                player_pitching.get(
                                    "homeRuns"
                                )
                            )
                            or 0.0
                        ),
                    }
                )

            plate_appearances = (
                _num(player_batting.get("plateAppearances"))
                or 0.0
            )

            if player_batting and plate_appearances > 0:
                player_rows.append(
                    {
                        "game_pk": game_pk,
                        "game_date": games_row["game_date"],
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "opponent_pitcher_id": opponent_pitcher_id,
                        "player_id": person.get("id"),
                        "player_name": person.get("fullName"),
                        "side": side,
                        "batting_order": _batting_order(
                            player.get("battingOrder")
                        ),
                        "plate_appearances": plate_appearances,
                        "at_bats": (
                            _num(player_batting.get("atBats"))
                            or 0.0
                        ),
                        "hits": (
                            _num(player_batting.get("hits"))
                            or 0.0
                        ),
                        "home_runs": (
                            _num(
                                player_batting.get("homeRuns")
                            )
                            or 0.0
                        ),
                        "walks": (
                            _num(
                                player_batting.get(
                                    "baseOnBalls"
                                )
                            )
                            or 0.0
                        ),
                        "strikeouts": (
                            _num(
                                player_batting.get(
                                    "strikeOuts"
                                )
                            )
                            or 0.0
                        ),
                        "total_bases": (
                            _num(
                                player_batting.get(
                                    "totalBases"
                                )
                            )
                            or 0.0
                        ),
                    }
                )

    # Bullpen totals include pitchers who did not start.
    for team_row in team_rows:
        relievers = [
            pitcher
            for pitcher in pitcher_rows
            if (
                pitcher["team_id"] == team_row["team_id"]
                and pitcher["games_started"] == 0
            )
        ]

        team_row["bullpen_innings"] = sum(
            pitcher["innings_pitched"]
            for pitcher in relievers
        )

        team_row["bullpen_pitches"] = sum(
            pitcher["pitches_thrown"]
            for pitcher in relievers
        )

    return (
        pd.DataFrame(team_rows),
        pd.DataFrame(pitcher_rows),
        pd.DataFrame(player_rows),
    )