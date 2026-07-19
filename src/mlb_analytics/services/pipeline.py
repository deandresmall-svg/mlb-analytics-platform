from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from mlb_analytics.config import Settings
from mlb_analytics.data.mlb_api import MLBClient
from mlb_analytics.data.odds_api import MLB_PROP_MARKETS, OddsAPIClient
from mlb_analytics.data.parsers import parse_boxscore, parse_schedule
from mlb_analytics.data.repository import Repository
from mlb_analytics.data.statcast import (
    StatcastClient,
    aggregate_statcast,
    chunk_ranges,
    prepare_statcast,
)
from mlb_analytics.data.venues import venue_info
from mlb_analytics.data.weather import WeatherClient
from mlb_analytics.features.game_features import FEATURES, build_game_features
from mlb_analytics.features.player_features import (
    HIT_FEATURES,
    HR_FEATURES,
    K_FEATURES,
    build_batter_prediction_rows,
    build_batter_training,
    build_pitcher_k_training,
    build_pitcher_prediction_row,
)
from mlb_analytics.features.scores import add_batter_scores, add_pitcher_scores
from mlb_analytics.features.validated_signals import (
    add_batter_signals,
    add_pitcher_signals,
)
from mlb_analytics.models.base import BinaryTimeModel, CountTimeModel


ProgressCallback = Callable[[int, int, str], None]


class AnalyticsService:
    def __init__(self, settings: Settings):
        self.s = settings
        settings.ensure_directories()
        self.mlb = MLBClient(
            settings.api_base_url,
            settings.api_timeout_seconds,
            settings.api_max_retries,
        )
        self.weather = WeatherClient(
            settings.weather_base_url,
            settings.weather_archive_url,
            settings.api_timeout_seconds,
            settings.api_max_retries,
        )
        self.repo = Repository(settings.database_url)
        self.repo.initialize()
        self.odds = OddsAPIClient(
            settings.odds_api_key,
            settings.odds_api_base_url,
            settings.api_timeout_seconds,
        )
        self.statcast = StatcastClient(
            settings.statcast_base_url,
            settings.statcast_timeout_seconds,
            settings.api_max_retries,
        )

    def sync_schedule(
        self,
        start: date,
        end: date | None = None,
        with_weather: bool = True,
    ) -> pd.DataFrame:
        games = parse_schedule(self.mlb.schedule(start, end))
        if games.empty:
            return games

        for column in [
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "precipitation",
            "park_factor",
        ]:
            games[column] = None

        for index, row in games.iterrows():
            latitude, longitude, park_factor = venue_info(row.venue)
            games.at[index, "park_factor"] = park_factor
            if not with_weather or latitude is None:
                continue
            try:
                weather = self.weather.nearest(
                    self.weather.hourly(
                        latitude,
                        longitude,
                        date.fromisoformat(row.game_date),
                    ),
                    row.game_time,
                )
                games.at[index, "temperature"] = weather.get("temperature_2m")
                games.at[index, "humidity"] = weather.get("relative_humidity_2m")
                games.at[index, "pressure"] = weather.get("surface_pressure")
                games.at[index, "wind_speed"] = weather.get("wind_speed_10m")
                games.at[index, "wind_direction"] = weather.get(
                    "wind_direction_10m"
                )
                games.at[index, "precipitation"] = weather.get("precipitation")
            except Exception:
                pass

        self.repo.upsert_games(games)
        return games

    def backfill(
        self,
        start: date,
        end: date,
        include_boxscores: bool = True,
    ) -> dict[str, int]:
        synced = 0
        boxscores = 0
        day = start
        while day <= end:
            games = self.sync_schedule(day, day, with_weather=True)
            synced += len(games)
            required_scores = {"away_score", "home_score"}
            if (
                include_boxscores
                and not games.empty
                and required_scores.issubset(games.columns)
            ):
                completed = games.dropna(subset=["away_score", "home_score"])
                for _, game in completed.iterrows():
                    try:
                        team, pitcher, batter = parse_boxscore(
                            int(game.game_pk),
                            self.mlb.boxscore(int(game.game_pk)),
                            game,
                        )
                        self.repo.upsert_team_stats(team)
                        self.repo.upsert_pitcher_stats(pitcher)
                        self.repo.upsert_batting(batter)
                        boxscores += 1
                    except Exception:
                        continue
            day += timedelta(days=1)
        return {"games": synced, "boxscores": boxscores}

    def sync_statcast(
        self,
        start: date,
        end: date,
        chunk_days: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict:
        if end < start:
            raise ValueError("Statcast end date cannot be before start date")

        ranges = chunk_ranges(
            start,
            end,
            chunk_days or self.s.statcast_chunk_days,
        )
        totals = {
            "pitches": 0,
            "batter_games": 0,
            "pitcher_games": 0,
            "batter_pitch_type_rows": 0,
            "pitcher_pitch_type_rows": 0,
        }
        chunks: list[dict] = []
        errors: list[str] = []

        for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
            message = f"Downloading {chunk_start} through {chunk_end}"
            if progress:
                progress(index - 1, len(ranges), message)
            try:
                pitches = self.statcast.fetch(chunk_start, chunk_end)
                aggregates = aggregate_statcast(pitches)
                counts = {
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                    "pitches": self.repo.upsert_statcast_pitches(pitches),
                    "batter_games": self.repo.upsert_statcast_batters(
                        aggregates["batters"]
                    ),
                    "pitcher_games": self.repo.upsert_statcast_pitchers(
                        aggregates["pitchers"]
                    ),
                    "batter_pitch_type_rows": (
                        self.repo.upsert_statcast_batter_pitch_types(
                            aggregates["batter_pitch_types"]
                        )
                    ),
                    "pitcher_pitch_type_rows": (
                        self.repo.upsert_statcast_pitcher_pitch_types(
                            aggregates["pitcher_pitch_types"]
                        )
                    ),
                }
                chunks.append(counts)
                for key in totals:
                    totals[key] += int(counts[key])
            except Exception as exc:
                errors.append(f"{chunk_start} through {chunk_end}: {exc}")

            if progress:
                progress(index, len(ranges), message)

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "chunks": chunks,
            "errors": errors,
            **totals,
        }

    def rebuild_statcast_aggregates(
        self,
        start: date,
        end: date,
    ) -> dict[str, int]:
        pitches = self.repo.statcast_pitches_between(start, end)
        corrected_pitches = prepare_statcast(pitches)
        aggregates = aggregate_statcast(corrected_pitches)
        return {
            "pitches_read": len(corrected_pitches),
            "pitch_flags_updated": self.repo.upsert_statcast_pitches(
                corrected_pitches
            ),
            "batter_games": self.repo.upsert_statcast_batters(
                aggregates["batters"]
            ),
            "pitcher_games": self.repo.upsert_statcast_pitchers(
                aggregates["pitchers"]
            ),
            "batter_pitch_type_rows": self.repo.upsert_statcast_batter_pitch_types(
                aggregates["batter_pitch_types"]
            ),
            "pitcher_pitch_type_rows": self.repo.upsert_statcast_pitcher_pitch_types(
                aggregates["pitcher_pitch_types"]
            ),
        }

    def datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        games = self.repo.completed_games()
        team = self.repo.query("SELECT * FROM team_game_stats")
        pitching = self.repo.query("SELECT * FROM pitcher_game_stats")
        batting = self.repo.query("SELECT * FROM player_game_batting")
        statcast_batters = self.repo.query("SELECT * FROM statcast_batter_game")
        statcast_pitchers = self.repo.query("SELECT * FROM statcast_pitcher_game")
        batter_pitch_types = self.repo.query(
            "SELECT * FROM statcast_batter_pitch_type_game"
        )
        pitcher_pitch_types = self.repo.query(
            "SELECT * FROM statcast_pitcher_pitch_type_game"
        )
        return (
            build_game_features(games, team, pitching),
            build_batter_training(
                batting,
                pitching,
                statcast_batters,
                statcast_pitchers,
                batter_pitch_types,
                pitcher_pitch_types,
            ),
            build_pitcher_k_training(
                pitching,
                statcast_pitchers,
                games,
                batting,
                batter_pitch_types,
                pitcher_pitch_types,
            ),
        )

    def train_all(self) -> dict:
        game, batter, strikeout = self.datasets()
        output: dict[str, dict] = {}
        targets = [
            (
                "home_win",
                game,
                FEATURES,
                "label",
                BinaryTimeModel(self.s.calibration_method),
            ),
            (
                "hit",
                batter,
                HIT_FEATURES,
                "hit",
                BinaryTimeModel(self.s.calibration_method),
            ),
            (
                "home_run",
                batter,
                HR_FEATURES,
                "home_run",
                BinaryTimeModel(self.s.calibration_method),
            ),
            (
                "strikeouts",
                strikeout,
                K_FEATURES,
                "strikeouts",
                CountTimeModel(),
            ),
        ]
        for name, dataframe, features, label, model in targets:
            try:
                metrics = model.fit(dataframe, features, label)
                path = self.s.model_dir / f"{name}.joblib"
                model.save(
                    path,
                    {
                        "features": features,
                        "target": name,
                        "trained_at": datetime.utcnow().isoformat(),
                        "statcast_enabled": any(
                            feature.startswith("sc_") for feature in features
                        ),
                        "pitch_matchup_enabled": any(
                            feature.startswith("pt_") for feature in features
                        ),
                    },
                )
                output[name] = asdict(metrics)
            except Exception as exc:
                output[name] = {"error": str(exc), "rows": len(dataframe)}
        return output

    def game_predictions(self, slate_date: date) -> pd.DataFrame:
        games = self.repo.games_for_date(slate_date)
        if games.empty:
            return games
        team = self.repo.query("SELECT * FROM team_game_stats")
        pitching = self.repo.query("SELECT * FROM pitcher_game_stats")
        features = build_game_features(games, team, pitching)
        path = self.s.model_dir / "home_win.joblib"
        if not path.exists():
            return games.assign(model_status="Train the home-win model first")
        model = BinaryTimeModel.load(path)
        probability = model.predict_frame(features, FEATURES)
        output = games.copy()
        output["home_win_probability"] = probability
        output["away_win_probability"] = 1 - probability
        output["pick"] = output.apply(
            lambda row: (
                row.home_team
                if row.home_win_probability >= 0.5
                else row.away_team
            ),
            axis=1,
        )
        output["confidence"] = output[
            ["home_win_probability", "away_win_probability"]
        ].max(axis=1)
        return output

    def batter_predictions_for_game(
        self,
        game_row: pd.Series,
        batting: pd.DataFrame | None = None,
        pitching: pd.DataFrame | None = None,
        statcast_batters: pd.DataFrame | None = None,
        statcast_pitchers: pd.DataFrame | None = None,
        batter_pitch_types: pd.DataFrame | None = None,
        pitcher_pitch_types: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        batting = (
            batting if batting is not None
            else self.repo.query("SELECT * FROM player_game_batting")
        )
        pitching = (
            pitching if pitching is not None
            else self.repo.query("SELECT * FROM pitcher_game_stats")
        )
        statcast_batters = (
            statcast_batters if statcast_batters is not None
            else self.repo.query("SELECT * FROM statcast_batter_game")
        )
        statcast_pitchers = (
            statcast_pitchers if statcast_pitchers is not None
            else self.repo.query("SELECT * FROM statcast_pitcher_game")
        )
        batter_pitch_types = (
            batter_pitch_types if batter_pitch_types is not None
            else self.repo.query("SELECT * FROM statcast_batter_pitch_type_game")
        )
        pitcher_pitch_types = (
            pitcher_pitch_types if pitcher_pitch_types is not None
            else self.repo.query("SELECT * FROM statcast_pitcher_pitch_type_game")
        )
        game_date = pd.Timestamp(game_row["game_date"]).date()
        outputs: list[pd.DataFrame] = []

        for side in ("away", "home"):
            opponent_side = "home" if side == "away" else "away"
            rows = build_batter_prediction_rows(
                batting,
                pitching,
                int(game_row[f"{side}_team_id"]),
                game_row.get(f"{opponent_side}_probable_pitcher_id"),
                game_date,
                side,
                statcast_batters,
                statcast_pitchers,
                batter_pitch_types,
                pitcher_pitch_types,
            )
            if rows.empty:
                continue
            rows["team"] = game_row[f"{side}_team"]
            rows["opponent_pitcher"] = game_row.get(
                f"{opponent_side}_probable_pitcher", "TBD"
            )
            for target, features in [
                ("hit", HIT_FEATURES),
                ("home_run", HR_FEATURES),
            ]:
                path = self.s.model_dir / f"{target}.joblib"
                if path.exists():
                    model = BinaryTimeModel.load(path)
                    rows[f"{target}_probability"] = model.predict_frame(
                        rows, features
                    )
            outputs.append(rows)

        if not outputs:
            return pd.DataFrame()

        output = pd.concat(outputs, ignore_index=True)
        output = add_batter_scores(output)
        output = add_batter_signals(output)
        for target in ("hit", "home_run"):
            column = f"{target}_probability"
            if column in output.columns:
                output[f"{target}_confidence"] = output[column].map(
                    self._confidence_label
                )
        return output

    def pitcher_predictions_for_game(
        self,
        game_row: pd.Series,
        pitching: pd.DataFrame | None = None,
        statcast_pitchers: pd.DataFrame | None = None,
        batting: pd.DataFrame | None = None,
        batter_pitch_types: pd.DataFrame | None = None,
        pitcher_pitch_types: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        pitching = (
            pitching if pitching is not None
            else self.repo.query("SELECT * FROM pitcher_game_stats")
        )
        statcast_pitchers = (
            statcast_pitchers if statcast_pitchers is not None
            else self.repo.query("SELECT * FROM statcast_pitcher_game")
        )
        batting = (
            batting if batting is not None
            else self.repo.query("SELECT * FROM player_game_batting")
        )
        batter_pitch_types = (
            batter_pitch_types if batter_pitch_types is not None
            else self.repo.query("SELECT * FROM statcast_batter_pitch_type_game")
        )
        pitcher_pitch_types = (
            pitcher_pitch_types if pitcher_pitch_types is not None
            else self.repo.query("SELECT * FROM statcast_pitcher_pitch_type_game")
        )
        path = self.s.model_dir / "strikeouts.joblib"
        if not path.exists():
            return pd.DataFrame()
        model = CountTimeModel.load(path)
        rows: list[pd.DataFrame] = []
        game_date = pd.Timestamp(game_row["game_date"]).date()

        for side in ("away", "home"):
            opponent_side = "home" if side == "away" else "away"
            frame = build_pitcher_prediction_row(
                pitching,
                game_row.get(f"{side}_probable_pitcher_id"),
                game_date,
                statcast_pitchers,
                batting,
                game_row.get(f"{opponent_side}_team_id"),
                batter_pitch_types,
                pitcher_pitch_types,
            )
            if frame.empty:
                continue
            frame["team"] = game_row[f"{side}_team"]
            frame["opponent"] = game_row[f"{opponent_side}_team"]
            frame["projected_strikeouts"] = model.predict_frame(
                frame, K_FEATURES
            )
            rows.append(frame)

        if not rows:
            return pd.DataFrame()
        output = add_pitcher_scores(pd.concat(rows, ignore_index=True))
        return add_pitcher_signals(output)

    def odds_for_slate(self, slate_date: date) -> dict:
        if not self.s.odds_api_key:
            return {
                "rows": pd.DataFrame(),
                "usage": {},
                "error": "ODDS_API_KEY is not configured",
                "event_errors": [],
                "matched_games": 0,
                "requested_games": 0,
            }
        games = self.repo.games_for_date(slate_date)
        if games.empty:
            return {
                "rows": pd.DataFrame(),
                "usage": {},
                "error": (
                    "No MLB games are stored for the selected date. "
                    "Sync the schedule first."
                ),
                "event_errors": [],
                "matched_games": 0,
                "requested_games": 0,
            }
        try:
            events = self.odds.events_for_date(slate_date)
        except Exception as exc:
            return {
                "rows": pd.DataFrame(),
                "usage": self.odds.usage.__dict__,
                "error": str(exc),
                "event_errors": [],
                "matched_games": 0,
                "requested_games": 0,
            }

        frames: list[pd.DataFrame] = []
        event_errors: list[str] = []
        matched = 0
        total_cost = 0
        for _, game in games.iterrows():
            event = self.odds.match_event(game, events)
            if not event:
                event_errors.append(
                    f"No Odds API event match for "
                    f"{game.get('away_team')} at {game.get('home_team')}"
                )
                continue
            matched += 1
            try:
                payload = self.odds.event_odds(
                    event["id"],
                    MLB_PROP_MARKETS,
                    self.s.odds_regions,
                    self.s.odds_bookmakers or None,
                )
                total_cost += self.odds.usage.last or 0
                frame = self.odds.flatten_props(payload)
                if not frame.empty:
                    frame["game_pk"] = game.get("game_pk")
                    frames.append(frame)
            except Exception as exc:
                event_errors.append(
                    f"{game.get('away_team')} at {game.get('home_team')}: {exc}"
                )

        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        usage = self.odds.usage.__dict__.copy()
        usage["refresh_cost"] = total_cost
        return {
            "rows": rows,
            "usage": usage,
            "event_errors": event_errors,
            "matched_games": matched,
            "requested_games": len(games),
        }

    @staticmethod
    def _confidence_label(probability) -> str:
        if pd.isna(probability):
            return "Unavailable"
        if probability >= 0.68:
            return "High"
        if probability >= 0.58:
            return "Medium"
        return "Low"

    def slate_prop_predictions(self, slate_date: date) -> dict[str, pd.DataFrame]:
        games = self.repo.games_for_date(slate_date)
        if games.empty:
            return {"batters": pd.DataFrame(), "pitchers": pd.DataFrame()}

        batting = self.repo.query("SELECT * FROM player_game_batting")
        pitching = self.repo.query("SELECT * FROM pitcher_game_stats")
        statcast_batters = self.repo.query("SELECT * FROM statcast_batter_game")
        statcast_pitchers = self.repo.query("SELECT * FROM statcast_pitcher_game")
        batter_pitch_types = self.repo.query(
            "SELECT * FROM statcast_batter_pitch_type_game"
        )
        pitcher_pitch_types = self.repo.query(
            "SELECT * FROM statcast_pitcher_pitch_type_game"
        )

        batter_frames: list[pd.DataFrame] = []
        pitcher_frames: list[pd.DataFrame] = []
        for _, game in games.iterrows():
            try:
                batters = self.batter_predictions_for_game(
                    game,
                    batting,
                    pitching,
                    statcast_batters,
                    statcast_pitchers,
                    batter_pitch_types,
                    pitcher_pitch_types,
                )
                if not batters.empty:
                    batters = batters.copy()
                    batters["game_pk"] = game.get("game_pk")
                    batters["matchup"] = (
                        f"{game.get('away_team')} at {game.get('home_team')}"
                    )
                    batters["game_time"] = game.get("game_time")
                    batter_frames.append(batters)
            except Exception:
                pass
            try:
                pitchers = self.pitcher_predictions_for_game(
                    game,
                    pitching,
                    statcast_pitchers,
                    batting,
                    batter_pitch_types,
                    pitcher_pitch_types,
                )
                if not pitchers.empty:
                    pitchers = pitchers.copy()
                    pitchers["game_pk"] = game.get("game_pk")
                    pitchers["matchup"] = (
                        f"{game.get('away_team')} at {game.get('home_team')}"
                    )
                    pitchers["game_time"] = game.get("game_time")
                    pitcher_frames.append(pitchers)
            except Exception:
                pass

        return {
            "batters": (
                pd.concat(batter_frames, ignore_index=True)
                if batter_frames
                else pd.DataFrame()
            ),
            "pitchers": (
                pd.concat(pitcher_frames, ignore_index=True)
                if pitcher_frames
                else pd.DataFrame()
            ),
        }
