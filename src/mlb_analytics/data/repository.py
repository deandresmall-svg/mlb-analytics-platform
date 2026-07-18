from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import create_engine, text


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS games(
        game_pk INTEGER PRIMARY KEY,
        game_date TEXT,
        game_time TEXT,
        status TEXT,
        away_team_id INTEGER,
        away_team TEXT,
        home_team_id INTEGER,
        home_team TEXT,
        away_score REAL,
        home_score REAL,
        venue TEXT,
        away_probable_pitcher_id INTEGER,
        away_probable_pitcher TEXT,
        home_probable_pitcher_id INTEGER,
        home_probable_pitcher TEXT,
        temperature REAL,
        humidity REAL,
        pressure REAL,
        wind_speed REAL,
        wind_direction REAL,
        precipitation REAL,
        park_factor REAL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_game_stats(
        game_pk INTEGER,
        game_date TEXT,
        team_id INTEGER,
        side TEXT,
        opponent_id INTEGER,
        runs REAL,
        hits REAL,
        home_runs REAL,
        walks REAL,
        strikeouts REAL,
        at_bats REAL,
        total_bases REAL,
        bullpen_innings REAL,
        bullpen_pitches REAL,
        PRIMARY KEY(game_pk, team_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pitcher_game_stats(
        game_pk INTEGER,
        game_date TEXT,
        team_id INTEGER,
        player_id INTEGER,
        player_name TEXT,
        side TEXT,
        games_started REAL,
        innings_pitched REAL,
        pitches_thrown REAL,
        hits REAL,
        runs REAL,
        earned_runs REAL,
        walks REAL,
        strikeouts REAL,
        home_runs REAL,
        PRIMARY KEY(game_pk, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_game_batting(
        game_pk INTEGER,
        game_date TEXT,
        team_id INTEGER,
        opponent_id INTEGER,
        opponent_pitcher_id INTEGER,
        player_id INTEGER,
        player_name TEXT,
        side TEXT,
        batting_order REAL,
        plate_appearances REAL,
        at_bats REAL,
        hits REAL,
        home_runs REAL,
        walks REAL,
        strikeouts REAL,
        total_bases REAL,
        PRIMARY KEY(game_pk, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target TEXT,
        model_version TEXT,
        trained_at TEXT,
        rows INTEGER,
        brier REAL,
        roc_auc REAL,
        log_loss REAL,
        accuracy REAL,
        mae REAL,
        artifact_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS statcast_pitches(
        game_pk INTEGER,
        game_date TEXT,
        game_type TEXT,
        home_team TEXT,
        away_team TEXT,
        inning INTEGER,
        inning_topbot TEXT,
        at_bat_number INTEGER,
        pitch_number INTEGER,
        batter INTEGER,
        pitcher INTEGER,
        player_name TEXT,
        stand TEXT,
        p_throws TEXT,
        events TEXT,
        description TEXT,
        type TEXT,
        pitch_type TEXT,
        pitch_name TEXT,
        zone INTEGER,
        plate_x REAL,
        plate_z REAL,
        release_speed REAL,
        release_spin_rate REAL,
        release_extension REAL,
        pfx_x REAL,
        pfx_z REAL,
        launch_speed REAL,
        launch_angle REAL,
        hit_distance_sc REAL,
        bb_type TEXT,
        hc_x REAL,
        hc_y REAL,
        estimated_ba_using_speedangle REAL,
        estimated_slg_using_speedangle REAL,
        estimated_woba_using_speedangle REAL,
        woba_value REAL,
        woba_denom REAL,
        babip_value REAL,
        iso_value REAL,
        launch_speed_angle INTEGER,
        is_swing INTEGER,
        is_whiff INTEGER,
        is_called_strike INTEGER,
        is_in_zone INTEGER,
        is_out_zone INTEGER,
        is_chase INTEGER,
        is_bbe INTEGER,
        is_hard_hit INTEGER,
        is_barrel INTEGER,
        is_sweet_spot INTEGER,
        is_ground_ball INTEGER,
        is_fly_ball INTEGER,
        is_line_drive INTEGER,
        is_popup INTEGER,
        PRIMARY KEY(game_pk, at_bat_number, pitch_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS statcast_batter_game(
        game_pk INTEGER,
        game_date TEXT,
        player_id INTEGER,
        player_hand TEXT,
        pitches INTEGER,
        plate_appearances INTEGER,
        swings INTEGER,
        whiffs INTEGER,
        called_strikes INTEGER,
        in_zone_pitches INTEGER,
        out_zone_pitches INTEGER,
        chases INTEGER,
        bbe INTEGER,
        launch_speed_sum REAL,
        launch_speed_count INTEGER,
        max_exit_velocity REAL,
        launch_angle_sum REAL,
        launch_angle_count INTEGER,
        hard_hits INTEGER,
        barrels INTEGER,
        sweet_spot INTEGER,
        ground_balls INTEGER,
        fly_balls INTEGER,
        line_drives INTEGER,
        popups INTEGER,
        xba_sum REAL,
        xba_count INTEGER,
        xslg_sum REAL,
        xslg_count INTEGER,
        xwoba_sum REAL,
        xwoba_count INTEGER,
        release_speed_sum REAL,
        release_speed_count INTEGER,
        max_release_speed REAL,
        release_spin_sum REAL,
        release_spin_count INTEGER,
        pfx_x_sum REAL,
        pfx_z_sum REAL,
        movement_count INTEGER,
        PRIMARY KEY(game_pk, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS statcast_pitcher_game(
        game_pk INTEGER,
        game_date TEXT,
        player_id INTEGER,
        player_hand TEXT,
        pitches INTEGER,
        plate_appearances INTEGER,
        swings INTEGER,
        whiffs INTEGER,
        called_strikes INTEGER,
        in_zone_pitches INTEGER,
        out_zone_pitches INTEGER,
        chases INTEGER,
        bbe INTEGER,
        launch_speed_sum REAL,
        launch_speed_count INTEGER,
        max_exit_velocity REAL,
        launch_angle_sum REAL,
        launch_angle_count INTEGER,
        hard_hits INTEGER,
        barrels INTEGER,
        sweet_spot INTEGER,
        ground_balls INTEGER,
        fly_balls INTEGER,
        line_drives INTEGER,
        popups INTEGER,
        xba_sum REAL,
        xba_count INTEGER,
        xslg_sum REAL,
        xslg_count INTEGER,
        xwoba_sum REAL,
        xwoba_count INTEGER,
        release_speed_sum REAL,
        release_speed_count INTEGER,
        max_release_speed REAL,
        release_spin_sum REAL,
        release_spin_count INTEGER,
        pfx_x_sum REAL,
        pfx_z_sum REAL,
        movement_count INTEGER,
        PRIMARY KEY(game_pk, player_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS statcast_batter_pitch_type_game(
        game_pk INTEGER,
        game_date TEXT,
        player_id INTEGER,
        pitch_type TEXT,
        opponent_hand TEXT,
        player_hand TEXT,
        pitches INTEGER,
        plate_appearances INTEGER,
        swings INTEGER,
        whiffs INTEGER,
        called_strikes INTEGER,
        in_zone_pitches INTEGER,
        out_zone_pitches INTEGER,
        chases INTEGER,
        bbe INTEGER,
        launch_speed_sum REAL,
        launch_speed_count INTEGER,
        max_exit_velocity REAL,
        launch_angle_sum REAL,
        launch_angle_count INTEGER,
        hard_hits INTEGER,
        barrels INTEGER,
        sweet_spot INTEGER,
        ground_balls INTEGER,
        fly_balls INTEGER,
        line_drives INTEGER,
        popups INTEGER,
        xba_sum REAL,
        xba_count INTEGER,
        xslg_sum REAL,
        xslg_count INTEGER,
        xwoba_sum REAL,
        xwoba_count INTEGER,
        release_speed_sum REAL,
        release_speed_count INTEGER,
        max_release_speed REAL,
        release_spin_sum REAL,
        release_spin_count INTEGER,
        pfx_x_sum REAL,
        pfx_z_sum REAL,
        movement_count INTEGER,
        PRIMARY KEY(game_pk, player_id, pitch_type, opponent_hand)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS statcast_pitcher_pitch_type_game(
        game_pk INTEGER,
        game_date TEXT,
        player_id INTEGER,
        pitch_type TEXT,
        opponent_hand TEXT,
        player_hand TEXT,
        pitches INTEGER,
        plate_appearances INTEGER,
        swings INTEGER,
        whiffs INTEGER,
        called_strikes INTEGER,
        in_zone_pitches INTEGER,
        out_zone_pitches INTEGER,
        chases INTEGER,
        bbe INTEGER,
        launch_speed_sum REAL,
        launch_speed_count INTEGER,
        max_exit_velocity REAL,
        launch_angle_sum REAL,
        launch_angle_count INTEGER,
        hard_hits INTEGER,
        barrels INTEGER,
        sweet_spot INTEGER,
        ground_balls INTEGER,
        fly_balls INTEGER,
        line_drives INTEGER,
        popups INTEGER,
        xba_sum REAL,
        xba_count INTEGER,
        xslg_sum REAL,
        xslg_count INTEGER,
        xwoba_sum REAL,
        xwoba_count INTEGER,
        release_speed_sum REAL,
        release_speed_count INTEGER,
        max_release_speed REAL,
        release_spin_sum REAL,
        release_spin_count INTEGER,
        pfx_x_sum REAL,
        pfx_z_sum REAL,
        movement_count INTEGER,
        PRIMARY KEY(game_pk, player_id, pitch_type, opponent_hand)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_batting_player_date ON player_game_batting(player_id, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_pitching_player_date ON pitcher_game_stats(player_id, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_statcast_pitches_date ON statcast_pitches(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_statcast_pitches_batter ON statcast_pitches(batter, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_statcast_pitches_pitcher ON statcast_pitches(pitcher, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_statcast_batter_date ON statcast_batter_game(player_id, game_date)",
    "CREATE INDEX IF NOT EXISTS idx_statcast_pitcher_date ON statcast_pitcher_game(player_id, game_date)",
]


class Repository:
    def __init__(self, url: str):
        self.engine = create_engine(url, future=True)

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            for statement in SCHEMA:
                connection.execute(text(statement))
            self._migrate_player_game_batting(connection)
            self._migrate_statcast_pitcher_pitch_type(connection)

    @staticmethod
    def _migrate_player_game_batting(connection) -> None:
        existing_columns = {
            row[1]
            for row in connection.execute(
                text("PRAGMA table_info(player_game_batting)")
            ).fetchall()
        }
        migrations = {
            "opponent_id": "INTEGER",
            "opponent_pitcher_id": "INTEGER",
            "batting_order": "REAL",
        }
        for column_name, column_type in migrations.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(
                        f"ALTER TABLE player_game_batting "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )

    @staticmethod
    def _migrate_statcast_pitcher_pitch_type(connection) -> None:
        # A prerelease build briefly created an unused pfx_z column. Keeping this
        # migration harmless allows users to install over that build.
        existing = {
            row[1]
            for row in connection.execute(
                text("PRAGMA table_info(statcast_pitcher_pitch_type_game)")
            ).fetchall()
        }
        if existing and "pfx_z_sum" not in existing:
            connection.execute(
                text(
                    "ALTER TABLE statcast_pitcher_pitch_type_game "
                    "ADD COLUMN pfx_z_sum REAL"
                )
            )

    def _upsert(
        self,
        table: str,
        dataframe: pd.DataFrame,
        keys: list[str],
        chunk_size: int = 5000,
    ) -> int:
        if dataframe.empty:
            return 0

        columns = list(dataframe.columns)
        missing_keys = [key for key in keys if key not in columns]
        if missing_keys:
            raise ValueError(f"Missing upsert keys for {table}: {missing_keys}")

        update_columns = [column for column in columns if column not in keys]
        column_sql = ",".join(columns)
        values_sql = ",".join(f":{column}" for column in columns)
        conflict_sql = ",".join(keys)

        if update_columns:
            update_sql = ",".join(
                f"{column}=excluded.{column}" for column in update_columns
            )
            query = text(
                f"INSERT INTO {table} ({column_sql}) VALUES ({values_sql}) "
                f"ON CONFLICT({conflict_sql}) DO UPDATE SET {update_sql}"
            )
        else:
            query = text(
                f"INSERT INTO {table} ({column_sql}) VALUES ({values_sql}) "
                f"ON CONFLICT({conflict_sql}) DO NOTHING"
            )

        clean = dataframe.astype(object).where(pd.notna(dataframe), None)
        records = clean.to_dict("records")
        with self.engine.begin() as connection:
            for offset in range(0, len(records), chunk_size):
                connection.execute(query, records[offset : offset + chunk_size])
        return len(records)

    def upsert_games(self, dataframe: pd.DataFrame) -> int:
        return self._upsert("games", dataframe, ["game_pk"])

    def upsert_team_stats(self, dataframe: pd.DataFrame) -> int:
        return self._upsert("team_game_stats", dataframe, ["game_pk", "team_id"])

    def upsert_pitcher_stats(self, dataframe: pd.DataFrame) -> int:
        return self._upsert("pitcher_game_stats", dataframe, ["game_pk", "player_id"])

    def upsert_batting(self, dataframe: pd.DataFrame) -> int:
        return self._upsert("player_game_batting", dataframe, ["game_pk", "player_id"])

    def upsert_statcast_pitches(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "statcast_pitches",
            dataframe,
            ["game_pk", "at_bat_number", "pitch_number"],
        )

    def upsert_statcast_batters(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "statcast_batter_game", dataframe, ["game_pk", "player_id"]
        )

    def upsert_statcast_pitchers(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "statcast_pitcher_game", dataframe, ["game_pk", "player_id"]
        )

    def upsert_statcast_batter_pitch_types(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "statcast_batter_pitch_type_game",
            dataframe,
            ["game_pk", "player_id", "pitch_type", "opponent_hand"],
        )

    def upsert_statcast_pitcher_pitch_types(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "statcast_pitcher_pitch_type_game",
            dataframe,
            ["game_pk", "player_id", "pitch_type", "opponent_hand"],
        )

    def query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        return pd.read_sql(text(query), self.engine, params=params or {})

    def games_for_date(self, game_date) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM games WHERE game_date = :game_date ORDER BY game_time",
            {"game_date": game_date.isoformat()},
        )

    def games_between(self, start, end) -> pd.DataFrame:
        return self.query(
            """
            SELECT * FROM games
            WHERE game_date BETWEEN :start_date AND :end_date
            ORDER BY game_date, game_time
            """,
            {"start_date": start.isoformat(), "end_date": end.isoformat()},
        )

    def completed_games(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT * FROM games
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            """
        )

    def statcast_pitches_between(self, start, end) -> pd.DataFrame:
        return self.query(
            """
            SELECT * FROM statcast_pitches
            WHERE game_date BETWEEN :start_date AND :end_date
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            """,
            {"start_date": start.isoformat(), "end_date": end.isoformat()},
        )

    def coverage(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT
                (SELECT COUNT(*) FROM games) AS games,
                (SELECT COUNT(*) FROM team_game_stats) AS team_rows,
                (SELECT COUNT(*) FROM pitcher_game_stats) AS pitcher_rows,
                (SELECT COUNT(*) FROM player_game_batting) AS batter_rows,
                (SELECT COUNT(*) FROM statcast_pitches) AS statcast_pitches,
                (SELECT COUNT(*) FROM statcast_batter_game) AS statcast_batter_games,
                (SELECT COUNT(*) FROM statcast_pitcher_game) AS statcast_pitcher_games,
                (SELECT MIN(game_date) FROM statcast_pitches) AS statcast_start,
                (SELECT MAX(game_date) FROM statcast_pitches) AS statcast_end
            """
        )
