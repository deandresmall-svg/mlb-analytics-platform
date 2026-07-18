from __future__ import annotations

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
]


class Repository:
    def __init__(self, url: str):
        self.engine = create_engine(url, future=True)

    def initialize(self) -> None:
        with self.engine.begin() as connection:
            for statement in SCHEMA:
                connection.execute(text(statement))

            self._migrate_player_game_batting(connection)

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
            if column_name in existing_columns:
                continue

            connection.execute(
                text(
                    f"""
                    ALTER TABLE player_game_batting
                    ADD COLUMN {column_name} {column_type}
                    """
                )
            )

    def _upsert(
        self,
        table: str,
        dataframe: pd.DataFrame,
        keys: list[str],
    ) -> int:
        if dataframe.empty:
            return 0

        columns = list(dataframe.columns)
        update_columns = [
            column
            for column in columns
            if column not in keys
        ]

        column_sql = ",".join(columns)
        values_sql = ",".join(f":{column}" for column in columns)
        conflict_sql = ",".join(keys)

        if update_columns:
            update_sql = ",".join(
                f"{column}=excluded.{column}"
                for column in update_columns
            )

            query = text(
                f"""
                INSERT INTO {table} ({column_sql})
                VALUES ({values_sql})
                ON CONFLICT({conflict_sql})
                DO UPDATE SET {update_sql}
                """
            )
        else:
            query = text(
                f"""
                INSERT INTO {table} ({column_sql})
                VALUES ({values_sql})
                ON CONFLICT({conflict_sql})
                DO NOTHING
                """
            )

        clean_dataframe = dataframe.copy()

        for column in clean_dataframe.columns:
            clean_dataframe[column] = clean_dataframe[column].where(
                clean_dataframe[column].notna(),
                None,
            )

        records = clean_dataframe.to_dict("records")

        with self.engine.begin() as connection:
            connection.execute(query, records)

        return len(clean_dataframe)

    def upsert_games(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "games",
            dataframe,
            ["game_pk"],
        )

    def upsert_team_stats(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "team_game_stats",
            dataframe,
            ["game_pk", "team_id"],
        )

    def upsert_pitcher_stats(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "pitcher_game_stats",
            dataframe,
            ["game_pk", "player_id"],
        )

    def upsert_batting(self, dataframe: pd.DataFrame) -> int:
        return self._upsert(
            "player_game_batting",
            dataframe,
            ["game_pk", "player_id"],
        )

    def query(
        self,
        query: str,
        params: dict | None = None,
    ) -> pd.DataFrame:
        return pd.read_sql(
            text(query),
            self.engine,
            params=params or {},
        )

    def games_for_date(self, game_date) -> pd.DataFrame:
        return self.query(
            """
            SELECT *
            FROM games
            WHERE game_date = :game_date
            ORDER BY game_time
            """,
            {"game_date": game_date.isoformat()},
        )

    def games_between(self, start, end) -> pd.DataFrame:
        return self.query(
            """
            SELECT *
            FROM games
            WHERE game_date BETWEEN :start_date AND :end_date
            ORDER BY game_date, game_time
            """,
            {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )

    def completed_games(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT *
            FROM games
            WHERE home_score IS NOT NULL
              AND away_score IS NOT NULL
            """
        )

    def coverage(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT
                (SELECT COUNT(*) FROM games) AS games,
                (SELECT COUNT(*) FROM team_game_stats) AS team_rows,
                (SELECT COUNT(*) FROM pitcher_game_stats) AS pitcher_rows,
                (SELECT COUNT(*) FROM player_game_batting) AS batter_rows
            """
        )