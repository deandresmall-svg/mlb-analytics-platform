from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "MLB_DATABASE_URL", "sqlite:///data/mlb_analytics.db"
    )
    api_base_url: str = os.getenv(
        "MLB_API_BASE_URL", "https://statsapi.mlb.com/api/v1"
    )
    weather_base_url: str = os.getenv(
        "WEATHER_BASE_URL", "https://api.open-meteo.com/v1"
    )
    weather_archive_url: str = os.getenv(
        "WEATHER_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
    )
    api_timeout_seconds: float = float(os.getenv("API_TIMEOUT_SECONDS", "20"))
    api_max_retries: int = int(os.getenv("API_MAX_RETRIES", "3"))
    calibration_method: str = os.getenv("CALIBRATION_METHOD", "sigmoid")
    model_dir: Path = Path(os.getenv("MODEL_DIR", "models"))
    reliability_snapshot_path: Path = Path(
        os.getenv("RELIABILITY_SNAPSHOT_PATH", "data/reliability_snapshot.json")
    )
    min_training_games: int = int(os.getenv("MIN_TRAINING_GAMES", "300"))
    default_backfill_days: int = int(os.getenv("DEFAULT_BACKFILL_DAYS", "365"))

    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    odds_api_base_url: str = os.getenv(
        "ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4"
    )
    odds_regions: str = os.getenv("ODDS_REGIONS", "us,us2")
    odds_bookmakers: str = os.getenv("ODDS_BOOKMAKERS", "")

    statcast_base_url: str = os.getenv(
        "STATCAST_BASE_URL",
        "https://baseballsavant.mlb.com/statcast_search/csv",
    )
    statcast_timeout_seconds: float = float(
        os.getenv("STATCAST_TIMEOUT_SECONDS", "90")
    )
    statcast_chunk_days: int = int(os.getenv("STATCAST_CHUNK_DAYS", "3"))
    statcast_default_sync_days: int = int(
        os.getenv("STATCAST_DEFAULT_SYNC_DAYS", "30")
    )

    def ensure_directories(self) -> None:
        Path("data").mkdir(exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reliability_snapshot_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
