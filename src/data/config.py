from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    raw_data_dir: Path = Field(default_factory=lambda: Path("data"))
    artifacts_dir: Path = Field(default_factory=lambda: Path("artifacts"))
    batch_size: int = 50_000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    timestamp_column: str = "time"

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> DataSettings:
    settings = DataSettings()
    Path(settings.raw_data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.artifacts_dir).mkdir(parents=True, exist_ok=True)
    return settings
