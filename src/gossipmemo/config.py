from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised before startup when required server configuration is absent."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _required_env(name: str) -> str:
    value = _env(name)
    if not value:
        raise ConfigurationError(f"required environment variable is missing: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    database_path: Path = Path("data/gossipmemo.db")
    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str = ""
    llm_timeout_seconds: float = 120.0
    extraction_batch_size: int = 6
    extraction_batch_timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("llm_base_url", self.llm_base_url),
                ("llm_api_key", self.llm_api_key),
                ("llm_model", self.llm_model),
            )
            if not value.strip()
        ]
        if missing:
            raise ConfigurationError(
                "required settings are empty: " + ", ".join(missing)
            )
        if self.llm_timeout_seconds <= 0:
            raise ConfigurationError("llm_timeout_seconds must be greater than zero")
        if self.extraction_batch_size < 1:
            raise ConfigurationError("extraction_batch_size must be greater than zero")
        if self.extraction_batch_timeout_seconds <= 0:
            raise ConfigurationError(
                "extraction_batch_timeout_seconds must be greater than zero"
            )
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=_required_env("GOSSIPMEMO_LLM_BASE_URL").rstrip("/"),
            llm_api_key=_required_env("GOSSIPMEMO_LLM_API_KEY"),
            llm_model=_required_env("GOSSIPMEMO_LLM_MODEL"),
            database_path=Path(
                _env("GOSSIPMEMO_DATABASE_PATH", "data/gossipmemo.db")
            ),
            host=_env("GOSSIPMEMO_HOST", "127.0.0.1"),
            port=int(_env("GOSSIPMEMO_PORT", "8765")),
            api_key=_env("GOSSIPMEMO_API_KEY"),
            llm_timeout_seconds=float(
                _env("GOSSIPMEMO_LLM_TIMEOUT_SECONDS", "120")
            ),
            extraction_batch_size=int(
                _env("GOSSIPMEMO_EXTRACTION_BATCH_SIZE", "6")
            ),
            extraction_batch_timeout_seconds=float(
                _env("GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS", "1800")
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read environment configuration once and distribute the immutable value."""

    return Settings.from_env()


__all__ = ["ConfigurationError", "Settings", "get_settings"]
