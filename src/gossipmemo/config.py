from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_INDUCTION_TIME_PATTERN = re.compile(r"([01][0-9]|2[0-3]):[0-5][0-9]")


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
    user_name: str = "CurrentUser"
    database_path: Path = Path("data/gossipmemo.db")
    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str = ""
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 5
    llm_retry_base_seconds: float = 1.0
    llm_retry_max_seconds: float = 30.0
    llm_max_tokens: int | None = None
    extraction_batch_size: int = 6
    extraction_batch_timeout_seconds: float = 1800.0
    logging_level: str = "INFO"
    logging_format: str = "json"
    llm_context_window_tokens: int = 65536
    llm_output_reserve_tokens: int = 8192
    llm_context_safety_tokens: int = 4096
    llm_trace_path: Path | None = None
    prompts_path: Path | None = None
    induction_time: str = "00:00"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int | None = None
    embedding_query_timeout_seconds: float = 2.0

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
        if not self.user_name.strip():
            raise ConfigurationError("user_name must not be empty")
        if self.llm_timeout_seconds <= 0:
            raise ConfigurationError("llm_timeout_seconds must be greater than zero")
        if self.llm_max_retries < 0:
            raise ConfigurationError("llm_max_retries must not be negative")
        if self.llm_retry_base_seconds <= 0:
            raise ConfigurationError(
                "llm_retry_base_seconds must be greater than zero"
            )
        if self.llm_retry_max_seconds < self.llm_retry_base_seconds:
            raise ConfigurationError(
                "llm_retry_max_seconds must be at least llm_retry_base_seconds"
            )
        if self.llm_max_tokens is not None and self.llm_max_tokens < 1:
            raise ConfigurationError("llm_max_tokens must be greater than zero")
        if self.extraction_batch_size < 1:
            raise ConfigurationError("extraction_batch_size must be greater than zero")
        if self.extraction_batch_timeout_seconds <= 0:
            raise ConfigurationError(
                "extraction_batch_timeout_seconds must be greater than zero"
            )
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        if self.logging_level.upper() not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ConfigurationError(
                "logging_level must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG"
            )
        if self.prompts_path is not None and not self.prompts_path.is_file():
            raise ConfigurationError(f"prompts file does not exist: {self.prompts_path}")
        if not _INDUCTION_TIME_PATTERN.fullmatch(self.induction_time):
            raise ConfigurationError(
                "induction_time must be a 24-hour HH:MM value between 00:00 and 23:59"
            )
        if self.logging_format.lower() not in {"json", "text"}:
            raise ConfigurationError("logging_format must be json or text")
        if (
            self.llm_context_window_tokens
            - self.llm_output_reserve_tokens
            - self.llm_context_safety_tokens
        ) <= 0:
            raise ConfigurationError("LLM context usable input tokens must be greater than zero")
        if self.llm_output_reserve_tokens < 0 or self.llm_context_safety_tokens < 0:
            raise ConfigurationError(
                "LLM output reserve and context safety tokens must not be negative")
        if self.llm_max_tokens is not None and self.llm_output_reserve_tokens < self.llm_max_tokens:
            raise ConfigurationError("llm_output_reserve_tokens must be at least llm_max_tokens")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ConfigurationError("embedding_dim must be greater than zero")
        if self.embedding_query_timeout_seconds <= 0:
            raise ConfigurationError("embedding_query_timeout_seconds must be greater than zero")

    @property
    def embedding_enabled(self) -> bool:
        """Whether the embedding subsystem is configured at all.

        An empty `embedding_model` is a legitimate off state, not a
        configuration error -- the system falls back to plain FTS recall.
        """

        return bool(self.embedding_model.strip())

    @classmethod
    def from_env(cls) -> Settings:
        llm_base_url = _required_env("GOSSIPMEMO_LLM_BASE_URL").rstrip("/")
        llm_api_key = _required_env("GOSSIPMEMO_LLM_API_KEY")
        embedding_base_url = _env("GOSSIPMEMO_EMBEDDING_BASE_URL")
        embedding_api_key = _env("GOSSIPMEMO_EMBEDDING_API_KEY")
        return cls(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=_required_env("GOSSIPMEMO_LLM_MODEL"),
            user_name=_env("GOSSIPMEMO_USER_NAME", "CurrentUser"),
            database_path=Path(
                _env("GOSSIPMEMO_DATABASE_PATH", "data/gossipmemo.db")
            ),
            host=_env("GOSSIPMEMO_HOST", "127.0.0.1"),
            port=int(_env("GOSSIPMEMO_PORT", "8765")),
            api_key=_env("GOSSIPMEMO_API_KEY"),
            llm_timeout_seconds=float(
                _env("GOSSIPMEMO_LLM_TIMEOUT_SECONDS", "120")
            ),
            llm_max_retries=int(_env("GOSSIPMEMO_LLM_MAX_RETRIES", "5")),
            llm_retry_base_seconds=float(
                _env("GOSSIPMEMO_LLM_RETRY_BASE_SECONDS", "1")
            ),
            llm_retry_max_seconds=float(
                _env("GOSSIPMEMO_LLM_RETRY_MAX_SECONDS", "30")
            ),
            llm_max_tokens=(int(value) if (value := _env("GOSSIPMEMO_LLM_MAX_TOKENS")) else None),
            extraction_batch_size=int(
                _env("GOSSIPMEMO_EXTRACTION_BATCH_SIZE", "6")
            ),
            extraction_batch_timeout_seconds=float(
                _env("GOSSIPMEMO_EXTRACTION_BATCH_TIMEOUT_SECONDS", "1800")
            ),
            logging_level=_env("GOSSIPMEMO_LOG_LEVEL", "INFO").upper(),
            logging_format=_env("GOSSIPMEMO_LOG_FORMAT", "json").lower(),
            llm_context_window_tokens=int(_env("GOSSIPMEMO_LLM_CONTEXT_WINDOW_TOKENS", "65536")),
            llm_output_reserve_tokens=int(_env("GOSSIPMEMO_LLM_OUTPUT_RESERVE_TOKENS", "8192")),
            llm_context_safety_tokens=int(_env("GOSSIPMEMO_LLM_CONTEXT_SAFETY_TOKENS", "4096")),
            llm_trace_path=(Path(value) if (value := _env("GOSSIPMEMO_LLM_TRACE_PATH")) else None),
            prompts_path=(Path(value) if (value := _env("GOSSIPMEMO_PROMPTS_PATH")) else None),
            induction_time=_env("GOSSIPMEMO_INDUCTION_TIME", "00:00"),
            embedding_base_url=(
                embedding_base_url.rstrip("/") if embedding_base_url else llm_base_url
            ),
            embedding_api_key=(embedding_api_key if embedding_api_key else llm_api_key),
            embedding_model=_env("GOSSIPMEMO_EMBEDDING_MODEL"),
            embedding_dim=(int(value) if (value := _env("GOSSIPMEMO_EMBEDDING_DIM")) else None),
            embedding_query_timeout_seconds=float(
                _env("GOSSIPMEMO_EMBEDDING_QUERY_TIMEOUT_SECONDS", "2")
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read environment configuration once and distribute the immutable value."""

    return Settings.from_env()


__all__ = ["ConfigurationError", "Settings", "get_settings"]
