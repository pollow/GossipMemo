from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal


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
    extraction_policy: Literal[
        "conservative", "balanced", "comprehensive"
    ] = "balanced"
    logging_level: str = "INFO"
    logging_format: str = "json"
    llm_context_window_tokens: int = 65536
    llm_output_reserve_tokens: int = 8192
    llm_context_safety_tokens: int = 4096

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
        if self.extraction_policy not in {
            "conservative",
            "balanced",
            "comprehensive",
        }:
            raise ConfigurationError(
                "extraction_policy must be conservative, balanced, or comprehensive"
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
        if self.logging_format.lower() not in {"json", "text"}:
            raise ConfigurationError("logging_format must be json or text")
        if self.llm_context_window_tokens - self.llm_output_reserve_tokens - self.llm_context_safety_tokens <= 0:
            raise ConfigurationError("LLM context usable input tokens must be greater than zero")
        if self.llm_output_reserve_tokens < 0 or self.llm_context_safety_tokens < 0:
            raise ConfigurationError("LLM output reserve and context safety tokens must not be negative")
        if self.llm_max_tokens is not None and self.llm_output_reserve_tokens < self.llm_max_tokens:
            raise ConfigurationError("llm_output_reserve_tokens must be at least llm_max_tokens")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=_required_env("GOSSIPMEMO_LLM_BASE_URL").rstrip("/"),
            llm_api_key=_required_env("GOSSIPMEMO_LLM_API_KEY"),
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
            extraction_policy=_env("GOSSIPMEMO_EXTRACTION_POLICY", "balanced"),
            logging_level=_env("GOSSIPMEMO_LOG_LEVEL", "INFO").upper(),
            logging_format=_env("GOSSIPMEMO_LOG_FORMAT", "json").lower(),
            llm_context_window_tokens=int(_env("GOSSIPMEMO_LLM_CONTEXT_WINDOW_TOKENS", "65536")),
            llm_output_reserve_tokens=int(_env("GOSSIPMEMO_LLM_OUTPUT_RESERVE_TOKENS", "8192")),
            llm_context_safety_tokens=int(_env("GOSSIPMEMO_LLM_CONTEXT_SAFETY_TOKENS", "4096")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read environment configuration once and distribute the immutable value."""

    return Settings.from_env()


__all__ = ["ConfigurationError", "Settings", "get_settings"]
