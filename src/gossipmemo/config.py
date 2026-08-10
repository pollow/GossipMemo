from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_path=Path(
                _env("GOSSIPMEMO_DATABASE_PATH", "data/gossipmemo.db")
            ),
            host=_env("GOSSIPMEMO_HOST", "127.0.0.1"),
            port=int(_env("GOSSIPMEMO_PORT", "8765")),
            api_key=_env("GOSSIPMEMO_API_KEY"),
            llm_base_url=_env(
                "GOSSIPMEMO_LLM_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            llm_api_key=_env("GOSSIPMEMO_LLM_API_KEY"),
            llm_model=_env("GOSSIPMEMO_LLM_MODEL"),
            llm_timeout_seconds=float(
                _env("GOSSIPMEMO_LLM_TIMEOUT_SECONDS", "120")
            ),
        )
