from __future__ import annotations

import sys

import pytest

from gossipmemo.cli import main
from gossipmemo.config import ConfigurationError, Settings, get_settings


REQUIRED_LLM_ENV = (
    "GOSSIPMEMO_LLM_BASE_URL",
    "GOSSIPMEMO_LLM_API_KEY",
    "GOSSIPMEMO_LLM_MODEL",
)


@pytest.fixture(autouse=True)
def clear_global_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_llm_environment_refuses_configuration(monkeypatch):
    for name in REQUIRED_LLM_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="GOSSIPMEMO_LLM_BASE_URL"):
        get_settings()


def test_global_settings_are_loaded_from_environment_once(monkeypatch, tmp_path):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1/")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_DATABASE_PATH", str(tmp_path / "world.db"))

    first = get_settings()
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-b")
    second = get_settings()

    assert first is second
    assert second.llm_base_url == "http://model.test/v1"
    assert second.llm_model == "model-a"
    assert second.database_path == tmp_path / "world.db"


def test_settings_reject_empty_llm_values():
    with pytest.raises(ConfigurationError, match="llm_api_key"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="",
            llm_model="model-a",
        )


def test_serve_command_exits_before_uvicorn_when_config_is_missing(
    monkeypatch,
):
    for name in REQUIRED_LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["gossipmemo", "serve"])

    with pytest.raises(SystemExit, match="GossipMemo configuration error"):
        main()
