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
    monkeypatch.setenv("GOSSIPMEMO_EXTRACTION_POLICY", "comprehensive")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MAX_RETRIES", "7")
    monkeypatch.setenv("GOSSIPMEMO_LLM_RETRY_BASE_SECONDS", "2.5")
    monkeypatch.setenv("GOSSIPMEMO_LLM_RETRY_MAX_SECONDS", "45")

    first = get_settings()
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-b")
    second = get_settings()

    assert first is second
    assert second.llm_base_url == "http://model.test/v1"
    assert second.llm_model == "model-a"
    assert second.database_path == tmp_path / "world.db"
    assert second.extraction_policy == "comprehensive"
    assert second.user_name == "CurrentUser"
    assert second.llm_max_retries == 7
    assert second.llm_retry_base_seconds == 2.5
    assert second.llm_retry_max_seconds == 45


def test_user_name_is_loaded_and_must_not_be_empty(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_USER_NAME", "  Deus  ")

    assert get_settings().user_name == "Deus"
    with pytest.raises(ConfigurationError, match="user_name"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            user_name=" ",
        )


def test_settings_reject_empty_llm_values():
    with pytest.raises(ConfigurationError, match="llm_api_key"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="",
            llm_model="model-a",
        )


def test_settings_reject_invalid_extraction_policy():
    with pytest.raises(ConfigurationError, match="extraction_policy"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            extraction_policy="aggressive",  # type: ignore[arg-type]
        )


def test_settings_reject_negative_max_retries():
    with pytest.raises(ConfigurationError, match="llm_max_retries"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            llm_max_retries=-1,
        )


def test_output_reserve_must_cover_configured_max_tokens():
    with pytest.raises(ConfigurationError, match="output_reserve_tokens"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            llm_max_tokens=9000,
        )


def test_settings_reject_retry_max_below_base():
    with pytest.raises(ConfigurationError, match="llm_retry_max_seconds"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            llm_retry_base_seconds=10,
            llm_retry_max_seconds=5,
        )


def test_serve_command_exits_before_uvicorn_when_config_is_missing(
    monkeypatch,
):
    for name in REQUIRED_LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["gossipmemo", "serve"])

    with pytest.raises(SystemExit, match="GossipMemo configuration error"):
        main()
