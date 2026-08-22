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
    monkeypatch.setenv("GOSSIPMEMO_LLM_MAX_RETRIES", "7")
    monkeypatch.setenv("GOSSIPMEMO_LLM_RETRY_BASE_SECONDS", "2.5")
    monkeypatch.setenv("GOSSIPMEMO_LLM_RETRY_MAX_SECONDS", "45")
    monkeypatch.setenv("GOSSIPMEMO_INDUCTION_TIME", "03:45")

    first = get_settings()
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-b")
    second = get_settings()

    assert first is second
    assert second.llm_base_url == "http://model.test/v1"
    assert second.llm_model == "model-a"
    assert second.database_path == tmp_path / "world.db"
    assert second.user_name == "CurrentUser"
    assert second.llm_max_retries == 7
    assert second.llm_retry_base_seconds == 2.5
    assert second.llm_retry_max_seconds == 45
    assert second.induction_time == "03:45"


def test_user_name_is_trimmed_when_loaded(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_USER_NAME", "  Deus  ")

    assert get_settings().user_name == "Deus"


@pytest.mark.parametrize(
    "field,kwargs",
    [
        ("llm_api_key", {"llm_api_key": ""}),
        ("llm_max_retries", {"llm_max_retries": -1}),
        ("llm_retry_max_seconds", {"llm_retry_base_seconds": 10, "llm_retry_max_seconds": 5}),
        ("output_reserve_tokens", {"llm_max_tokens": 9000}),
        ("user_name", {"user_name": " "}),
        ("embedding_dim", {"embedding_dim": 0}),
        ("embedding_query_timeout_seconds", {"embedding_query_timeout_seconds": 0}),
        *(("induction_time", {"induction_time": value})
          for value in ("24:00", "9:00", "noon", "12:60", "", "03:45:00")),
    ],
)
def test_settings_reject_invalid_values(field, kwargs):
    base = {
        "llm_base_url": "http://model.test/v1",
        "llm_api_key": "secret",
        "llm_model": "model-a",
    }
    with pytest.raises(ConfigurationError, match=field):
        Settings(**{**base, **kwargs})


def test_serve_command_exits_before_uvicorn_when_config_is_missing(
    monkeypatch,
):
    for name in REQUIRED_LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["gossipmemo", "serve"])

    with pytest.raises(SystemExit, match="GossipMemo configuration error"):
        main()


def _llm_env(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1/")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    for name in (
        "GOSSIPMEMO_EMBEDDING_BASE_URL",
        "GOSSIPMEMO_EMBEDDING_API_KEY",
        "GOSSIPMEMO_EMBEDDING_MODEL",
        "GOSSIPMEMO_EMBEDDING_DIM",
        "GOSSIPMEMO_EMBEDDING_QUERY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_embedding_defaults_are_off_but_still_inherit_the_llm_endpoint(monkeypatch):
    """An empty `embedding_model` is the documented off state and must not raise."""
    _llm_env(monkeypatch)

    settings = get_settings()

    assert settings.embedding_model == ""
    assert settings.embedding_enabled is False
    assert settings.embedding_dim is None
    assert settings.embedding_query_timeout_seconds == 2.0
    # base_url/api_key inherit even when the feature is off, so turning it on
    # needs one env var rather than three.
    assert settings.embedding_base_url == "http://model.test/v1"
    assert settings.embedding_api_key == "llm-secret"


def test_embedding_settings_are_loaded_and_override_the_inherited_endpoint(monkeypatch):
    _llm_env(monkeypatch)
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_BASE_URL", "http://192.168.1.113:8002/")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_DIM", "1024")
    monkeypatch.setenv("GOSSIPMEMO_LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_QUERY_TIMEOUT_SECONDS", "1.5")

    settings = get_settings()

    assert settings.embedding_enabled is True
    assert settings.embedding_model == "qwen3-embedding-0.6b"
    assert settings.embedding_base_url == "http://192.168.1.113:8002"
    assert settings.embedding_dim == 1024
    # The query timeout is deliberately independent of the LLM timeout.
    assert settings.embedding_query_timeout_seconds == 1.5
    assert settings.llm_timeout_seconds == 120.0
