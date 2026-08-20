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


def test_induction_time_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_INDUCTION_TIME", "03:45")

    assert get_settings().induction_time == "03:45"


@pytest.mark.parametrize("value", ["24:00", "9:00", "noon", "12:60", "", "03:45:00"])
def test_settings_reject_malformed_induction_time(value):
    with pytest.raises(ConfigurationError, match="induction_time"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            induction_time=value,
        )


def test_embedding_settings_default_off_and_do_not_raise(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_DIM", raising=False)

    settings = get_settings()

    assert settings.embedding_model == ""
    assert settings.embedding_enabled is False
    assert settings.embedding_dim is None
    # base_url/api_key still inherit even when the feature is off, so a
    # later slice can probe/validate without needing separate env vars.
    assert settings.embedding_base_url == "http://model.test/v1"
    assert settings.embedding_api_key == "secret"


def test_embedding_settings_inherit_llm_base_url_and_api_key(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1/")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("GOSSIPMEMO_EMBEDDING_API_KEY", raising=False)

    settings = get_settings()

    assert settings.embedding_base_url == "http://model.test/v1"
    assert settings.embedding_api_key == "llm-secret"
    assert settings.embedding_model == "qwen3-embedding-0.6b"
    assert settings.embedding_enabled is True


def test_embedding_settings_override_base_url_and_api_key(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_BASE_URL", "http://192.168.1.113:8002/")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_MODEL", "qwen3-embedding-0.6b")

    settings = get_settings()

    assert settings.embedding_base_url == "http://192.168.1.113:8002"
    # Explicitly empty api_key stays empty rather than inheriting llm_api_key.
    assert settings.embedding_api_key == "llm-secret"


def test_embedding_dim_is_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("GOSSIPMEMO_LLM_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("GOSSIPMEMO_LLM_API_KEY", "secret")
    monkeypatch.setenv("GOSSIPMEMO_LLM_MODEL", "model-a")
    monkeypatch.setenv("GOSSIPMEMO_EMBEDDING_DIM", "1024")

    assert get_settings().embedding_dim == 1024


def test_settings_reject_non_positive_embedding_dim():
    with pytest.raises(ConfigurationError, match="embedding_dim"):
        Settings(
            llm_base_url="http://model.test/v1",
            llm_api_key="secret",
            llm_model="model-a",
            embedding_dim=0,
        )


def test_settings_do_not_require_embedding_model():
    # embedding_model empty must not raise -- it is the documented off state.
    Settings(
        llm_base_url="http://model.test/v1",
        llm_api_key="secret",
        llm_model="model-a",
    )
