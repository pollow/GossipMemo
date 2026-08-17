from __future__ import annotations

import json
import logging

import httpx
import pytest

from gossipmemo.app import create_app
from gossipmemo.config import ConfigurationError, Settings
from gossipmemo.logging import StructuredFormatter, request_id_context


def _settings(**overrides):
    values = {
        "llm_base_url": "http://model.test/v1",
        "llm_api_key": "secret",
        "llm_model": "model-a",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    "field,value", [("logging_level", "verbose"), ("logging_format", "xml")]
)
def test_logging_configuration_is_validated(field, value):
    with pytest.raises(ConfigurationError, match=field):
        _settings(**{field: value})


def test_json_formatter_includes_request_id_but_not_message_content():
    token = request_id_context.set("req-123")
    try:
        record = logging.getLogger("test").makeRecord(
            "test", logging.INFO, __file__, 1, "ingest_completed", (), None,
            extra={"space_id": "space", "message_count": 2, "api_key": "secret"},
        )
        payload = json.loads(StructuredFormatter().format(record))
    finally:
        request_id_context.reset(token)
    assert payload["event"] == "ingest_completed"
    assert payload["request_id"] == "req-123"
    assert payload["message_count"] == 2
    assert "content" not in payload
    assert "api_key" not in payload


@pytest.mark.asyncio
async def test_http_request_id_is_preserved_or_replaced():
    app = create_app(_settings(), world=object())  # Lifespan is not entered here.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        preserved = await client.get("/missing", headers={"X-Request-ID": "req-123"})
        invalid = "x" * 129
        replaced = await client.get("/missing", headers={"X-Request-ID": invalid})

    assert preserved.headers["X-Request-ID"] == "req-123"
    assert replaced.headers["X-Request-ID"] != invalid
