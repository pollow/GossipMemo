"""Hermes MemoryProvider integration for a GossipMemo server.

The plugin is intentionally self-contained.  Hermes is an optional runtime
dependency, so importing this module while developing GossipMemo (or running
its SDK tests) does not require Hermes to be installed.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gossipmemo_client import GossipMemo, GossipMemoError

try:  # Hermes supplies the real ABC when the plugin is loaded by Hermes.
    from agent.memory_provider import MemoryProvider
except ImportError:  # pragma: no cover - exercised only outside Hermes.
    class MemoryProvider:  # type: ignore[no-redef]
        """Small import-time fallback for SDK/plugin development."""

        pass


logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8765"
_DEFAULT_SPACE_ID = "personal"
_CONFIG_FILENAME = "gossipmemo.json"
_ENV_BASE_URL = ("GOSSIPMEMO_BASE_URL", "GOSSIPMEMO_URL")
_ENV_API_KEY = "GOSSIPMEMO_API_KEY"
_ENV_SPACE_ID = "GOSSIPMEMO_SPACE_ID"


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _load_config(hermes_home: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not hermes_home:
        return {}
    path = Path(hermes_home) / _CONFIG_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _string_config(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key)
    return str(value).strip() if value is not None and str(value).strip() else default


def _json_result(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"error": str(value)}, ensure_ascii=False)


def _compact(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _schema(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": dict(properties),
            "required": list(required),
            "additionalProperties": False,
        },
    }


class GossipMemoMemoryProvider(MemoryProvider):
    """A thin, provenance-preserving bridge to the GossipMemo HTTP server."""

    def __init__(
        self,
        *,
        client_factory: Callable[..., GossipMemo] | None = None,
    ) -> None:
        # Resolve the module-level class at construction time so embedding
        # applications (and tests) can replace the transport cleanly.
        self._client_factory = client_factory if client_factory is not None else GossipMemo
        self._client: GossipMemo | None = None
        self._session_id = ""
        self._user_id = "hermes-user"
        self._space_id = _DEFAULT_SPACE_ID
        self._source_provider = "hermes"
        self._write_enabled = True
        self._queue: queue.Queue[list[dict[str, Any]] | None] | None = None
        self._writer: threading.Thread | None = None
        self._stop_requested = False
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache: dict[str, dict[str, Any]] = {}
        self._context_cache: dict[str, dict[str, Any]] = {}
        self._pending_turns: dict[str, list[dict[str, str]]] = {}
        self._prefetch_threads: list[threading.Thread] = []

    @property
    def name(self) -> str:
        return "gossipmemo"

    def is_available(self) -> bool:
        """Check local configuration only; do not make a network request."""

        config = _load_config(os.environ.get("HERMES_HOME"))
        configured_url = _env_first(*_ENV_BASE_URL) or str(config.get("base_url", "")).strip()
        # A local unauthenticated server is valid, but an unconfigured plugin
        # should not activate merely because localhost is the SDK default.
        configured = bool(configured_url or _env_first(_ENV_API_KEY))
        space_id = _env_first(_ENV_SPACE_ID) or _string_config(
            config, "space_id", _DEFAULT_SPACE_ID
        )
        return configured and bool(space_id)

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Fields used by ``hermes memory setup``.

        The API key is optional because a local GossipMemo server can be
        intentionally unauthenticated.  If a key is used, setup writes it to
        the environment rather than the profile JSON file.
        """

        return [
            {
                "key": "base_url",
                "description": "GossipMemo server URL",
                "default": _DEFAULT_BASE_URL,
                "env_var": "GOSSIPMEMO_BASE_URL",
            },
            {
                "key": "space_id",
                "description": "GossipMemo memory space",
                "default": _DEFAULT_SPACE_ID,
                "env_var": "GOSSIPMEMO_SPACE_ID",
            },
            {
                "key": "api_key",
                "description": "GossipMemo API key (optional for local servers)",
                "secret": True,
                "required": False,
                "env_var": _ENV_API_KEY,
            },
        ]

    def save_config(self, values: Mapping[str, Any], hermes_home: str) -> None:
        """Persist non-secret settings in the active Hermes profile.

        Secrets are deliberately omitted even if a caller accidentally passes
        them in ``values``; Hermes setup normally sends secrets separately to
        its environment-file writer.
        """

        path = Path(hermes_home) / _CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_config(hermes_home)
        for key in ("base_url", "space_id"):
            value = values.get(key)
            if value is not None and str(value).strip():
                existing[key] = str(value).strip()
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = str(kwargs.get("hermes_home", "") or "")
        config_value = kwargs.get("config")
        config = dict(config_value) if isinstance(
            config_value, Mapping) else _load_config(hermes_home)

        base_url = (
            _env_first(*_ENV_BASE_URL)
            or str(kwargs.get("base_url", "")).strip()
            or _string_config(config, "base_url", _DEFAULT_BASE_URL)
        )
        api_key = (
            _env_first(_ENV_API_KEY)
            or str(kwargs.get("api_key", "")).strip()
        )
        self._space_id = (
            _env_first(_ENV_SPACE_ID)
            or str(kwargs.get("space_id", "")).strip()
            or _string_config(config, "space_id", _DEFAULT_SPACE_ID)
        )
        self._session_id = str(session_id or "")
        self._user_id = str(
            kwargs.get("user_id") or kwargs.get("agent_identity") or "hermes-user"
        )
        self._source_provider = str(kwargs.get("source_provider") or "hermes")
        # Hermes advises external providers not to ingest cron/subagent system
        # prompts as the primary user's memories.
        context = str(kwargs.get("agent_context", "primary") or "primary")
        self._write_enabled = context in {"", "primary"}
        self._stop_requested = False
        self._prefetch_cache.clear()
        self._context_cache.clear()
        self._pending_turns.clear()
        try:
            self._client = self._client_factory(
                base_url=base_url,
                api_key=api_key,
                space_id=self._space_id,
            )
        except TypeError:
            # A tiny positional fallback is useful for adapters that wrap the
            # SDK client with a three-argument factory.
            self._client = self._client_factory(base_url, api_key, self._space_id)
        self._queue = queue.Queue()
        self._writer = threading.Thread(
            target=self._write_loop,
            name="gossipmemo-sync",
            daemon=True,
        )
        self._writer.start()

    def system_prompt_block(self) -> str:
        return (
            "# GossipMemo memory\n"
            "GossipMemo keeps provenance-aware memories about people and relationships. "
            "Use gossipmemo_search before relying on social context, gossipmemo_store "
            "for explicit durable facts, gossipmemo_dossier for a person's current "
            "profile, and gossipmemo_retract to correct a memory."
        )

    def _conversation_key(self, session_id: str = "") -> str | None:
        value = session_id or self._session_id
        return value or None

    def _turn_messages(
        self,
        user_content: str,
        assistant_content: str,
        session_id: str = "",
    ) -> list[dict[str, Any]]:
        conversation_key = self._conversation_key(session_id)
        now = datetime.now(timezone.utc).isoformat()
        source_base: dict[str, Any] = {
            "provider": self._source_provider,
            "conversation_key": conversation_key,
        }
        user_source = dict(source_base)
        user_source["metadata"] = {"role": "user"}
        assistant_source = dict(source_base)
        assistant_source["metadata"] = {"role": "assistant"}
        return [
            {
                "author": "user",
                "content": user_content,
                "occurred_at": now,
                "source": user_source,
            },
            {
                "author": "assistant",
                "content": assistant_content,
                "occurred_at": now,
                "source": assistant_source,
            },
        ]

    def _write_loop(self) -> None:
        while True:
            work_queue = self._queue
            if work_queue is None:
                return
            batch = work_queue.get()
            try:
                if batch is None:
                    return
                client = self._client
                if client is not None:
                    client.ingest(batch)
            except Exception as exc:  # noqa: BLE001 - a daemon must stay alive.
                logger.warning("GossipMemo turn sync failed: %s", exc)
            finally:
                work_queue.task_done()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Queue a completed turn and return without waiting for HTTP."""

        if not self._write_enabled or not self._client or not self._queue:
            return
        if not user_content or not assistant_content:
            return
        # ``messages`` may include tool-call payloads and workspace output.  A
        # first-version bridge stores the user/assistant turn itself, while the
        # optional argument keeps the ABC signature forward-compatible.
        del messages
        conversation = session_id or self._session_id or "default"
        with self._prefetch_lock:
            pending = self._pending_turns.get(conversation, [])
            idem = ""
            for index, item in enumerate(pending):
                if item.get("content") == user_content:
                    idem = item.get("idempotency_key", "")
                    del pending[index]
                    break
        if not idem:
            idem = uuid.uuid4().hex
        messages = self._turn_messages(user_content, assistant_content, session_id)
        messages[0]["idempotency_key"] = idem
        # The same stable key is used by the turn façade and this asynchronous
        # assistant write, so a slow prefetch cannot create a duplicate user row.
        self._queue.put(messages)

    def _turn_context(self, query: str, key: str) -> tuple[str, str]:
        client = self._client
        if client is None:
            return "", ""
        with self._prefetch_lock:
            cached = dict(self._context_cache.get(key, {}))
            pending = self._pending_turns.setdefault(key, [])
            entry = next((item for item in pending if item.get("content") == query), None)
            if entry is None:
                entry = {"content": query, "idempotency_key": uuid.uuid4().hex, "started": False}
                pending.append(entry)
            idem = entry["idempotency_key"]
            event = entry.setdefault("event", threading.Event())
            if entry.get("started"):
                started = False
            else:
                entry["started"] = True
                started = True
        if not started:
            event.wait(timeout=0.35)
            with self._prefetch_lock:
                return str(entry.get("formatted", "")), idem
        try:
            result = client.turn(
                query,
                context_version=cached.get("version"),
                memory_limit=5,
                idempotency_key=idem,
                source={"provider": self._source_provider, "conversation_key": key},
            )
        except Exception:
            with self._prefetch_lock:
                entry["event"].set()
            raise
        if not isinstance(result, Mapping):
            with self._prefetch_lock:
                entry["event"].set()
            return "", idem
        update = result.get("context_update")
        with self._prefetch_lock:
            if isinstance(update, Mapping):
                self._context_cache[key] = dict(update)
            elif cached:
                update = cached
        merged = {
            "bundle": update or cached,
            "known_people": result.get("known_people", []),
            "memory_recall": result.get("memory_recall", []),
            "guidance": result.get("guidance", {}),
        }
        formatted = self._format_context(merged)
        with self._prefetch_lock:
            entry["formatted"] = formatted
            entry["event"].set()
        return formatted, idem

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm a small query result for the next turn in a daemon thread."""

        client = self._client
        if not client or not query or not query.strip():
            return
        key = session_id or self._session_id or "default"

        def _prefetch() -> None:
            try:
                try:
                    formatted, _ = self._turn_context(query, key)
                except AttributeError:
                    return
                if formatted:
                    with self._prefetch_lock:
                        self._prefetch_cache[key] = {"formatted": formatted}
            except Exception as exc:  # noqa: BLE001 - recall is non-fatal.
                logger.debug("GossipMemo prefetch failed: %s", exc)

        thread = threading.Thread(target=_prefetch, name="gossipmemo-prefetch", daemon=True)
        self._prefetch_threads = [item for item in self._prefetch_threads if item.is_alive()]
        self._prefetch_threads.append(thread)
        thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = session_id or self._session_id or "default"
        with self._prefetch_lock:
            cached_entry = self._prefetch_cache.pop(key, {})
        cached = cached_entry.get("formatted", "") if isinstance(cached_entry, Mapping) else ""
        if cached or not query.strip() or not self._client:
            return cached

        # Hermes calls prefetch before the first queue_prefetch opportunity.
        # Give a fast local server a small synchronous window so turn one can
        # use context, while a slow server remains non-blocking and deposits
        # its result in the cache for the next turn.
        result: list[str] = []

        def _first_fetch() -> None:
            try:
                try:
                    formatted, _ = self._turn_context(query, key)
                except AttributeError:
                    return
                if formatted:
                    result.append(formatted)
                    with self._prefetch_lock:
                        self._prefetch_cache[key] = {"formatted": formatted}
            except Exception as exc:  # noqa: BLE001 - recall is non-fatal.
                logger.debug("GossipMemo first-turn prefetch failed: %s", exc)

        thread = threading.Thread(
            target=_first_fetch, name="gossipmemo-prefetch-first", daemon=True)
        thread.start()
        thread.join(timeout=0.25)
        if result:
            with self._prefetch_lock:
                self._prefetch_cache.pop(key, None)
            return result[0]
        return ""

    @staticmethod
    def _format_context(result: Any) -> str:
        if not isinstance(result, Mapping):
            return ""
        lines: list[str] = ["[GossipMemo context]"]
        bundle = result.get("bundle") if isinstance(result.get("bundle"), Mapping) else result
        user_model = bundle.get("user_model") if isinstance(bundle, Mapping) else None
        if user_model:
            card = user_model.get("profile_card") if isinstance(user_model, Mapping) else user_model
            if card:
                lines.append(f"User model: {_compact(card, 900)}")
        continuity = bundle.get("continuity") if isinstance(bundle, Mapping) else None
        if continuity:
            text = continuity.get("text") if isinstance(continuity, Mapping) else continuity
            if _compact(text):
                lines.append(f"Continuity: {_compact(text, 900)}")
        guidance = result.get("guidance", {}) if isinstance(result.get("guidance"), Mapping) else (
            bundle.get("guidance", {}) if isinstance(bundle, Mapping) else {})
        guidance_items = guidance.get("items", []) if isinstance(guidance, Mapping) else []
        if isinstance(guidance_items, list):
            seen_guidance: set[str] = set()
            goal_lines: list[str] = []
            for item in guidance_items[:8]:
                if not isinstance(item, Mapping):
                    continue
                item_id = str(item.get("id") or item.get("content") or "")
                content = _compact(item.get("content"), 420)
                if not content or item_id in seen_guidance:
                    continue
                seen_guidance.add(item_id)
                if item.get("kind") == "hypothesis":
                    lines.append(f"Tentative hypothesis: {content}")
                else:
                    goal_lines.append(f"Optional learning goal: {content}")
            if goal_lines:
                # Several goals arrive at once and they are a random sample of
                # open directions, not a shortlist chosen for this moment. Say
                # so, or the natural reading is a checklist to work through.
                lines.append(
                    "About the optional learning goals below: they are a random sample of "
                    "long-term directions, not questions to ask and not a checklist. Ignore "
                    "them by default. Use one only if the conversation already touches it, "
                    "and then in your own words — confirm, follow up, keep observing, or let "
                    "it go, as the moment warrants. Never walk through them in turn; that "
                    "turns the conversation into an interview.")
                lines.extend(goal_lines)
        people = bundle.get("people", []) if isinstance(bundle, Mapping) else []
        people = list(people) + list(result.get("known_people", []))
        seen_people: set[str] = set()
        if isinstance(people, list):
            for person in people:
                if not isinstance(person, Mapping):
                    continue
                name = _compact(person.get("display_name") or person.get("id"), 160)
                stable_id = str(person.get("id") or name)
                if not name or stable_id in seen_people:
                    continue
                seen_people.add(stable_id)
                profile = _compact(person.get("profile_card"), 700)
                if name:
                    lines.append(f"Person: {name}" + (f" — {profile}" if profile else ""))
                if len(seen_people) >= 12:
                    break
        memories = list(bundle.get("memories", [])) if isinstance(bundle, Mapping) else []
        memories += list(result.get("memory_recall", []))
        seen_memories: set[str] = set()
        if isinstance(memories, list):
            for memory in memories:
                if not isinstance(memory, Mapping):
                    continue
                content = _compact(memory.get("content"), 700)
                if not content or content in seen_memories:
                    continue
                seen_memories.add(content)
                basis = _compact(memory.get("basis"), 60)
                if content:
                    lines.append(f"Memory ({basis or 'unknown'}): {content}")
                if len(seen_memories) >= 20:
                    break
        relationships = result.get("relationships")
        if isinstance(relationships, list):
            for relationship in relationships[:12]:
                if not isinstance(relationship, Mapping):
                    continue
                summary = _compact(relationship.get("summary"), 700)
                if summary:
                    lines.append(f"Relationship: {summary}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            _schema(
                "gossipmemo_search",
                "Search provenance-aware memories, people, and relationships in GossipMemo.",
                {
                    "query": {"type": "string", "description": "Question or memory to search for."},
                    "people": {"type": "array", "items": {"type": "string"}},
                    "include_relationships": {"type": "boolean", "default": True},
                    "expand_relationships": {"type": "integer", "enum": [0, 1], "default": 0},
                    "include_evidence": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 30},
                },
                ("query",),
            ),
            _schema(
                "gossipmemo_store",
                "Store an explicit durable memory with person reference strings.",
                {
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": ["fact", "event", "preference", "plan", "situation", "impression"]},
                    "people": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Person reference strings mentioned by the memory.",
                    },
                    "about_user": {"type": "boolean", "description": "Whether this memory is about the current user."},
                    "valid_from": {"type": "string"},
                    "valid_to": {"type": "string"},
                },
                ("content",),
            ),
            _schema(
                "gossipmemo_dossier",
                "Read a person's or relationship's current GossipMemo dossier without query synthesis.",
                {
                    "person_id": {"type": "string"},
                    "relationship_id": {"type": "string"},
                },
            ),
            _schema(
                "gossipmemo_retract",
                "Retract a GossipMemo memory while retaining its provenance and correction history.",
                {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ("memory_id",),
            ),
            _schema(
                "gossipmemo_merge_people",
                "Merge two confirmed Person records. Only call after the user "
                "has confirmed that both records are the same person and which "
                "record should remain canonical.",
                {
                    "source_person_id": {
                        "type": "string",
                        "description": "The duplicate person record to merge away.",
                    },
                    "target_person_id": {
                        "type": "string",
                        "description": "The canonical person record to keep.",
                    },
                },
                ("source_person_id", "target_person_id"),
            ),
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        client = self._client
        if not client:
            return _json_result({"error": "GossipMemo is not initialized"})
        if not isinstance(args, Mapping):
            args = {}
        try:
            if tool_name == "gossipmemo_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return _json_result({"error": "query is required"})
                result = client.query(
                    query,
                    people=args.get("people") or [],
                    include_relationships=bool(args.get("include_relationships", True)),
                    expand_relationships=int(args.get("expand_relationships", 0)),
                    include_evidence=bool(args.get("include_evidence", True)),
                    limit=min(max(int(args.get("limit", 30)), 1), 100),
                )
                return _json_result(result)
            if tool_name == "gossipmemo_store":
                content = str(args.get("content", "")).strip()
                if not content:
                    return _json_result({"error": "content is required"})
                result = client.add_memory(
                    content,
                    kind=str(args.get("kind", "fact")),
                    people=args.get("people") or [],
                    valid_from=args.get("valid_from"),
                    valid_to=args.get("valid_to"),
                    about_user=bool(args.get("about_user", False)),
                )
                return _json_result(result)
            if tool_name == "gossipmemo_dossier":
                person_id = str(args.get("person_id", "")).strip()
                relationship_id = str(args.get("relationship_id", "")).strip()
                if person_id:
                    dossier = getattr(client, "person_dossier", None)
                    result = (
                        dossier(person_id)
                        if callable(dossier)
                        else client.query(
                            {
                                "question": "dossier",
                                "people": [person_id],
                                "include_relationships": True,
                                "expand_relationships": 1,
                                "include_evidence": True,
                                "limit": 100,
                            }
                        )
                    )
                    return _json_result(result)
                if relationship_id:
                    dossier = getattr(client, "relationship_dossier", None)
                    result = (
                        dossier(relationship_id)
                        if callable(dossier)
                        else client.query(
                            {
                                "question": "dossier",
                                "include_relationships": True,
                                "expand_relationships": 1,
                                "include_evidence": True,
                                "limit": 100,
                                "relationship_id": relationship_id,
                            }
                        )
                    )
                    return _json_result(result)
                return _json_result({"error": "person_id or relationship_id is required"})
            if tool_name == "gossipmemo_retract":
                memory_id = str(args.get("memory_id", "")).strip()
                if not memory_id:
                    return _json_result({"error": "memory_id is required"})
                return _json_result(client.retract(memory_id, reason=args.get("reason")))
            if tool_name == "gossipmemo_merge_people":
                source_person_id = str(args.get("source_person_id", "")).strip()
                target_person_id = str(args.get("target_person_id", "")).strip()
                if not source_person_id or not target_person_id:
                    return _json_result(
                        {"error": "source_person_id and target_person_id are required"}
                    )
                return _json_result(
                    client.merge_person(source_person_id, target_person_id)
                )
            return _json_result({"error": f"unknown tool: {tool_name}"})
        except GossipMemoError as exc:
            return _json_result({"error": str(exc), "status_code": exc.status_code})
        except (TypeError, ValueError) as exc:
            return _json_result({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - tool failures stay tool-local.
            logger.warning("GossipMemo tool %s failed: %s", tool_name, exc)
            return _json_result({"error": str(exc)})

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Mirror a built-in Hermes memory write without blocking the turn."""

        del target, metadata
        client = self._client
        if not self._write_enabled or action != "add" or not content or not client:
            return

        def _store() -> None:
            try:
                client.add_memory(content)
            except Exception as exc:  # noqa: BLE001 - mirror is best effort.
                logger.debug("GossipMemo built-in memory mirror failed: %s", exc)

        threading.Thread(target=_store, name="gossipmemo-memory-write", daemon=True).start()

    def shutdown(self) -> None:
        queue_value = self._queue
        writer = self._writer
        if queue_value is not None and writer is not None and writer.is_alive():
            self._stop_requested = True
            queue_value.put(None)
            writer.join(timeout=3.0)
        for thread in self._prefetch_threads:
            thread.join(timeout=1.0)
        client = self._client
        self._client = None
        self._queue = None
        self._writer = None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort.
                logger.debug("GossipMemo client close failed: %s", exc)


# Both spellings are useful to plugin authors and preserve a simple public API.
GossipMemoProvider = GossipMemoMemoryProvider


def register(ctx: Any) -> None:
    """Register this provider with Hermes' plugin context."""

    ctx.register_memory_provider(GossipMemoMemoryProvider())


__all__ = [
    "GossipMemoMemoryProvider",
    "GossipMemoProvider",
    "register",
]
