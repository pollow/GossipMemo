"""Hermes plugin integration for a GossipMemo server.

The plugin is intentionally self-contained.  Hermes is an optional runtime
dependency, so importing this module while developing GossipMemo (or running
its SDK tests) does not require Hermes to be installed.

This registers as a standalone Hermes plugin (``plugins.enabled:
[gossipmemo]``) -- tools, hooks, and (eventually) middleware, all wired
through a real ``PluginContext`` so tools land in Hermes' tool registry and
are ``tool_search``-deferrable. There is no ``memory.provider`` mode: it
existed briefly and was dropped because it had exactly one consumer (the
migration this plugin was built for) and duplicating the registration path
for zero remaining users is exactly the abstraction this repo's AGENTS.md
rules out. Do not set ``memory.provider: gossipmemo`` alongside
``plugins.enabled: [gossipmemo]`` -- see the README for why that produces
duplicate writes.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gossipmemo_client import GossipMemo, GossipMemoError

try:  # Only available inside a running Hermes process; used to resolve the
    # profile-scoped config directory the same way memory_manager does when
    # it auto-injects ``hermes_home`` ahead of ``initialize()``.
    from hermes_constants import get_hermes_home as _hermes_get_hermes_home
except ImportError:  # pragma: no cover - exercised only outside Hermes.
    _hermes_get_hermes_home = None


logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8765"
_DEFAULT_SPACE_ID = "personal"
_CONFIG_FILENAME = "gossipmemo.json"
_ENV_BASE_URL = ("GOSSIPMEMO_BASE_URL", "GOSSIPMEMO_URL")
_ENV_API_KEY = "GOSSIPMEMO_API_KEY"
_ENV_SPACE_ID = "GOSSIPMEMO_SPACE_ID"

# User-model rendering: per-item and per-category budgets, and the
# bookkeeping keys that a Mapping-shaped category carries alongside its
# actual content (never rendered as items themselves).
_USER_MODEL_ITEM_LIMIT = 240
_USER_MODEL_CATEGORY_LIMIT = 700
_USER_MODEL_BOOKKEEPING_KEYS = {"valid_from", "valid_to", "items"}


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


class GossipMemoMemoryProvider:
    """A thin, provenance-preserving bridge to the GossipMemo HTTP server.

    Despite the name (kept for API/diff stability), this is now a plain
    engine class -- it no longer subclasses any Hermes ABC. It is driven
    entirely through the plugin adapter below (``_GossipMemoPluginAdapter``
    and ``register()``).
    """

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
        # A conversation is strictly serial, so at most one turn is in flight
        # per conversation: a single slot, not a list that can only grow.
        self._current_turn: dict[str, dict[str, Any]] = {}
        self._prefetch_threads: list[threading.Thread] = []
        # Set once system_prompt_block() has actually delivered the stable
        # (user-model/hypothesis) half for a conversation, so prefetch/
        # _format_context knows it can stop repeating that half per turn. A
        # dead server or a session predating system_prompt_block() leaves
        # this unset, and the volatile-channel rendering keeps emitting the
        # stable half exactly as before -- never silently dropped.
        self._stable_delivered: dict[str, bool] = {}
        # Conversation keys plugin-mode hooks have already treated as a
        # "first turn" this session. Hermes announces the genuine first turn
        # of a session via pre_llm_call's is_first_turn kwarg, but it does
        # not re-announce a session_id rotation mid-conversation -- context
        # compression, /resume, and /branch all silently swap the session_id
        # a later hook call carries. Because every per-session dict here
        # (_context_cache, _prefetch_cache, _current_turn, _stable_delivered)
        # is keyed on that session_id, an unannounced rotation lands on a key
        # none of them have seen, and relying solely on is_first_turn would
        # mean the stable block is never fetched (and _stable_delivered never
        # set) again for the rest of that conversation. _claim_first_turn
        # treats "a key we have not seen before" as a first turn in its own
        # right, independent of Hermes' flag, so rotation is covered without
        # depending on any hook announcing it.
        self._session_started_keys: set[str] = set()

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
        self._current_turn.clear()
        self._stable_delivered.clear()
        self._session_started_keys.clear()
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
        instructions = (
            "# GossipMemo memory\n"
            "GossipMemo keeps provenance-aware memories about people and relationships. "
            "Use gossipmemo_query before relying on social context, gossipmemo_store "
            "for explicit durable facts, gossipmemo_dossier for a person's current "
            "profile, and gossipmemo_retract to correct a memory."
        )
        stable = self._fetch_stable_block()
        if stable:
            return f"{instructions}\n\n{stable}"
        return instructions

    def _claim_first_turn(self, key: str) -> bool:
        """Return ``True`` the first time ``key`` is seen this session.

        Always call this (never short-circuit around it) so a key gets
        registered on every call, whether or not the caller ends up treating
        it as a first turn. See the ``_session_started_keys`` comment in
        ``__init__`` for why an unseen key is treated as a first turn.
        """

        with self._prefetch_lock:
            if key in self._session_started_keys:
                return False
            self._session_started_keys.add(key)
            return True

    def _fetch_stable_block(self, *, session_id: str = "") -> str:
        """Best-effort fetch of the slowly-changing (user-model/hypothesis) half.

        Hermes calls ``system_prompt_block`` once per session while assembling
        the (cached, never rebuilt mid-session) system prompt, so this must
        never raise and must never block that assembly for long -- a slow or
        dead GossipMemo server should just mean no stable block this session,
        not a broken prompt. ``session_id`` lets plugin-mode callers key this
        fetch by the *current* per-call session_id (which may have rotated
        since ``initialize()``) rather than always falling back to the
        session_id captured at initialization.
        """

        key = self._conversation_key(session_id) or "default"
        with self._prefetch_lock:
            cached = self._context_cache.get(key)
        bundle: Any = cached if isinstance(cached, Mapping) and cached else None
        if bundle is None:
            client = self._client
            if client is None:
                return ""
            fetched: list[Any] = []

            def _fetch() -> None:
                try:
                    value = client.context()
                except Exception as exc:  # noqa: BLE001 - must never break prompt assembly.
                    logger.debug("GossipMemo system_prompt_block context fetch failed: %s", exc)
                    return
                if isinstance(value, Mapping):
                    fetched.append(value)

            thread = threading.Thread(
                target=_fetch, name="gossipmemo-system-prompt-context", daemon=True)
            thread.start()
            thread.join(timeout=0.5)
            if not fetched:
                return ""
            bundle = fetched[0]
            with self._prefetch_lock:
                self._context_cache.setdefault(key, dict(bundle))
        lines = self._render_stable_lines(bundle)
        if not lines:
            return ""
        self._stable_delivered[key] = True
        return "\n".join(lines)

    @staticmethod
    def _render_stable_lines(bundle: Any) -> list[str]:
        """Render just the slowly-changing half: user model + hypotheses.

        Mirrors the corresponding fragments of ``_format_context`` so the
        line shapes (``User model — ``, ``Tentative hypothesis: ``) stay
        byte-identical between the two channels.
        """

        if not isinstance(bundle, Mapping):
            return []
        lines: list[str] = []
        user_model = bundle.get("user_model")
        if user_model:
            card = user_model.get("profile_card") if isinstance(
                user_model, Mapping) else user_model
            if card:
                lines.extend(GossipMemoMemoryProvider._render_user_model(card))
        guidance = bundle.get("guidance")
        guidance_items = guidance.get("items", []) if isinstance(guidance, Mapping) else []
        if isinstance(guidance_items, list):
            seen_guidance: set[str] = set()
            for item in guidance_items[:8]:
                if not isinstance(item, Mapping) or item.get("kind") != "hypothesis":
                    continue
                item_id = str(item.get("id") or item.get("content") or "")
                content = _compact(item.get("content"), 420)
                if not content or item_id in seen_guidance:
                    continue
                seen_guidance.add(item_id)
                lines.append(f"Tentative hypothesis: {content}")
        return lines

    def _context_for_volatile_render(
        self, merged: Mapping[str, Any], key: str
    ) -> Mapping[str, Any]:
        """Strip the stable half out of a turn result before static rendering.

        Only takes effect once ``system_prompt_block`` has actually delivered
        the stable block for this conversation (tracked in
        ``self._stable_delivered``); otherwise the input passes through
        unchanged and ``_format_context`` keeps emitting everything, exactly
        as it did before this split existed.
        """

        if not self._stable_delivered.get(key):
            return merged

        def _without_hypotheses(guidance: Any) -> Any:
            if not isinstance(guidance, Mapping):
                return guidance
            items = guidance.get("items")
            if not isinstance(items, list):
                return guidance
            filtered = dict(guidance)
            filtered["items"] = [
                item for item in items
                if not (isinstance(item, Mapping) and item.get("kind") == "hypothesis")
            ]
            return filtered

        result = dict(merged)
        bundle = result.get("bundle")
        if isinstance(bundle, Mapping):
            bundle = dict(bundle)
            bundle.pop("user_model", None)
            if "guidance" in bundle:
                bundle["guidance"] = _without_hypotheses(bundle["guidance"])
            result["bundle"] = bundle
        if "guidance" in result:
            result["guidance"] = _without_hypotheses(result["guidance"])
        return result

    def _conversation_key(self, session_id: str = "") -> str | None:
        value = session_id or self._session_id
        return value or None

    def _turn_messages(
        self,
        user_content: str,
        assistant_content: str,
        session_id: str = "",
        *,
        include_user: bool = True,
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
        messages: list[dict[str, Any]] = []
        if include_user:
            messages.append(
                {
                    "author": "user",
                    "content": user_content,
                    "occurred_at": now,
                    "source": user_source,
                }
            )
        messages.append(
            {
                "author": "assistant",
                "content": assistant_content,
                "occurred_at": now,
                "source": assistant_source,
            }
        )
        return messages

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
                    client.turn(batch)
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
            # The turn is over either way, so the slot is always released here.
            entry = self._current_turn.pop(conversation, None)
            if entry is not None and entry.get("content") == user_content:
                idem = str(entry.get("idempotency_key", ""))
                persisted = bool(entry.get("persisted"))
            else:
                # No prefetch ran for this content (or Hermes called us
                # directly): assume nothing was written yet.
                idem = ""
                persisted = False
        if persisted:
            # The prefetch turn already stored the user message, so resending it
            # would only rely on the server to deduplicate work we can skip.
            self._queue.put(
                self._turn_messages(
                    user_content, assistant_content, session_id, include_user=False
                )
            )
            return
        messages = self._turn_messages(user_content, assistant_content, session_id)
        # The idempotency key stays a fuse rather than a routine deduplicator:
        # the prefetch write may have committed on the server even though its
        # response never reached us (a timeout), and only this stable key keeps
        # that ambiguous case from creating a duplicate user row.
        messages[0]["idempotency_key"] = idem or uuid.uuid4().hex
        self._queue.put(messages)

    def _turn_context(self, query: str, key: str) -> tuple[str, str]:
        client = self._client
        if client is None:
            return "", ""
        with self._prefetch_lock:
            cached = dict(self._context_cache.get(key, {}))
            entry = self._current_turn.get(key)
            if entry is None or entry.get("content") != query:
                # A different content means the previous turn is over and its
                # entry can never be claimed again, so the slot is overwritten.
                entry = {
                    "content": query,
                    "idempotency_key": uuid.uuid4().hex,
                    "started": False,
                    "persisted": False,
                    "event": threading.Event(),
                }
                self._current_turn[key] = entry
            idem = entry["idempotency_key"]
            event = entry["event"]
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
            # Only a well-formed response with message IDs proves the user
            # message reached storage; every other path (exception, timeout,
            # unexpected body) leaves ``persisted`` false and makes sync_turn
            # resend the user message.
            message_ids = result.get("message_ids")
            entry["persisted"] = bool(
                isinstance(message_ids, Sequence)
                and not isinstance(message_ids, (str, bytes))
                and len(message_ids) > 0
            )
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
        formatted = self._format_context(self._context_for_volatile_render(merged, key))
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
    def _render_user_model(card: Any) -> list[str]:
        """Render a UserModel profile card as one line per category.

        ``card`` is LLM-produced free-form JSON, nominally shaped as
        ``{category: {"valid_from": ..., "valid_to": ..., "items": [...]}}``.
        This renders readable text (never a raw dict repr, never
        ``valid_to``), truncating on item boundaries with an explicit
        ``(+N more)`` marker rather than cutting mid-item.
        """

        if not isinstance(card, Mapping):
            compacted = _compact(card, 900)
            return [f"User model: {compacted}"] if compacted else []
        lines: list[str] = []
        for key, value in card.items():
            label = str(key).replace("_", " ")
            valid_from: Any = None
            if isinstance(value, Mapping):
                valid_from = value.get("valid_from")
                raw_items = value.get("items")
                if isinstance(raw_items, list):
                    items: list[Any] = raw_items
                elif raw_items is not None:
                    items = [raw_items]
                else:
                    items = [
                        item_value
                        for item_key, item_value in value.items()
                        if item_key not in _USER_MODEL_BOOKKEEPING_KEYS and item_value is not None
                    ]
            elif isinstance(value, list):
                items = value
            elif value is not None:
                items = [value]
            else:
                items = []
            rendered_items = [
                text for item in items if (text := _compact(item, _USER_MODEL_ITEM_LIMIT))
            ]
            if not rendered_items:
                continue
            kept: list[str] = []
            used = 0
            for text in rendered_items:
                added_cost = len(text) + (2 if kept else 0)  # "; " separator
                if kept and used + added_cost > _USER_MODEL_CATEGORY_LIMIT:
                    break
                kept.append(text)
                used += added_cost
            dropped = len(rendered_items) - len(kept)
            body = "; ".join(kept)
            if dropped:
                body += f" (+{dropped} more)"
            prefix = f"User model — {label}"
            if valid_from:
                prefix += f" (since {_compact(valid_from, 40)})"
            lines.append(f"{prefix}: {body}")
        return lines

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
                lines.extend(GossipMemoMemoryProvider._render_user_model(card))
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
                "gossipmemo_query",
                "Ask a question and get a synthesized, provenance-aware answer over memories, "
                "people, and relationships in GossipMemo. This runs an LLM synthesis call, so "
                "it is slower than a lookup; reserve it for genuine questions, not routine checks. "
                "For a plain keyword lookup ('what do I have on file about X'), prefer the cheaper "
                "gossipmemo_recall instead.",
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
                "gossipmemo_recall",
                "Cheap, LLM-free keyword search over stored GossipMemo memories. This is the "
                "default lookup for 'what do I have on file matching these keywords' -- it does "
                "direct FTS matching with no synthesis, so prefer it over gossipmemo_query for "
                "plain lookups; reach for gossipmemo_query only when you need a synthesized "
                "answer.",
                {
                    "q": {"type": "string", "description": "Keyword text to search for."},
                    "about_user": {
                        "type": "boolean",
                        "description": "Restrict to (or exclude) memories about the current user.",
                    },
                    "person_ids": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                ("q",),
            ),
            _schema(
                "gossipmemo_store",
                "Store an explicit durable memory with person reference strings.",
                {
                    "content": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "fact", "event", "preference", "plan", "situation", "impression",
                        ],
                    },
                    "people": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Person reference strings mentioned by the memory.",
                    },
                    "about_user": {
                        "type": "boolean",
                        "description": "Whether this memory is about the current user.",
                    },
                    "valid_from": {"type": "string"},
                    "valid_to": {"type": "string"},
                },
                ("content",),
            ),
            _schema(
                "gossipmemo_dossier",
                "Read a person's or relationship's current GossipMemo dossier without query "
                "synthesis.",
                {
                    "person_id": {"type": "string"},
                    "relationship_id": {"type": "string"},
                },
            ),
            _schema(
                "gossipmemo_retract",
                "Retract a GossipMemo memory while retaining its provenance and correction "
                "history.",
                {
                    "memory_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ("memory_id",),
            ),
            _schema(
                "gossipmemo_people",
                "List or search the known Person records in this GossipMemo space "
                "(id, display name, and aliases for each). This is the way to find "
                "candidate duplicate records -- use it before gossipmemo_merge_people "
                "to see whether two similar people already exist.",
                {
                    "q": {
                        "type": "string",
                        "description": "Optional text to match against display names and "
                                       "aliases. Omit to list all people.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                },
            ),
            _schema(
                "gossipmemo_guidance",
                "Retrieve the open learning goals and working hypotheses GossipMemo currently "
                "holds for the user, or for given people -- the things it still wants to find "
                "out or is tentatively inferring. This is the full list for an explicit ask, not "
                "the small sample that already rides along in context/turn prefetch. Treat "
                "hypotheses as tentative: never state one to the user as settled fact. Treat "
                "learning goals as optional directions, not instructions to follow or a "
                "checklist to work through -- use one only if the conversation already touches "
                "it, in your own words.",
                {
                    "person_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Optional person IDs to focus on. Omit for the "
                                       "user/global scope.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["hypothesis", "learning_goal"],
                        "description": "Restrict to just hypotheses or just learning goals. "
                                       "Omit for both.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                },
            ),
            _schema(
                "gossipmemo_merge_people",
                "Merge two confirmed Person records. Use gossipmemo_people first to "
                "discover candidate duplicates. Only call this after the user "
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
            if tool_name == "gossipmemo_query":
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
            if tool_name == "gossipmemo_recall":
                q = str(args.get("q", "")).strip()
                if not q:
                    return _json_result({"error": "q is required"})
                about_user = args.get("about_user")
                result = client.recall(
                    q,
                    about_user=bool(about_user) if about_user is not None else None,
                    person_ids=args.get("person_ids") or [],
                    limit=min(max(int(args.get("limit", 20)), 1), 100),
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
            if tool_name == "gossipmemo_people":
                q = str(args.get("q", "")).strip()
                limit = min(max(int(args.get("limit", 50)), 1), 200)
                result = client.list_people(q, limit=limit)
                return _json_result(result)
            if tool_name == "gossipmemo_guidance":
                kind = args.get("kind")
                kind = str(kind).strip() if kind else None
                if kind and kind not in ("hypothesis", "learning_goal"):
                    return _json_result(
                        {"error": "kind must be 'hypothesis' or 'learning_goal'"})
                result = client.guidance(
                    person_ids=args.get("person_ids") or [],
                    kind=kind,
                    limit=min(max(int(args.get("limit", 50)), 1), 200),
                )
                return _json_result(result)
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

_TOOLSET = "gossipmemo"
_CRON_SESSION_PREFIX = "cron_"


def _is_primary_session(session_id: str) -> bool:
    """Guard against ingesting non-primary-user sessions as the user's memories.

    Hermes' ``MemoryProvider`` ABC (used by the now-removed memory-provider
    path) received an explicit ``agent_context`` kwarg for this; plugin hooks
    carry no such kwarg. Hermes names cron sessions
    ``cron_<job_id>_<timestamp>``, which is the only reliable textual signal
    plugin hooks carry for "this is not the primary interactive user" -- so
    that is what this guard is built from. It is not a full equivalent (e.g.
    it has no signal for subagent sessions), but it preserves the intent of
    not ingesting cron turns as the user's memories.
    """

    return not str(session_id or "").startswith(_CRON_SESSION_PREFIX)


class _GossipMemoPluginAdapter:
    """Adapts :class:`GossipMemoMemoryProvider` onto a real Hermes plugin context.

    Owns only the wiring between Hermes' hook kwargs and the provider's
    existing methods; all behavior (client lifecycle, prefetch caching,
    idempotency, formatting) stays in ``GossipMemoMemoryProvider`` unchanged.
    """

    def __init__(self, provider: GossipMemoMemoryProvider) -> None:
        self._provider = provider

    def on_session_start(self, **kwargs: Any) -> None:
        session_id = str(kwargs.get("session_id") or "")
        init_kwargs: dict[str, Any] = {}
        if _hermes_get_hermes_home is not None:
            try:
                init_kwargs["hermes_home"] = str(_hermes_get_hermes_home())
            except Exception as exc:  # noqa: BLE001 - initialize must not raise.
                logger.debug("GossipMemo could not resolve hermes_home: %s", exc)
        self._provider.initialize(session_id, **init_kwargs)
        # initialize() defaults _write_enabled to True (no agent_context kwarg
        # is available here); override with the reconstructed guard.
        self._provider._write_enabled = _is_primary_session(session_id)

    def pre_llm_call(self, **kwargs: Any) -> dict[str, Any]:
        session_id = str(kwargs.get("session_id") or "")
        user_message = str(kwargs.get("user_message") or "")
        parts: list[str] = []
        # Always claim the key, even when is_first_turn already makes the
        # left side of the `or` True below -- short-circuiting around the
        # claim would leave this key unregistered, and the very next turn
        # (same key, is_first_turn now False) would then read as unseen and
        # re-fetch the stable block again.
        key = self._provider._conversation_key(session_id) or "default"
        is_new_key = self._provider._claim_first_turn(key)
        if kwargs.get("is_first_turn") or is_new_key:
            # system_prompt_block() has no hook equivalent in plugin mode
            # (that lands in slice 3b, via middleware). Prepend just the
            # stable (user-model/hypothesis) half here so it is not silently
            # lost; this also flips _stable_delivered so the volatile half
            # below stops repeating it, exactly as it would in provider mode.
            # Treating an unseen key as a first turn (not just Hermes'
            # is_first_turn flag) is what keeps this working across a
            # mid-conversation session_id rotation -- context compression,
            # /resume, and /branch all swap session_id without announcing it
            # through any hook, so is_first_turn alone would never be True
            # again for the rest of that conversation.
            stable = self._provider._fetch_stable_block(session_id=session_id)
            if stable:
                parts.append(stable)
        volatile = self._provider.prefetch(user_message, session_id=session_id)
        if volatile:
            parts.append(volatile)
        return {"context": "\n\n".join(parts)}

    def post_llm_call(self, **kwargs: Any) -> None:
        session_id = str(kwargs.get("session_id") or "")
        user_message = kwargs.get("user_message")
        assistant_response = kwargs.get("assistant_response")
        if not user_message or not assistant_response:
            return
        self._provider.sync_turn(
            str(user_message), str(assistant_response), session_id=session_id
        )

    def on_session_finalize(self, **kwargs: Any) -> None:
        del kwargs
        self._provider.shutdown()


def _make_tool_handler(
    provider: GossipMemoMemoryProvider, tool_name: str
) -> Callable[..., str]:
    def _handler(args: dict[str, Any], **kwargs: Any) -> str:
        return provider.handle_tool_call(tool_name, args, **kwargs)

    return _handler


def _register_as_plugin(ctx: Any, provider: GossipMemoMemoryProvider) -> None:
    """Wire ``provider`` onto a real Hermes ``PluginContext``.

    A standalone helper (rather than inlined into ``register()``) so tests
    can drive the wiring against a fake context and a specific provider
    instance without going through module-level plugin construction.
    """

    adapter = _GossipMemoPluginAdapter(provider)
    for schema in provider.get_tool_schemas():
        ctx.register_tool(
            name=schema["name"],
            toolset=_TOOLSET,
            schema=schema,
            handler=_make_tool_handler(provider, schema["name"]),
        )
    ctx.register_hook("on_session_start", adapter.on_session_start)
    ctx.register_hook("pre_llm_call", adapter.pre_llm_call)
    ctx.register_hook("post_llm_call", adapter.post_llm_call)
    ctx.register_hook("on_session_finalize", adapter.on_session_finalize)


def register(ctx: Any) -> None:
    """Register this plugin's tools and hooks with Hermes' plugin context."""

    _register_as_plugin(ctx, GossipMemoMemoryProvider())


__all__ = [
    "GossipMemoMemoryProvider",
    "GossipMemoProvider",
    "register",
]
