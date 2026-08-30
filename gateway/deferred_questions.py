"""Durable questions that plugins deliver after a gateway session is idle."""

from __future__ import annotations

import json
import asyncio
import contextvars
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


QuestionState = Literal["queued", "delivering", "awaiting", "handling"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeferredQuestion:
    id: str
    plugin_id: str
    session_key: str
    delivery_source: dict[str, object]
    question: str
    handler_name: str
    context: dict[str, object]
    dedupe_key: str
    state: QuestionState
    response: str | None
    result: DeferredQuestionResult | None
    delivery_attempted: bool
    created_at: float
    updated_at: float

    @property
    def platform(self) -> str:
        return str(
            self.delivery_source.get("delivery_transport")
            or self.delivery_source["platform"]
        )

    @property
    def adapter_profile(self) -> str:
        return str(self.delivery_source.get("adapter_profile") or "default")


@dataclass(frozen=True)
class DeferredQuestionResult:
    resolved: bool
    reply: str
    question: str | None = None

    @classmethod
    def done(cls, reply: str) -> "DeferredQuestionResult":
        return cls(resolved=True, reply=reply)

    @classmethod
    def clarify(cls, question: str) -> "DeferredQuestionResult":
        return cls(resolved=False, reply="", question=question)


DeferredQuestionHandler = Callable[
    [DeferredQuestion, str], Awaitable[DeferredQuestionResult]
]


class DeferredQuestionService:
    """Persist deferred questions and dispatch captured replies to plugins."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._handlers: dict[
            tuple[str, str], tuple[DeferredQuestionHandler, contextvars.Context]
        ] = {}
        self._adapters: dict[
            tuple[str, str], tuple[Any, asyncio.AbstractEventLoop]
        ] = {}
        self._ready_adapters: set[tuple[str, str]] = set()
        self._busy_callbacks: set[str] = set()
        self._handler_lock = asyncio.Lock()
        self._handling_retry_tasks: dict[str, asyncio.Task[None]] = {}
        self._recovery_task: asyncio.Task[None] | None = None
        self._recovery_requested = False
        self.handling_retry_seconds = 5.0
        self._initialize()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back each operation and always close its connection."""
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            from hermes_state import apply_durability_barriers

            apply_durability_barriers(conn)
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._lock, self._transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS deferred_questions (
                    id TEXT PRIMARY KEY,
                    plugin_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    question TEXT NOT NULL,
                    handler_name TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('queued', 'delivering', 'awaiting', 'handling')
                    ),
                    response TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    delivery_source_json TEXT NOT NULL,
                    result_json TEXT,
                    delivery_attempted INTEGER NOT NULL DEFAULT 0
                );
                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_deferred_questions_dedupe
                ON deferred_questions(plugin_id, dedupe_key);
                CREATE INDEX IF NOT EXISTS
                    ix_deferred_questions_session_state
                ON deferred_questions(session_key, state, created_at);
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> DeferredQuestion | None:
        if row is None:
            return None
        context = json.loads(row["context_json"])
        if not isinstance(context, dict):
            raise ValueError("deferred question context must be a JSON object")
        delivery_source = json.loads(row["delivery_source_json"])
        if not isinstance(delivery_source, dict):
            raise ValueError("deferred question delivery source must be a JSON object")
        result = None
        if row["result_json"] is not None:
            result_data = json.loads(row["result_json"])
            if not isinstance(result_data, dict):
                raise ValueError("deferred question result must be a JSON object")
            result = DeferredQuestionResult(
                resolved=bool(result_data["resolved"]),
                reply=str(result_data["reply"]),
                question=(
                    str(result_data["question"])
                    if result_data.get("question") is not None
                    else None
                ),
            )
        return DeferredQuestion(
            id=row["id"],
            plugin_id=row["plugin_id"],
            session_key=row["session_key"],
            delivery_source=delivery_source,
            question=row["question"],
            handler_name=row["handler_name"],
            context=context,
            dedupe_key=row["dedupe_key"],
            state=row["state"],
            response=row["response"],
            result=result,
            delivery_attempted=bool(row["delivery_attempted"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, question_id: str) -> DeferredQuestion:
        with self._lock, self._transaction() as conn:
            record = self._from_row(
                conn.execute(
                    "SELECT * FROM deferred_questions WHERE id = ?", (question_id,)
                ).fetchone()
            )
        if record is None:
            raise KeyError(question_id)
        return record

    def enqueue(
        self,
        *,
        plugin_id: str,
        session_key: str,
        delivery_source: dict[str, object],
        question: str,
        handler_name: str,
        context: dict[str, object],
        dedupe_key: str,
    ) -> DeferredQuestion:
        context_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
        delivery_source_json = json.dumps(
            delivery_source, sort_keys=True, separators=(",", ":")
        )
        chat_id = str(delivery_source.get("chat_id") or "")
        platform = str(
            delivery_source.get("delivery_transport")
            or delivery_source.get("platform")
            or ""
        )
        if not all(
            value.strip()
            for value in (
                plugin_id,
                platform,
                session_key,
                chat_id,
                question,
                handler_name,
                dedupe_key,
            )
        ):
            raise ValueError("deferred question fields must not be blank")
        now = time.time()
        question_id = uuid.uuid4().hex
        with self._lock, self._transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO deferred_questions (
                        id, plugin_id, platform, session_key, question,
                        handler_name, context_json, dedupe_key, state, response,
                        created_at, updated_at, delivery_source_json, result_json,
                        delivery_attempted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', NULL, ?, ?, ?, NULL, 0)
                    """,
                    (
                        question_id,
                        plugin_id,
                        platform,
                        session_key,
                        question,
                        handler_name,
                        context_json,
                        dedupe_key,
                        now,
                        now,
                        delivery_source_json,
                    ),
                )
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """
                    SELECT * FROM deferred_questions
                    WHERE plugin_id = ? AND dedupe_key = ?
                    """,
                    (plugin_id, dedupe_key),
                ).fetchone()
                existing = self._from_row(row)
                if existing is None:
                    raise
                record = existing
            else:
                record = self._from_row(
                    conn.execute(
                        "SELECT * FROM deferred_questions WHERE id = ?",
                        (question_id,),
                    ).fetchone()
                )
                if record is None:
                    raise RuntimeError("inserted deferred question disappeared")
        self._wake_record(record)
        return record

    def pending_for_session(
        self,
        session_key: str,
        *,
        adapter_profile: str | None = None,
    ) -> DeferredQuestion | None:
        sql = """
            SELECT * FROM deferred_questions
            WHERE session_key = ?
        """
        params: tuple[object, ...] = (session_key,)
        if adapter_profile is not None:
            normalized_profile = adapter_profile.strip() or "default"
            sql += """
                AND COALESCE(
                    json_extract(delivery_source_json, '$.adapter_profile'),
                    'default'
                ) = ?
            """
            params += (normalized_profile,)
        sql += " ORDER BY created_at ASC, id ASC LIMIT 1"
        with self._lock, self._transaction() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._from_row(row)

    def park_awaiting(self, question_id: str) -> DeferredQuestion | None:
        """Return an unanswered prompt to the queue while its handler is absent."""
        with self._lock, self._transaction() as conn:
            changed = conn.execute(
                """
                UPDATE deferred_questions
                SET state = 'queued', response = NULL, delivery_attempted = 0,
                    updated_at = ?
                WHERE id = ? AND state = 'awaiting'
                """,
                (time.time(), question_id),
            ).rowcount
        record = self.get(question_id) if changed else None
        if record is not None:
            self._wake_record(record)
        return record

    def claim_for_delivery(self, question_id: str) -> DeferredQuestion | None:
        now = time.time()
        with self._lock, self._transaction() as conn:
            changed = conn.execute(
                """
                UPDATE deferred_questions
                SET state = 'delivering', updated_at = ?
                WHERE id = ? AND state = 'queued'
                """,
                (now, question_id),
            ).rowcount
        return self.get(question_id) if changed else None

    def mark_awaiting(self, question_id: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE deferred_questions
                SET state = 'awaiting', delivery_attempted = 0, updated_at = ?
                WHERE id = ? AND state = 'delivering'
                """,
                (time.time(), question_id),
            )

    def requeue(self, question_id: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE deferred_questions
                SET state = 'queued', response = NULL, delivery_attempted = 0,
                    updated_at = ?
                WHERE id = ? AND state = 'delivering'
                """,
                (time.time(), question_id),
            )

    def register_handler(
        self,
        plugin_id: str,
        handler_name: str,
        handler: DeferredQuestionHandler,
    ) -> None:
        if not callable(handler):
            raise TypeError("deferred question handler must be callable")
        self._handlers[(plugin_id, handler_name)] = (
            handler,
            contextvars.copy_context(),
        )
        for key in tuple(self._ready_adapters):
            self._wake_binding(key)
        self._schedule_recovery()

    def unregister_handler(
        self,
        plugin_id: str,
        handler_name: str,
        handler: DeferredQuestionHandler,
    ) -> None:
        key = (plugin_id, handler_name)
        binding = self._handlers.get(key)
        if binding is not None and binding[0] is handler:
            self._handlers.pop(key, None)

    def has_handler(self, plugin_id: str, handler_name: str) -> bool:
        return (plugin_id, handler_name) in self._handlers

    @staticmethod
    def _adapter_key(platform: str, adapter_profile: str | None) -> tuple[str, str]:
        return (platform, (adapter_profile or "default").strip() or "default")

    def bind_adapter(
        self,
        platform: str,
        adapter: Any,
        *,
        adapter_profile: str | None = None,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        key = self._adapter_key(platform, adapter_profile)
        self._adapters[key] = (adapter, loop)
        self._ready_adapters.discard(key)

    def adapter_connected(
        self,
        platform: str,
        adapter: Any,
        *,
        adapter_profile: str | None = None,
    ) -> None:
        """Wake durable work only after this exact transport is usable."""
        key = self._adapter_key(platform, adapter_profile)
        binding = self._adapters.get(key)
        if binding is None or binding[0] is not adapter:
            return
        if key in self._ready_adapters:
            return
        _adapter, loop = binding
        self._ready_adapters.add(key)
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE deferred_questions SET state = 'queued', updated_at = ?
                WHERE platform = ? AND state = 'delivering'
                  AND delivery_attempted = 0
                  AND COALESCE(
                      json_extract(delivery_source_json, '$.adapter_profile'),
                      'default'
                  ) = ?
                """,
                (time.time(), platform, key[1]),
            )
        self._wake_binding(key)
        self._schedule_recovery(loop)

    def adapter_disconnected(
        self,
        platform: str,
        adapter: Any,
        *,
        adapter_profile: str | None = None,
    ) -> None:
        key = self._adapter_key(platform, adapter_profile)
        binding = self._adapters.get(key)
        if binding is not None and binding[0] is adapter:
            self._ready_adapters.discard(key)

    def _ready_adapter(
        self, platform: str, adapter_profile: str | None = None
    ) -> Any | None:
        key = self._adapter_key(platform, adapter_profile)
        binding = self._adapters.get(key)
        if binding is None or key not in self._ready_adapters:
            return None
        return binding[0]

    def _schedule_recovery(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if loop is None:
            loop = next(
                (
                    self._adapters[key][1]
                    for key in self._ready_adapters
                    if key in self._adapters
                ),
                None,
            )
        if loop is None or not loop.is_running():
            return

        def recover_captured() -> None:
            self._recovery_requested = True
            if self._recovery_task is None or self._recovery_task.done():
                self._recovery_task = asyncio.create_task(self._recover_handling())

        loop.call_soon_threadsafe(recover_captured, context=contextvars.Context())

    def _wake_record(self, record: DeferredQuestion) -> None:
        self._wake_binding(self._adapter_key(record.platform, record.adapter_profile))

    def _wake_binding(self, key: tuple[str, str]) -> None:
        binding = self._adapters.get(key)
        if binding is None or key not in self._ready_adapters:
            return
        _adapter, loop = binding
        platform, adapter_profile = key

        def schedule() -> None:
            asyncio.create_task(
                self.deliver_ready(platform, adapter_profile=adapter_profile)
            )

        if loop.is_running():
            loop.call_soon_threadsafe(schedule, context=contextvars.Context())

    def _mark_delivery_attempted(self, question_id: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                """
                UPDATE deferred_questions SET delivery_attempted = 1, updated_at = ?
                WHERE id = ? AND state IN ('delivering', 'handling')
                """,
                (time.time(), question_id),
            )

    def _queued(
        self, platform: str, session_key: str | None = None
    ) -> list[DeferredQuestion]:
        sql = """
            SELECT candidate.* FROM deferred_questions AS candidate
            WHERE candidate.platform = ? AND candidate.state = 'queued'
              AND NOT EXISTS (
                  SELECT 1 FROM deferred_questions AS older
                  WHERE older.session_key = candidate.session_key
                    AND (
                        older.created_at < candidate.created_at
                        OR (
                            older.created_at = candidate.created_at
                            AND older.id < candidate.id
                        )
                    )
              )
        """
        params: tuple[object, ...] = (platform,)
        if session_key is not None:
            sql += " AND candidate.session_key = ?"
            params += (session_key,)
        sql += " ORDER BY candidate.created_at ASC"
        with self._lock, self._transaction() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [record for row in rows if (record := self._from_row(row)) is not None]

    async def deliver_ready(
        self,
        platform: str,
        session_key: str | None = None,
        *,
        adapter_profile: str | None = None,
    ) -> None:
        for record in self._queued(platform, session_key):
            if (
                adapter_profile is not None
                and record.adapter_profile
                != self._adapter_key(platform, adapter_profile)[1]
            ):
                continue
            if not self.has_handler(record.plugin_id, record.handler_name):
                continue
            adapter = self._ready_adapter(record.platform, record.adapter_profile)
            if adapter is None:
                continue
            if adapter.is_session_active(record.session_key):
                if record.id in self._busy_callbacks:
                    continue
                self._busy_callbacks.add(record.id)

                async def after_delivery(
                    *,
                    question_id: str = record.id,
                    key: str = record.session_key,
                    profile: str = record.adapter_profile,
                ) -> None:
                    self._busy_callbacks.discard(question_id)
                    await self.deliver_ready(platform, key, adapter_profile=profile)

                adapter.register_session_idle_callback(
                    record.session_key, after_delivery
                )
                continue
            claimed = self.claim_for_delivery(record.id)
            if claimed is None:
                continue
            adapter = self._ready_adapter(claimed.platform, claimed.adapter_profile)
            if adapter is None:
                self.requeue(claimed.id)
                continue
            self._mark_delivery_attempted(claimed.id)
            result = await adapter.deliver_deferred_message(
                claimed.delivery_source, claimed.question
            )
            if not getattr(result, "success", False):
                if getattr(result, "retryable", False):
                    self.requeue(claimed.id)
            else:
                self.mark_awaiting(claimed.id)

    async def handle_response(
        self, session_key: str, response: str
    ) -> DeferredQuestionResult | None:
        with self._lock, self._transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM deferred_questions
                WHERE session_key = ? AND state = 'awaiting'
                ORDER BY created_at ASC LIMIT 1
                """,
                (session_key,),
            ).fetchone()
            record = self._from_row(row)
            if record is None:
                return None
            if not self.has_handler(record.plugin_id, record.handler_name):
                return None
            changed = conn.execute(
                """
                UPDATE deferred_questions
                SET state = 'handling', response = ?, updated_at = ?
                WHERE id = ? AND state = 'awaiting'
                """,
                (response, time.time(), record.id),
            ).rowcount
            if not changed:
                return None
        return await self._run_handler(self.get(record.id))

    def _schedule_handling_retry(self, record: DeferredQuestion) -> None:
        if record.id in self._handling_retry_tasks:
            return
        key = self._adapter_key(record.platform, record.adapter_profile)
        binding = self._adapters.get(key)
        if binding is None or self._ready_adapter(*key) is not binding[0]:
            return
        _adapter, loop = binding

        async def retry_once() -> None:
            try:
                await asyncio.sleep(self.handling_retry_seconds)
                if self._ready_adapter(*key) is not _adapter:
                    return
                pending = self.get(record.id)
                await self._run_handler(pending, schedule_retry=False)
            except KeyError:
                return
            except Exception:
                logger.error(
                    "Deferred-question acknowledgement retry failed for %s",
                    record.id,
                    exc_info=True,
                )
            finally:
                self._handling_retry_tasks.pop(record.id, None)

        def schedule() -> None:
            if record.id in self._handling_retry_tasks:
                return
            self._handling_retry_tasks[record.id] = asyncio.create_task(retry_once())

        if loop.is_running():
            loop.call_soon_threadsafe(schedule, context=contextvars.Context())

    async def _run_handler(
        self, record: DeferredQuestion, *, schedule_retry: bool = True
    ) -> DeferredQuestionResult | None:
        async with self._handler_lock:
            try:
                pending = self.get(record.id)
            except KeyError:
                return None
            if pending.state != "handling" or (
                pending.result is not None and pending.delivery_attempted
            ):
                return None
            return await self._run_handler_once(pending, schedule_retry=schedule_retry)

    async def _run_handler_once(
        self, record: DeferredQuestion, *, schedule_retry: bool
    ) -> DeferredQuestionResult:
        handler_binding = self._handlers.get((record.plugin_id, record.handler_name))
        if handler_binding is None:
            raise LookupError(
                f"no deferred question handler registered for "
                f"{record.plugin_id}.{record.handler_name}"
            )
        if record.response is None:
            raise ValueError("handling question has no captured response")
        result = record.result
        if result is None:
            handler, handler_context = handler_binding
            result = await asyncio.create_task(
                handler(record, record.response),
                context=handler_context.copy(),
            )
            if not isinstance(result, DeferredQuestionResult):
                raise TypeError("deferred question handler returned an invalid result")
            result_json = json.dumps(
                {
                    "resolved": result.resolved,
                    "reply": result.reply,
                    "question": result.question,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            with self._lock, self._transaction() as conn:
                conn.execute(
                    """
                    UPDATE deferred_questions
                    SET result_json = ?, delivery_attempted = 0, updated_at = ?
                    WHERE id = ? AND state = 'handling' AND result_json IS NULL
                    """,
                    (result_json, time.time(), record.id),
                )
        reply = result.reply if result.resolved else result.question
        if not result.resolved and (not reply or not reply.strip()):
            raise ValueError("clarification result requires a question")
        if reply:
            adapter = self._ready_adapter(record.platform, record.adapter_profile)
            if adapter is None:
                raise RuntimeError(f"adapter for {record.platform} is not connected")
            self._mark_delivery_attempted(record.id)
            delivered = await adapter.deliver_deferred_message(
                record.delivery_source, reply
            )
            if not getattr(delivered, "success", False):
                if getattr(delivered, "retryable", False):
                    with self._lock, self._transaction() as conn:
                        conn.execute(
                            """
                            UPDATE deferred_questions
                            SET delivery_attempted = 0, updated_at = ?
                            WHERE id = ? AND state = 'handling'
                            """,
                            (time.time(), record.id),
                        )
                    if schedule_retry:
                        self._schedule_handling_retry(record)
                elif result.resolved:
                    with self._lock, self._transaction() as conn:
                        conn.execute(
                            """
                            DELETE FROM deferred_questions
                            WHERE id = ? AND state = 'handling'
                            """,
                            (record.id,),
                        )
                    await self.deliver_ready(record.platform, record.session_key)
                else:
                    with self._lock, self._transaction() as conn:
                        conn.execute(
                            """
                            UPDATE deferred_questions
                            SET state = 'awaiting', question = ?, response = NULL,
                                result_json = NULL, delivery_attempted = 0,
                                updated_at = ?
                            WHERE id = ? AND state = 'handling'
                            """,
                            (result.question, time.time(), record.id),
                        )
                raise RuntimeError(
                    getattr(delivered, "error", None)
                    or "deferred-question reply delivery failed"
                )
        now = time.time()
        with self._lock, self._transaction() as conn:
            if result.resolved:
                conn.execute(
                    """
                    DELETE FROM deferred_questions
                    WHERE id = ? AND state = 'handling'
                    """,
                    (record.id,),
                )
            else:
                conn.execute(
                    """
                    UPDATE deferred_questions
                    SET state = 'awaiting', question = ?, response = NULL,
                        result_json = NULL, delivery_attempted = 0, updated_at = ?
                    WHERE id = ? AND state = 'handling'
                    """,
                    (result.question, now, record.id),
                )
        if result.resolved:
            await self.deliver_ready(record.platform, record.session_key)
        return result

    async def _recover_handling(self) -> None:
        while True:
            self._recovery_requested = False
            await self.retry_handling()
            if not self._recovery_requested:
                return

    async def retry_handling(
        self,
    ) -> list[tuple[str, DeferredQuestionResult]]:
        with self._lock, self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM deferred_questions
                WHERE state = 'handling'
                  AND (result_json IS NULL OR delivery_attempted = 0)
                ORDER BY created_at ASC
                """
            ).fetchall()
        results = []
        for row in rows:
            record = self._from_row(row)
            if record is None:
                continue
            if self._ready_adapter(record.platform, record.adapter_profile) is None:
                continue
            if (record.plugin_id, record.handler_name) not in self._handlers:
                continue
            try:
                result = await self._run_handler(record, schedule_retry=False)
            except Exception:
                logger.error(
                    "Deferred-question recovery failed for %s",
                    record.id,
                    exc_info=True,
                )
                continue
            if result is not None:
                results.append((record.id, result))
        return results


class DeferredQuestionClient:
    """Plugin-scoped facade over the host deferred-question service."""

    def __init__(
        self,
        service: DeferredQuestionService,
        plugin_id: str,
        track_registration: Callable[[str, Callable[[], None]], object] | None = None,
    ) -> None:
        self._service = service
        self._plugin_id = plugin_id
        self._track_registration = track_registration

    def register_handler(
        self, handler_name: str, handler: DeferredQuestionHandler
    ) -> None:
        self._service.register_handler(self._plugin_id, handler_name, handler)
        if self._track_registration is not None:
            self._track_registration(
                handler_name,
                lambda: self._service.unregister_handler(
                    self._plugin_id, handler_name, handler
                ),
            )

    def enqueue(
        self,
        *,
        session_key: str,
        delivery_source: dict[str, object],
        question: str,
        context: dict[str, object],
        dedupe_key: str,
        handler_name: str,
    ) -> DeferredQuestion:
        from hermes_cli.plugin_capabilities import plugin_capability_granted

        if not plugin_capability_granted(self._plugin_id, "gateway.platform_actions"):
            raise PermissionError(
                "gateway.platform_actions capability is required for deferred questions"
            )
        return self._service.enqueue(
            plugin_id=self._plugin_id,
            session_key=session_key,
            delivery_source=delivery_source,
            question=question,
            handler_name=handler_name,
            context=context,
            dedupe_key=dedupe_key,
        )


_singleton_lock = threading.Lock()
_singletons: dict[Path, DeferredQuestionService] = {}


def get_deferred_question_service() -> DeferredQuestionService:
    """Return the profile-scoped host service shared by plugins and adapters."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home().expanduser().resolve(strict=False)
    with _singleton_lock:
        service = _singletons.get(home)
        if service is None:
            service = DeferredQuestionService(home / "state.db")
            _singletons[home] = service
        return service
