from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


def _service(path: Path):
    from gateway.deferred_questions import DeferredQuestionService

    return DeferredQuestionService(path)


def _register_default_handler(service) -> None:
    if service.has_handler("plow-chat", "invite-consent"):
        return
    from gateway.deferred_questions import DeferredQuestionResult

    async def default_handler(_record, _answer):
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", default_handler)


def _enqueue(
    service,
    *,
    dedupe_key: str = "invite-consent",
    register_handler: bool = True,
):
    if register_handler:
        _register_default_handler(service)
    return service.enqueue(
        plugin_id="plow-chat",
        session_key="plow_chat:home:owner",
        delivery_source={
            "platform": "plow_chat",
            "chat_id": "home",
            "chat_type": "dm",
            "user_id": "owner",
        },
        question="May I send invites?",
        handler_name="invite-consent",
        context={"source_chat_uid": "cht_source"},
        dedupe_key=dedupe_key,
    )


def _awaiting_question(service, *, adapter=None):
    question = _enqueue(service)
    adapter = adapter or _DeliveryAdapter(service, active=False)
    service._adapters[("plow_chat", "default")] = (
        adapter,
        __import__("asyncio").get_running_loop(),
    )
    service._ready_adapters.add(("plow_chat", "default"))
    service.claim_for_delivery(question.id)
    service.mark_awaiting(question.id)
    return question, adapter


def _awaiting_from_source(
    service, source, session_key: str, *, dedupe_key="invite-consent"
):
    question = service.enqueue(
        plugin_id="plow-chat",
        session_key=session_key,
        delivery_source=source.to_dict(),
        question="May I send invites?",
        handler_name="invite-consent",
        context={},
        dedupe_key=dedupe_key,
    )
    service.claim_for_delivery(question.id)
    service.mark_awaiting(question.id)
    return question


def test_enqueue_deduplicates_one_unresolved_question(tmp_path: Path) -> None:
    service = _service(tmp_path / "questions.sqlite3")

    first = _enqueue(service)
    second = _enqueue(service)

    assert second.id == first.id
    assert service.pending_for_session(first.session_key) == first


def test_pending_uses_queue_tiebreaker_for_equal_timestamps(tmp_path: Path) -> None:
    service = _service(tmp_path / "questions.sqlite3")

    with (
        patch("gateway.deferred_questions.time.time", return_value=1.0),
        patch("gateway.deferred_questions.uuid.uuid4") as uuid4,
    ):
        uuid4.side_effect = [
            SimpleNamespace(hex="b-question"),
            SimpleNamespace(hex="a-question"),
        ]
        _enqueue(service, dedupe_key="later-id")
        lower_id = _enqueue(service, dedupe_key="lower-id")

    assert service.pending_for_session(lower_id.session_key) == lower_id


def test_each_transaction_closes_its_sqlite_connection(tmp_path: Path) -> None:
    import sqlite3

    service = _service(tmp_path / "questions.sqlite3")
    with service._transaction() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_response_is_persisted_before_handler_and_resolves(
    tmp_path: Path,
) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    question, _adapter = _awaiting_question(service)
    observed = []

    async def handle(record, answer):
        reopened = _service(tmp_path / "questions.sqlite3")
        captured = reopened.get(record.id)
        observed.append((captured.state, captured.response, answer))
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)

    result = await service.handle_response(question.session_key, "Sure!")

    assert result == DeferredQuestionResult.done("Consent recorded.")
    assert observed == [("handling", "Sure!", "Sure!")]
    with pytest.raises(KeyError):
        service.get(question.id)
    assert service.pending_for_session(question.session_key) is None


@pytest.mark.asyncio
async def test_handler_runs_in_its_registration_context(tmp_path: Path) -> None:
    import contextvars

    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    question, _adapter = _awaiting_question(service)
    active_profile = contextvars.ContextVar("active_profile", default="default")
    observed = []

    async def handle(_record, _answer):
        observed.append(active_profile.get())
        return DeferredQuestionResult.done("Consent recorded.")

    token = active_profile.set("work")
    try:
        service.register_handler("plow-chat", "invite-consent", handle)
    finally:
        active_profile.reset(token)

    await service.handle_response(question.session_key, "yes")

    assert observed == ["work"]


@pytest.mark.asyncio
async def test_reconnect_recovery_does_not_rerun_active_handler(tmp_path: Path) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    question, adapter = _awaiting_question(service)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handle(_record, _answer):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)
    handling = asyncio.create_task(
        service.handle_response(question.session_key, "Sure!")
    )
    await started.wait()

    service.adapter_connected("plow_chat", adapter)
    await asyncio.sleep(0)
    assert calls == 1
    recovery = service._recovery_task
    release.set()
    assert await handling == DeferredQuestionResult.done("Consent recorded.")
    if recovery is not None:
        await recovery
    await asyncio.sleep(0)
    assert service._handling_retry_tasks == {}


@pytest.mark.asyncio
async def test_reconnect_wake_retries_after_active_handler_fails(
    tmp_path: Path,
) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    question, _adapter = _awaiting_question(service)
    service.handling_retry_seconds = 0
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handle(_record, _answer):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise RuntimeError("temporary")
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)
    handling = asyncio.create_task(
        service.handle_response(question.session_key, "Sure!")
    )
    await started.wait()

    await asyncio.sleep(0)
    assert calls == 1
    recovery = service._recovery_task
    assert recovery is not None
    release.set()
    with pytest.raises(RuntimeError, match="temporary"):
        await handling
    await recovery

    assert calls == 2
    with pytest.raises(KeyError):
        service.get(question.id)


@pytest.mark.asyncio
async def test_reconnect_wake_runs_second_recovery_pass_for_skipped_platform(
    tmp_path: Path,
) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")

    def enqueue(platform: str, key: str):
        return service.enqueue(
            plugin_id="plow-chat",
            session_key=f"{platform}:home:owner",
            delivery_source={
                "platform": platform,
                "chat_id": "home",
                "user_id": "owner",
            },
            question="May I send invites?",
            handler_name="invite-consent",
            context={},
            dedupe_key=key,
        )

    skipped = enqueue("platform_b", "b")
    active = enqueue("platform_a", "a")
    with service._lock, service._transaction() as conn:
        conn.execute(
            "UPDATE deferred_questions SET state = 'handling', response = 'yes'"
        )

    adapter_a = _DeliveryAdapter(service, active=False)
    adapter_b = _DeliveryAdapter(service, active=False)
    service.bind_adapter("platform_a", adapter_a)
    service.bind_adapter("platform_b", adapter_b)
    service.adapter_connected("platform_a", adapter_a)
    started = asyncio.Event()
    release = asyncio.Event()
    handled: list[str] = []

    async def handle(record, _answer):
        handled.append(record.id)
        if record.id == active.id:
            started.set()
            await release.wait()
        return DeferredQuestionResult.done("")

    service.register_handler("plow-chat", "invite-consent", handle)
    await asyncio.sleep(0)
    recovery = service._recovery_task
    assert recovery is not None
    await started.wait()

    service.adapter_connected("platform_b", adapter_b)
    release.set()
    await recovery

    assert handled == [active.id, skipped.id]
    with pytest.raises(KeyError):
        service.get(active.id)
    with pytest.raises(KeyError):
        service.get(skipped.id)


@pytest.mark.asyncio
async def test_overlapping_recovery_does_not_retry_ambiguous_delivery(
    tmp_path: Path,
) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")

    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[
            SimpleNamespace(success=False, error="delivery timed out", retryable=False)
        ],
    )
    question, _adapter = _awaiting_question(service, adapter=adapter)
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle(_record, _answer):
        started.set()
        await release.wait()
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)
    handling = asyncio.create_task(
        service.handle_response(question.session_key, "Sure!")
    )
    await started.wait()
    recovery = asyncio.create_task(service.retry_handling())
    await asyncio.sleep(0)

    release.set()
    with pytest.raises(RuntimeError, match="delivery timed out"):
        await handling
    assert await recovery == []

    with pytest.raises(KeyError):
        service.get(question.id)
    assert adapter.attempts == 1


@pytest.mark.asyncio
async def test_resolved_ambiguous_ack_does_not_block_next_question(
    tmp_path: Path,
) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    first = _enqueue(service, dedupe_key="first")
    second = _enqueue(service, dedupe_key="second")

    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[
            SimpleNamespace(success=False, error="delivery timed out", retryable=False)
        ],
    )
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)
    service.claim_for_delivery(first.id)
    service.mark_awaiting(first.id)

    async def handle(_record, _answer):
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)

    with pytest.raises(RuntimeError, match="delivery timed out"):
        await service.handle_response(first.session_key, "yes")

    with pytest.raises(KeyError):
        service.get(first.id)
    assert service.get(second.id).state == "awaiting"
    assert adapter.sent[-1] == ("home", second.question)


@pytest.mark.asyncio
async def test_unclear_response_reasks_same_question(tmp_path: Path) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    question, _adapter = _awaiting_question(service)

    async def handle(_record, _answer):
        return DeferredQuestionResult.clarify("Would you like me to send invites?")

    service.register_handler("plow-chat", "invite-consent", handle)

    result = await service.handle_response(question.session_key, "What do you mean?")

    assert result == DeferredQuestionResult.clarify(
        "Would you like me to send invites?"
    )
    pending = service.pending_for_session(question.session_key)
    assert pending is not None
    assert pending.id == question.id
    assert pending.state == "awaiting"
    assert pending.question == "Would you like me to send invites?"
    assert pending.response is None


@pytest.mark.asyncio
async def test_handling_response_retries_after_restart(tmp_path: Path) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    path = tmp_path / "questions.sqlite3"
    first_service = _service(path)
    question, _first_adapter = _awaiting_question(first_service)

    async def fail(_record, _answer):
        raise RuntimeError("temporary")

    first_service.register_handler("plow-chat", "invite-consent", fail)
    with pytest.raises(RuntimeError, match="temporary"):
        await first_service.handle_response(question.session_key, "Sure!")

    captured = first_service.get(question.id)
    assert captured.state == "handling"
    assert captured.response == "Sure!"

    restarted = _service(path)
    restarted_adapter = _DeliveryAdapter(restarted, active=False)
    restarted._adapters[("plow_chat", "default")] = (
        restarted_adapter,
        __import__("asyncio").get_running_loop(),
    )
    restarted._ready_adapters.add(("plow_chat", "default"))
    answers = []

    async def recover(_record, answer):
        answers.append(answer)
        return DeferredQuestionResult.done("Recovered.")

    restarted.register_handler("plow-chat", "invite-consent", recover)
    results = await restarted.retry_handling()

    assert results == [(question.id, DeferredQuestionResult.done("Recovered."))]
    assert answers == ["Sure!"]
    with pytest.raises(KeyError):
        restarted.get(question.id)


@pytest.mark.asyncio
async def test_reply_delivery_recovery_does_not_rerun_handler(tmp_path: Path) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    path = tmp_path / "questions.sqlite3"
    service = _service(path)

    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[SimpleNamespace(success=False, error="offline", retryable=True)],
    )
    question, _adapter = _awaiting_question(service, adapter=adapter)
    service.handling_retry_seconds = 0
    handler_calls = 0

    async def handle(_record, _answer):
        nonlocal handler_calls
        handler_calls += 1
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)

    with pytest.raises(RuntimeError, match="offline"):
        await service.handle_response(question.session_key, "Sure!")

    persisted = service.get(question.id)
    assert persisted.state == "handling"
    assert persisted.result == DeferredQuestionResult.done("Consent recorded.")

    await __import__("asyncio").sleep(0)
    retry_tasks = tuple(service._handling_retry_tasks.values())
    assert len(retry_tasks) == 1
    await __import__("asyncio").gather(*retry_tasks)

    assert handler_calls == 1
    assert adapter.attempts == 2
    with pytest.raises(KeyError):
        service.get(question.id)


@pytest.mark.asyncio
async def test_overlapping_adapter_binds_run_one_handling_recovery(
    tmp_path: Path,
) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult

    path = tmp_path / "questions.sqlite3"
    first = _service(path)
    question = _enqueue(first)
    first_adapter = _DeliveryAdapter(first, active=False)
    first._adapters[("plow_chat", "default")] = (
        first_adapter,
        asyncio.get_running_loop(),
    )
    first.claim_for_delivery(question.id)
    first.mark_awaiting(question.id)

    async def fail(_record, _answer):
        raise RuntimeError("temporary")

    first.register_handler("plow-chat", "invite-consent", fail)
    with pytest.raises(RuntimeError, match="temporary"):
        await first.handle_response(question.session_key, "yes")

    restarted = _service(path)
    adapter = _DeliveryAdapter(restarted, active=False)
    calls = 0

    async def recover(_record, _answer):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return DeferredQuestionResult.done("recovered")

    restarted.bind_adapter("plow_chat", adapter)
    restarted.bind_adapter("plow_chat", adapter)
    restarted.adapter_connected("plow_chat", adapter)
    restarted.register_handler("plow-chat", "invite-consent", recover)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    if restarted._recovery_task is not None:
        await restarted._recovery_task

    assert calls == 1
    assert adapter.sent == [("home", "recovered")]
    with pytest.raises(KeyError):
        restarted.get(question.id)


@pytest.mark.asyncio
async def test_resolved_dedupe_key_can_be_enqueued_again(tmp_path: Path) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    first, _adapter = _awaiting_question(service)

    async def handle(_record, _answer):
        return DeferredQuestionResult.done("done")

    service.register_handler("plow-chat", "invite-consent", handle)
    await service.handle_response(first.session_key, "yes")

    second = _enqueue(service)

    assert second.id != first.id


class _DeliveryAdapter:
    def __init__(self, service, *, active: bool, outcomes=()) -> None:
        self.service = service
        self.active = active
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.sent = []

    def is_session_active(self, session_key: str) -> bool:
        assert session_key == "plow_chat:home:owner"
        return self.active

    async def deliver_deferred_message(self, delivery_source, content):
        self.attempts += 1
        if self.outcomes:
            return self.outcomes.pop(0)
        pending = self.service.pending_for_session("plow_chat:home:owner")
        assert pending is not None
        assert pending.state in {"delivering", "handling"}
        self.sent.append((delivery_source["chat_id"], content))
        return SimpleNamespace(success=True, error=None, retryable=False)


class _GatewayAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True), platform)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="reply")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_deferred_delivery_restores_slack_workspace_and_thread() -> None:
    from gateway.session import SessionSource

    adapter = _GatewayAdapter(Platform.SLACK)
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="reply")
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="channel",
        chat_type="channel",
        thread_id="thread-ts",
        scope_id="workspace",
    )

    await adapter.deliver_deferred_message(source.to_dict(), "May I send invites?")

    adapter._send_with_retry.assert_awaited_once_with(
        "channel",
        "May I send invites?",
        reply_to=None,
        metadata={
            "thread_id": "thread-ts",
            "slack_team_id": "workspace",
            "notify": True,
        },
    )


@pytest.mark.asyncio
async def test_relay_deferred_delivery_uses_transport_and_restores_routing(
    tmp_path: Path,
) -> None:
    from gateway.session import SessionSource

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter(Platform.RELAY)
    adapter.prime_routing_cache = MagicMock()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel",
        chat_type="group",
        user_id="owner",
        scope_id="guild",
        delivery_transport="relay",
    )
    question = service.enqueue(
        plugin_id="plow-chat",
        session_key="discord:channel",
        delivery_source=source.to_dict(),
        question="May I send invites?",
        handler_name="invite-consent",
        context={},
        dedupe_key="invite-consent",
    )
    _register_default_handler(service)

    assert question.platform == "relay"
    service.bind_adapter("relay", adapter)
    service.adapter_connected("relay", adapter)
    await service.deliver_ready("relay")

    assert service.get(question.id).state == "awaiting"
    adapter.prime_routing_cache.assert_called_once()
    assert adapter.sent == [("channel", "May I send invites?", None, {"notify": True})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "release_path", ["cleanup", "direct", "stale_heal", "cleanup_concurrent"]
)
async def test_busy_question_wakes_only_after_session_guard_is_released(
    tmp_path: Path,
    release_path: str,
) -> None:
    import asyncio

    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter._running = True
    adapter.set_deferred_question_service(service)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="home",
        chat_type="dm",
        user_id="owner",
    )
    session_key = build_session_key(source)
    question = service.enqueue(
        plugin_id="plow-chat",
        session_key=session_key,
        delivery_source=source.to_dict(),
        question="May I send invites?",
        handler_name="invite-consent",
        context={},
        dedupe_key="invite-consent",
    )
    service.register_handler("plow-chat", "invite-consent", AsyncMock())
    guard = asyncio.Event()

    async def done():
        return None

    adapter._active_sessions[session_key] = guard
    owner = asyncio.create_task(done())
    adapter._session_tasks[session_key] = owner
    await owner

    await service.deliver_ready("telegram")

    assert adapter.sent == []
    assert session_key in adapter._session_idle_callbacks
    if release_path == "cleanup":
        adapter._cleanup_finished_session_task(session_key, guard)
    elif release_path in {"stale_heal", "cleanup_concurrent"}:
        from gateway.platforms.base import MessageEvent, MessageType

        finish_turn = asyncio.Event()
        delivery_started = asyncio.Event()
        finish_delivery = asyncio.Event()
        handler_started = asyncio.Event()

        async def handle_unrelated(_event):
            handler_started.set()
            await finish_turn.wait()
            return "ordinary"

        original_send = adapter.send

        async def block_deferred_send(*args, **kwargs):
            delivery_started.set()
            await finish_delivery.wait()
            return await original_send(*args, **kwargs)

        adapter._message_handler = handle_unrelated
        adapter.send = block_deferred_send
        if release_path == "cleanup_concurrent":
            adapter._cleanup_finished_session_task(session_key, guard)
            await delivery_started.wait()
        first_ingress = asyncio.create_task(
            adapter.handle_message(
                MessageEvent(
                    text="first unrelated",
                    source=source,
                    message_id="first-unrelated-turn",
                    message_type=MessageType.TEXT,
                )
            )
        )
        if release_path == "stale_heal":
            await delivery_started.wait()
        second_ingress = asyncio.create_task(
            adapter.handle_message(
                MessageEvent(
                    text="second unrelated",
                    source=source,
                    message_id="second-unrelated-turn",
                    message_type=MessageType.TEXT,
                )
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        handler_started_before_delivery = handler_started.is_set()
        finish_delivery.set()
        await asyncio.gather(first_ingress, second_ingress)
    else:
        adapter._release_session_guard(session_key, guard=guard)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    state_before_turn_finishes = service.get(question.id).state
    sent_before_turn_finishes = list(adapter.sent)
    if release_path in {"stale_heal", "cleanup_concurrent"}:
        finish_turn.set()
        await asyncio.gather(*tuple(adapter._background_tasks))
        assert not handler_started_before_delivery
    assert state_before_turn_finishes == "awaiting"
    assert sent_before_turn_finishes == [
        ("home", "May I send invites?", None, {"notify": True})
    ]


@pytest.mark.asyncio
async def test_reply_exposed_during_idle_delivery_is_captured(tmp_path: Path) -> None:
    import asyncio

    from gateway.deferred_questions import DeferredQuestionResult
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter._running = True
    adapter.set_deferred_question_service(service)
    adapter.set_authorization_check(lambda *_args: True)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="home",
        chat_type="dm",
        user_id="owner",
    )
    session_key = build_session_key(source)
    handled = []

    async def handle_deferred(_record, answer):
        handled.append(answer)
        return DeferredQuestionResult.done("")

    service.register_handler("plow-chat", "invite-consent", handle_deferred)
    service.enqueue(
        plugin_id="plow-chat",
        session_key=session_key,
        delivery_source=source.to_dict(),
        question="May I send invites?",
        handler_name="invite-consent",
        context={},
        dedupe_key="invite-consent",
    )
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    adapter._session_tasks[session_key] = asyncio.current_task()
    await service.deliver_ready("telegram")

    delivery_exposed = asyncio.Event()
    finish_delivery = asyncio.Event()
    original_send = adapter.send

    async def expose_then_block(*args, **kwargs):
        result = await original_send(*args, **kwargs)
        delivery_exposed.set()
        await finish_delivery.wait()
        return result

    adapter.send = expose_then_block
    ordinary_handler = AsyncMock(return_value="ordinary")
    adapter._message_handler = ordinary_handler
    adapter._cleanup_finished_session_task(session_key, guard)
    await delivery_exposed.wait()
    reply = asyncio.create_task(
        adapter.handle_message(
            MessageEvent(
                text="yes",
                source=source,
                message_id="fast-reply",
                message_type=MessageType.TEXT,
            )
        )
    )
    finish_delivery.set()
    await reply
    await asyncio.gather(*tuple(adapter._background_tasks))

    assert handled == ["yes"]
    ordinary_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_profile_owned_adapter_delivers_its_own_questions(tmp_path: Path) -> None:
    import asyncio

    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.deferred_questions import DeferredQuestionResult
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    primary = _GatewayAdapter()
    primary.set_owner_profile("work")
    primary._running = True
    primary.set_deferred_question_service(service, profile_name="default")
    primary.set_authorization_check(lambda *_args: True)
    primary._message_handler = AsyncMock(return_value="ordinary")
    secondary = _GatewayAdapter()
    secondary.set_owner_profile("default")
    secondary._running = True
    secondary.set_deferred_question_service(service)
    secondary.set_authorization_check(lambda *_args: True)
    secondary._message_handler = AsyncMock(return_value="ordinary")

    class Runner(GatewayAuthorizationMixin):
        pass

    runner = Runner()
    runner.adapters = {Platform.TELEGRAM: secondary}
    runner._profile_adapters = {"work": {Platform.TELEGRAM: primary}}
    primary.gateway_runner = runner
    handled = []

    async def handle(record, answer):
        handled.append((record.id, answer))
        return DeferredQuestionResult.done("")

    service.register_handler("plow-chat", "invite-consent", handle)

    primary_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shared-chat",
        chat_type="dm",
        user_id="owner",
        profile="default",
        adapter_profile="work",
    )
    secondary_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shared-chat",
        chat_type="dm",
        user_id="owner",
        profile="default",
        adapter_profile="default",
    )
    session_key = build_session_key(primary_source, profile="default")
    primary_question = service.enqueue(
        plugin_id="plow-chat",
        session_key=session_key,
        delivery_source=primary_source.to_dict(),
        question="Primary prompt",
        handler_name="invite-consent",
        context={},
        dedupe_key="primary",
    )
    secondary_question = service.enqueue(
        plugin_id="plow-chat",
        session_key=session_key,
        delivery_source=secondary_source.to_dict(),
        question="Secondary prompt",
        handler_name="invite-consent",
        context={},
        dedupe_key="secondary",
    )

    await service.deliver_ready("telegram")

    assert service.get(primary_question.id).state == "awaiting"
    assert service.get(secondary_question.id).state == "queued"
    assert primary.sent == [("shared-chat", "Primary prompt", None, {"notify": True})]
    assert secondary.sent == []

    await secondary.handle_message(
        MessageEvent(
            text="too early",
            source=SessionSource.from_dict(secondary_source.to_dict()),
            message_id="secondary-early",
            message_type=MessageType.TEXT,
        )
    )
    await asyncio.gather(*tuple(secondary._background_tasks))

    assert handled == []
    assert service.get(primary_question.id).state == "awaiting"

    await primary.handle_message(
        MessageEvent(
            text="yes-primary",
            source=SessionSource.from_dict(primary_source.to_dict()),
            message_id="primary-answer",
            message_type=MessageType.TEXT,
        )
    )

    assert handled == [(primary_question.id, "yes-primary")]
    assert service.get(secondary_question.id).state == "awaiting"
    assert secondary.sent[-1] == (
        "shared-chat",
        "Secondary prompt",
        None,
        {"notify": True},
    )

    await secondary.handle_message(
        MessageEvent(
            text="yes-secondary",
            source=SessionSource.from_dict(secondary_source.to_dict()),
            message_id="secondary-answer",
            message_type=MessageType.TEXT,
        )
    )

    assert handled == [
        (primary_question.id, "yes-primary"),
        (secondary_question.id, "yes-secondary"),
    ]


@pytest.mark.asyncio
async def test_question_waits_while_plugin_handler_is_unavailable(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "questions.sqlite3")
    question = _enqueue(service, register_handler=False)
    adapter = _DeliveryAdapter(service, active=False)
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)

    await service.deliver_ready("plow_chat")

    assert adapter.sent == []
    assert service.get(question.id).state == "queued"

    service.register_handler("plow-chat", "invite-consent", AsyncMock())
    await __import__("asyncio").sleep(0)
    await __import__("asyncio").sleep(0)

    assert adapter.sent == [("home", "May I send invites?")]
    assert service.get(question.id).state == "awaiting"


@pytest.mark.asyncio
async def test_reply_is_not_consumed_while_plugin_handler_is_unavailable(
    tmp_path: Path,
) -> None:
    import asyncio

    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter._running = True
    adapter.set_deferred_question_service(service)
    adapter.set_authorization_check(lambda *_args: True)
    adapter._message_handler = AsyncMock(return_value="ordinary")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="home",
        chat_type="dm",
        user_id="owner",
    )
    session_key = build_session_key(source)
    question = _awaiting_from_source(service, source, session_key)

    async def unloaded(_record, _answer):
        raise AssertionError("disabled plugin handler must not run")

    service.register_handler("plow-chat", "invite-consent", unloaded)
    service.unregister_handler("plow-chat", "invite-consent", unloaded)

    await adapter.handle_message(
        MessageEvent(
            text="yes",
            source=source,
            message_id="msg-answer",
            message_type=MessageType.TEXT,
        )
    )
    await asyncio.gather(*tuple(adapter._background_tasks))

    assert service.get(question.id).state == "queued"
    adapter._message_handler.assert_awaited_once()
    assert adapter.sent == [("home", "ordinary", "msg-answer", {"notify": True})]

    service.register_handler("plow-chat", "invite-consent", AsyncMock())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert service.get(question.id).state == "awaiting"
    assert adapter.sent[-1] == (
        "home",
        "May I send invites?",
        None,
        {"notify": True},
    )


@pytest.mark.asyncio
async def test_failed_delivery_returns_question_to_queue(tmp_path: Path) -> None:
    service = _service(tmp_path / "questions.sqlite3")
    question = _enqueue(service)
    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[SimpleNamespace(success=False, error="offline", retryable=True)],
    )
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)

    await service.deliver_ready("plow_chat")

    assert service.get(question.id).state == "queued"


@pytest.mark.asyncio
async def test_disconnect_before_prompt_delivery_leaves_question_retryable(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "questions.sqlite3")
    question = _enqueue(service)

    class DisconnectingAdapter(_DeliveryAdapter):
        def is_session_active(self, session_key: str) -> bool:
            service.adapter_disconnected("plow_chat", self)
            return False

    adapter = DisconnectingAdapter(service, active=False)
    service._adapters[("plow_chat", "default")] = (
        adapter,
        __import__("asyncio").get_running_loop(),
    )
    service._ready_adapters.add(("plow_chat", "default"))

    await service.deliver_ready("plow_chat")

    assert adapter.sent == []
    assert service.get(question.id).state == "queued"


@pytest.mark.asyncio
async def test_disconnected_handling_recovery_waits_for_reconnect(
    tmp_path: Path,
) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")

    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[SimpleNamespace(success=False, error="offline", retryable=True)],
    )
    question, _adapter = _awaiting_question(service, adapter=adapter)
    service.handling_retry_seconds = 0
    handler_calls = 0

    async def handle(_record, _answer):
        nonlocal handler_calls
        handler_calls += 1
        return DeferredQuestionResult.done("Consent recorded.")

    service.register_handler("plow-chat", "invite-consent", handle)
    with pytest.raises(RuntimeError, match="offline"):
        await service.handle_response(question.session_key, "Sure!")

    service.adapter_disconnected("plow_chat", adapter)
    await __import__("asyncio").sleep(0)
    retry_tasks = tuple(service._handling_retry_tasks.values())
    await __import__("asyncio").gather(*retry_tasks)

    assert handler_calls == 1
    assert adapter.attempts == 1
    assert service.get(question.id).state == "handling"

    service.adapter_connected("plow_chat", adapter)
    await __import__("asyncio").sleep(0)
    await __import__("asyncio").sleep(0)

    assert handler_calls == 1
    assert adapter.attempts == 2
    with pytest.raises(KeyError):
        service.get(question.id)


@pytest.mark.asyncio
async def test_ambiguous_delivery_is_not_retried_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "questions.sqlite3"
    service = _service(path)
    question = _enqueue(service)

    ambiguous = SimpleNamespace(
        success=False, error="delivery timed out", retryable=False
    )
    first_adapter = _DeliveryAdapter(service, active=False, outcomes=[ambiguous])
    service._adapters[("plow_chat", "default")] = (
        first_adapter,
        __import__("asyncio").get_running_loop(),
    )
    service._ready_adapters.add(("plow_chat", "default"))
    await service.deliver_ready("plow_chat")

    assert first_adapter.attempts == 1
    assert service.get(question.id).state == "delivering"

    restarted = _service(path)
    restarted_adapter = _DeliveryAdapter(restarted, active=False, outcomes=[ambiguous])
    restarted.bind_adapter("plow_chat", restarted_adapter)
    restarted.adapter_connected("plow_chat", restarted_adapter)
    await __import__("asyncio").sleep(0)
    await __import__("asyncio").sleep(0)

    assert restarted_adapter.attempts == 0
    assert restarted.get(question.id).state == "delivering"


@pytest.mark.asyncio
async def test_failed_delivery_waits_for_an_external_wake(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "questions.sqlite3")
    question = _enqueue(service)

    adapter = _DeliveryAdapter(
        service,
        active=False,
        outcomes=[SimpleNamespace(success=False, error="offline", retryable=True)],
    )
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)
    await __import__("asyncio").sleep(0)
    await __import__("asyncio").sleep(0)

    assert adapter.attempts == 1
    assert service.get(question.id).state == "queued"

    await service.deliver_ready("plow_chat")

    assert adapter.attempts == 2
    assert service.get(question.id).state == "awaiting"


@pytest.mark.asyncio
async def test_wake_does_not_copy_the_enqueuers_context_into_delivery(
    tmp_path: Path,
) -> None:
    import contextvars

    service = _service(tmp_path / "questions.sqlite3")
    marker = contextvars.ContextVar("deferred_delivery_marker", default=None)

    class ContextAdapter(_DeliveryAdapter):
        def __init__(self, service) -> None:
            super().__init__(service, active=False)
            self.context_values = []

        async def deliver_deferred_message(self, delivery_source, content):
            self.context_values.append(marker.get())
            return await super().deliver_deferred_message(delivery_source, content)

    adapter = ContextAdapter(service)
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)
    token = marker.set("member-turn")
    try:
        _enqueue(service)
    finally:
        marker.reset(token)
    for _ in range(3):
        await __import__("asyncio").sleep(0)

    assert adapter.context_values == [None]


def test_plugin_client_namespaces_handler_and_dedupe_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.deferred_questions import DeferredQuestionClient
    from hermes_cli import plugin_capabilities

    service = _service(tmp_path / "questions.sqlite3")
    client = DeferredQuestionClient(service, "plow-chat")
    monkeypatch.setattr(
        plugin_capabilities,
        "plugin_capability_granted",
        lambda *_args, **_kwargs: True,
    )

    async def handler(_record, _answer):
        raise AssertionError("not called")

    client.register_handler("invite-consent", handler)
    question = client.enqueue(
        session_key="plow_chat:home:owner",
        delivery_source={"platform": "plow_chat", "chat_id": "home"},
        question="May I send invites?",
        context={"source_chat_uid": "cht_source"},
        dedupe_key="owner-consent",
        handler_name="invite-consent",
    )

    assert question.plugin_id == "plow-chat"
    assert question.handler_name == "invite-consent"
    assert question.dedupe_key == "owner-consent"


def test_plugin_client_denies_enqueue_without_platform_action_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway.deferred_questions import DeferredQuestionClient
    from hermes_cli import plugin_capabilities

    service = _service(tmp_path / "questions.sqlite3")
    client = DeferredQuestionClient(service, "plow-chat")
    monkeypatch.setattr(
        plugin_capabilities,
        "plugin_capability_granted",
        lambda plugin_id, capability: False,
    )

    with pytest.raises(PermissionError, match="gateway.platform_actions"):
        client.enqueue(
            session_key="plow_chat:home:owner",
            delivery_source={"platform": "plow_chat", "chat_id": "home"},
            question="May I send invites?",
            context={},
            dedupe_key="owner-consent",
            handler_name="invite-consent",
        )

    assert service.pending_for_session("plow_chat:home:owner") is None


def test_stale_handler_cleanup_does_not_remove_replacement(tmp_path: Path) -> None:
    service = _service(tmp_path / "questions.sqlite3")

    async def old(_record, _answer):
        raise AssertionError("not called")

    async def replacement(_record, _answer):
        raise AssertionError("not called")

    service.register_handler("plow-chat", "invite-consent", old)
    service.register_handler("plow-chat", "invite-consent", replacement)
    service.unregister_handler("plow-chat", "invite-consent", old)

    assert service._handlers[("plow-chat", "invite-consent")][0] is replacement


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True], ids=["queued", "delivering"])
async def test_binding_adapter_recovers_question_after_restart(
    tmp_path: Path, interrupted: bool
) -> None:
    path = tmp_path / "questions.sqlite3"
    first = _service(path)
    question = _enqueue(first)
    if interrupted:
        first.claim_for_delivery(question.id)
        assert first.get(question.id).state == "delivering"

    restarted = _service(path)
    _register_default_handler(restarted)
    adapter = _DeliveryAdapter(restarted, active=False)
    restarted.bind_adapter("plow_chat", adapter)
    await __import__("asyncio").sleep(0)

    assert adapter.sent == []

    restarted.adapter_connected("plow_chat", adapter)
    await __import__("asyncio").sleep(0)
    await __import__("asyncio").sleep(0)

    assert adapter.sent == [("home", "May I send invites?")]
    assert restarted.get(question.id).state == "awaiting"


@pytest.mark.asyncio
async def test_only_oldest_question_in_a_session_is_delivered(tmp_path: Path) -> None:
    from gateway.deferred_questions import DeferredQuestionResult

    service = _service(tmp_path / "questions.sqlite3")
    first = _enqueue(service, dedupe_key="first")
    second = _enqueue(service, dedupe_key="second")
    adapter = _DeliveryAdapter(service, active=False)
    service.bind_adapter("plow_chat", adapter)
    service.adapter_connected("plow_chat", adapter)
    await service.deliver_ready("plow_chat")

    assert adapter.sent == [("home", first.question)]
    assert service.get(first.id).state == "awaiting"
    assert service.get(second.id).state == "queued"

    async def handle(_record, _answer):
        return DeferredQuestionResult.done("first resolved")

    service.register_handler("plow-chat", "invite-consent", handle)
    await service.handle_response(first.session_key, "yes")

    assert adapter.sent == [
        ("home", first.question),
        ("home", "first resolved"),
        ("home", second.question),
    ]
    assert service.get(second.id).state == "awaiting"


def test_plugin_context_exposes_plugin_scoped_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gateway import deferred_questions
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    service = _service(tmp_path / "questions.sqlite3")
    monkeypatch.setattr(
        deferred_questions, "get_deferred_question_service", lambda: service
    )
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="Plow Chat", key="plow-chat"), manager)

    assert context.deferred_questions is context.deferred_questions

    async def handler(_record, _answer):
        raise AssertionError("not called")

    context.deferred_questions.register_handler("invite-consent", handler)
    assert service._handlers[("plow-chat", "invite-consent")][0] is handler
    assert manager.unload("plow-chat")
    assert ("plow-chat", "invite-consent") not in service._handlers


def test_host_service_is_scoped_to_the_active_hermes_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_constants
    from gateway import deferred_questions

    active = tmp_path / "profile-a"
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: active)
    monkeypatch.setattr(deferred_questions, "_singletons", {})

    first = deferred_questions.get_deferred_question_service()
    again = deferred_questions.get_deferred_question_service()
    active = tmp_path / "profile-b"
    second = deferred_questions.get_deferred_question_service()

    assert again is first
    assert second is not first
    assert first.path.parent == tmp_path / "profile-a"
    assert second.path.parent == tmp_path / "profile-b"
    assert first.path.name == "state.db"


def test_gateway_setup_surfaces_deferred_store_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import deferred_questions

    def fail():
        raise OSError("store unavailable")

    monkeypatch.setattr(deferred_questions, "get_deferred_question_service", fail)
    adapter = _GatewayAdapter()

    with pytest.raises(OSError, match="store unavailable"):
        adapter.set_message_handler(AsyncMock())


@pytest.mark.asyncio
async def test_adapter_intercepts_deferred_reply_before_busy_queue(
    tmp_path: Path,
) -> None:
    from gateway.deferred_questions import DeferredQuestionResult
    from gateway.platforms.base import (
        MessageEvent,
        MessageType,
    )
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter.set_deferred_question_service(service)
    service.adapter_connected("telegram", adapter)
    adapter.set_authorization_check(lambda *_args: True)
    adapter._message_handler = AsyncMock(return_value="ordinary")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="home",
        chat_type="dm",
        user_id="owner",
    )
    session_key = build_session_key(source)
    question = _awaiting_from_source(service, source, session_key)

    async def handle(_record, answer):
        assert answer == "Sure!"
        return DeferredQuestionResult.done("Great — I’ll send it now.")

    service.register_handler("plow-chat", "invite-consent", handle)
    adapter._active_sessions[session_key] = __import__("asyncio").Event()
    event = MessageEvent(
        text="Sure!",
        source=source,
        message_id="msg-answer",
        message_type=MessageType.TEXT,
    )

    await adapter.handle_message(event)

    assert adapter.sent == [
        ("home", "Great — I’ll send it now.", None, {"notify": True})
    ]
    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_not_awaited()
    with pytest.raises(KeyError):
        service.get(question.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin_user_id",
    ["owner", None],
    ids=["different-sender", "missing-origin-identity"],
)
async def test_adapter_leaves_question_pending_for_different_authorized_sender(
    tmp_path: Path,
    origin_user_id: str | None,
) -> None:
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter.config.extra["group_sessions_per_user"] = False
    adapter.set_deferred_question_service(service)
    adapter.set_authorization_check(lambda *_args: True)
    adapter._message_handler = AsyncMock(return_value="ordinary")
    owner = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shared",
        chat_type="group",
        user_id=origin_user_id,
    )
    session_key = build_session_key(owner, group_sessions_per_user=False)
    question = _awaiting_from_source(service, owner, session_key)
    handler = AsyncMock()
    service.register_handler("plow-chat", "invite-consent", handler)
    other = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shared",
        chat_type="group",
        user_id="other-member",
    )

    await adapter.handle_message(
        MessageEvent(
            text="yes",
            source=other,
            message_id="msg-other",
            message_type=MessageType.TEXT,
        )
    )

    assert service.get(question.id).state == "awaiting"
    handler.assert_not_awaited()
    adapter._message_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapter_selects_deferred_service_by_routed_profile(
    tmp_path: Path,
) -> None:
    from gateway.deferred_questions import DeferredQuestionResult
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    default_service = _service(tmp_path / "default.sqlite3")
    work_service = _service(tmp_path / "work.sqlite3")
    adapter = _GatewayAdapter()
    adapter.set_deferred_question_service(default_service)
    adapter.set_deferred_question_service(work_service, profile_name="work")
    work_service.adapter_connected("telegram", adapter)
    adapter.set_authorization_check(lambda *_args: True)
    adapter._message_handler = AsyncMock(return_value="ordinary")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="home",
        chat_type="dm",
        user_id="owner",
        profile="work",
    )
    session_key = build_session_key(source, profile="work")
    question = _awaiting_from_source(work_service, source, session_key)

    async def handle(_record, _answer):
        return DeferredQuestionResult.done("Consent recorded.")

    work_service.register_handler("plow-chat", "invite-consent", handle)

    await adapter.handle_message(
        MessageEvent(
            text="yes",
            source=source,
            message_id="msg-work",
            message_type=MessageType.TEXT,
        )
    )

    with pytest.raises(KeyError):
        work_service.get(question.id)
    assert default_service.pending_for_session(session_key) is None
    adapter._message_handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [False, None], ids=["denied", "unknown"])
async def test_adapter_rejects_unauthorized_deferred_reply(
    tmp_path: Path, authorization: bool | None
) -> None:
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter.set_deferred_question_service(service)
    adapter.set_authorization_check(lambda *_args: authorization)
    adapter._message_handler = AsyncMock(return_value="ordinary")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="shared",
        chat_type="group",
        user_id="outsider" if authorization is False else "unknown",
    )
    question = _awaiting_from_source(service, source, build_session_key(source))
    handler = AsyncMock()
    service.register_handler("plow-chat", "invite-consent", handler)

    await adapter.handle_message(
        MessageEvent(
            text="yes",
            source=source,
            message_id="msg-outsider",
            message_type=MessageType.TEXT,
        )
    )

    assert service.get(question.id).state == "awaiting"
    handler.assert_not_awaited()
    adapter._message_handler.assert_not_awaited()
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_slash_command_bypasses_pending_deferred_question(tmp_path: Path) -> None:
    import asyncio

    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource, build_session_key

    service = _service(tmp_path / "questions.sqlite3")
    adapter = _GatewayAdapter()
    adapter.set_deferred_question_service(service)
    adapter._message_handler = AsyncMock(return_value="status response")
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="home", chat_type="dm", user_id="owner"
    )
    question = _awaiting_from_source(service, source, build_session_key(source))

    await adapter.handle_message(
        MessageEvent(
            text="/status",
            source=source,
            message_id="msg-command",
            message_type=MessageType.TEXT,
        )
    )
    await asyncio.gather(*tuple(adapter._background_tasks))

    assert service.get(question.id).state == "awaiting"
    adapter._message_handler.assert_awaited_once()
    assert adapter.sent == [
        ("home", "status response", "msg-command", {"notify": True})
    ]
