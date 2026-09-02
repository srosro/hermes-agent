"""Regression tests for hidden-reasoning-only incomplete gateway turns."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome, SendResult
from gateway.session import SessionEntry, SessionSource, build_session_key


class CaptureSlackAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.SLACK)
        self.sent = []
        self.processing_hooks = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="slack-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.processing_hooks.append(("start", event.message_id))

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        self.processing_hooks.append(("complete", event.message_id, outcome))


def _make_incomplete_result() -> dict:
    # Mirror the REAL conversation-loop exhaustion shape: the sentinel text is
    # returned as BOTH final_response and error (agent/conversation_loop.py's
    # "remained incomplete after 3 continuation attempts" return).
    _sentinel = "Codex response remained incomplete after 3 continuation attempts"
    return {
        "final_response": _sentinel,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ],
        "tools": [],
        "history_offset": 0,
        "api_calls": 3,
        "partial": True,
        "completed": False,
        "interrupted": False,
        "error": _sentinel,
        "last_prompt_tokens": 0,
    }


def _make_runner(adapter: CaptureSlackAdapter) -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:slack:channel:C123:171717",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    # The transient-failure persistence path dedupes on platform message_id
    # (#47237). A bare MagicMock returns a truthy mock, which would wrongly
    # mark the user turn as a duplicate and skip persisting it.
    runner.session_store.has_platform_message_id = MagicMock(return_value=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(return_value=_make_incomplete_result())
    return runner


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="channel",
            thread_id="171717",
            user_id="U123",
        ),
        message_id="m-1",
    )


@pytest.mark.asyncio
async def test_incomplete_codex_turn_stays_out_of_slack_transcript(monkeypatch, tmp_path):
    adapter = CaptureSlackAdapter()
    runner = _make_runner(adapter)

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C123")

    adapter.set_message_handler(runner._handle_message)
    adapter._keep_typing = lambda *_args, **_kwargs: asyncio.Event().wait()

    event = _make_event()
    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.sent == []
    assert runner.session_store.update_session.called

    transcript_roles = [
        call.args[1]["role"]
        for call in runner.session_store.append_to_transcript.call_args_list
    ]
    assert transcript_roles == ["session_meta", "user"]
    assert runner.session_store.append_to_transcript.call_args_list[1].args[1]["content"] == "hello"
    assert adapter.processing_hooks == [
        ("start", "m-1"),
        ("complete", "m-1", ProcessingOutcome.SUCCESS),
    ]


# --------------------------------------------------------------------------
# Turn-stop explainer delivery boundary (plow-pbc/agent-mgr#107)
# The finalizer keeps the explainer inline on every surface, reports its exact
# text in result["turn_stop_explanation"], and emits a turn_stop.* status
# frame; these pin the gateway's consumer side — chat surfaces shed exactly
# that value (adapters with a diagnostic channel get the frame), raw surfaces
# keep the inline text, and quoted diagnostics in genuine answers survive.
# --------------------------------------------------------------------------
_EXPLAINER = "⚠️ No reply: the model returned empty content after retries."
_RAW_PLATFORM = next(iter(gateway_run._GATEWAY_RAW_TEXT_PLATFORMS))


@pytest.mark.parametrize(
    ("platform", "text", "expected"),
    [
        (Platform.SLACK, _EXPLAINER, ""),
        (Platform.SLACK, "fragment recovered." + "\n\n" + _EXPLAINER, "fragment recovered."),
        (_RAW_PLATFORM, _EXPLAINER, _EXPLAINER),
        # A genuine answer that merely quotes the diagnostic is not the
        # diagnostic: removal is by exact value, not wording.
        (Platform.SLACK, "The warning you saw means: " + _EXPLAINER, "The warning you saw means: " + _EXPLAINER),
    ],
    ids=["chat-strips", "chat-sheds-suffix", "raw-passthrough", "quoted-survives"],
)
def test_sanitize_turn_stop_explainer(platform, text, expected):
    assert (
        gateway_run._sanitize_gateway_final_response(
            platform, text, turn_stop_explanation=_EXPLAINER
        )
        == expected
    )


def test_sanitize_without_explanation_leaves_text_alone():
    assert (
        gateway_run._sanitize_gateway_final_response(Platform.SLACK, _EXPLAINER)
        == _EXPLAINER
    )


def test_status_coro_drops_turn_stop_without_diagnostic_channel():
    """Hookless adapters AND conversational status hooks (which post ordinary
    chat messages) both drop the frame; only a declared diagnostic channel
    receives it."""
    hookless = CaptureSlackAdapter()
    result = asyncio.run(
        gateway_run._send_or_update_status_coro(
            hookless, "chat-1", "turn_stop.empty_response_exhausted.abcd1234", _EXPLAINER, None
        )
    )
    assert result is None and hookless.sent == []

    conversational = CaptureSlackAdapter()
    hook_calls = []

    async def hook(chat_id, status_key, content, metadata=None):
        hook_calls.append(status_key)
        return SendResult(success=True)

    conversational.send_or_update_status = hook  # no delivers_diagnostic_status
    result = asyncio.run(
        gateway_run._send_or_update_status_coro(
            conversational, "chat-1", "turn_stop.empty_response_exhausted.abcd1234", _EXPLAINER, None
        )
    )
    assert result is None and hook_calls == [] and conversational.sent == []


def test_status_coro_delivers_turn_stop_to_diagnostic_channel():
    adapter = CaptureSlackAdapter()
    calls = []

    async def hook(chat_id, status_key, content, metadata=None):
        calls.append((chat_id, status_key, content))
        return SendResult(success=True)

    adapter.send_or_update_status = hook
    adapter.delivers_diagnostic_status = True
    result = asyncio.run(
        gateway_run._send_or_update_status_coro(
            adapter, "chat-1", "turn_stop.empty_response_exhausted.abcd1234", _EXPLAINER, None
        )
    )
    assert result.success
    assert calls and calls[0][1] == "turn_stop.empty_response_exhausted.abcd1234"
    assert adapter.sent == []
