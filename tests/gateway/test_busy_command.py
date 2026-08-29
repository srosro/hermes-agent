"""Smoke tests for gateway /busy command dispatch."""

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionSource


def _make_runner(busy_mode="interrupt"):
    """Create a GatewayRunner with known busy mode."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    runner._busy_input_mode = busy_mode
    runner._busy_text_mode = "queue" if busy_mode == "queue" else "interrupt"
    runner.adapters = {}
    runner._profile_adapters = {}
    return runner


def _make_event(
    text: str,
    chat_id: str = "chat-test",
    platform: Platform = Platform.TELEGRAM,
    thread_id: str | None = None,
    message_id: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(text=text, source=source, message_id=message_id)


class TestBusyCommand:
    """Test /busy command dispatch without config persistence."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("text", "initial"),
        [("/busy", "steer"), ("/busy status", "queue")],
    )
    async def test_status_returns_current_mode(self, text, initial):
        """Bare /busy and /busy status show the current busy mode."""
        runner = _make_runner(busy_mode=initial)
        event = _make_event(text)
        result = await runner._handle_busy_command(event)
        reply_text = str(result).lower()
        assert initial in reply_text
        assert "busy" in reply_text

    @pytest.mark.asyncio
    async def test_busy_invalid_arg(self):
        """/busy with invalid arg returns error."""
        runner = _make_runner()
        event = _make_event("/busy bananas")
        result = await runner._handle_busy_command(event)
        assert "unknown" in str(result).lower()

class TestBusyCommandPersistence:
    """Test /busy persistence with mocked save_config_value."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
    async def test_set_mode_persists(self, monkeypatch, mode):
        """Each supported mode is saved and applied immediately."""
        runner = _make_runner(busy_mode="interrupt" if mode != "interrupt" else "queue")
        monkeypatch.setattr("cli.save_config_values", lambda values: True)
        event = _make_event(f"/busy {mode}")
        result = await runner._handle_busy_command(event)
        assert mode in str(result).lower()
        assert runner._busy_input_mode == mode
        assert runner._busy_text_mode == ("queue" if mode == "queue" else "interrupt")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "managed_key",
        ["display.busy_input_mode", "display.busy_text_mode"],
    )
    async def test_managed_mode_is_not_written(self, monkeypatch, managed_key):
        """A managed input or compatibility key blocks persistence."""
        runner = _make_runner(busy_mode="interrupt")
        saved = []
        monkeypatch.setattr(
            "hermes_cli.managed_scope.is_key_managed",
            lambda key: key == managed_key,
        )
        monkeypatch.setattr(
            "cli.save_config_values", lambda values: saved.append(values)
        )

        result = await runner._handle_busy_command(_make_event("/busy queue"))

        assert "administrator" in str(result).lower()
        assert saved == []
        assert runner._busy_input_mode == "interrupt"

    @pytest.mark.asyncio
    async def test_save_failure_preserves_mode(self, monkeypatch):
        """When save_config_value returns False, mode is unchanged."""
        runner = _make_runner(busy_mode="steer")
        monkeypatch.setattr("cli.save_config_values", lambda values: False)
        event = _make_event("/busy queue")
        result = await runner._handle_busy_command(event)
        assert "unchanged" in str(result).lower()
        assert runner._busy_input_mode == "steer"
