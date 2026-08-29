import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.relay.ws_transport import _event_from_wire


def _wire(text: str, *, platform: str = "slack") -> dict:
    return {
        "text": text,
        "message_type": "command",
        "source": {
            "platform": platform,
            "chat_id": "D123",
            "chat_type": "dm",
            "user_id": "U123",
        },
    }


@pytest.mark.parametrize(
    ("wire_text", "input_type", "expected", "expected_type"),
    [
        ("/hermes sethome", "command", "/sethome", MessageType.COMMAND),
        ("/hermes\tsethome", "command", "/sethome", MessageType.COMMAND),
        (
            "/hermes model gpt-5.6 --provider openai",
            "command",
            "/model gpt-5.6 --provider openai",
            MessageType.COMMAND,
        ),
        ("/hermes", "command", "/help", MessageType.COMMAND),
        ("!busy status", "text", "/busy status", MessageType.COMMAND),
        ("!", "text", "!", MessageType.TEXT),
    ],
)
def test_slack_relay_parent_becomes_gateway_command(
    wire_text: str,
    input_type: str,
    expected: str,
    expected_type: MessageType,
):
    wire = _wire(wire_text)
    wire["message_type"] = input_type
    event = _event_from_wire(wire)

    assert event.text == expected
    assert event.message_type == expected_type
    assert event.source.platform == Platform.SLACK
    assert event.source.delivered_via_upstream_relay is True
