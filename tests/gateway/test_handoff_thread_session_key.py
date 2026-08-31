"""CLI-handoff destinations must key exactly like each adapter's ORGANIC
replies, or the next real message in the handed-off conversation derives a
different session key and spawns a fresh session.

The construction is adapter-owned (``build_handoff_dest_source``): each
adapter returns the source its own inbound path would derive, and the gateway
consumes the result with no platform branches. These tests drive the real
hooks with stubbed IO — no mirrors of production logic.

Key shapes pinned here:
  - Discord threads key on the thread's OWN id (organic in-thread messages
    are built with ``chat_id == thread id``).
  - Slack never keys ``"thread"``: organic replies are workspace-scoped
    ``"dm"``/``"group"`` sources; the conversations API decides IM vs
    channel, legacy homes recover their workspace from the adapter, and a
    lookup failure preserves the thread/no-thread intent.
  - Telegram private-chat topics key as DM topics with the real user id.
"""

import asyncio

from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key


def _home(platform, chat_id, scope_id=None, user_id=None, thread_id=None, chat_type=None):
    return HomeChannel(
        platform=platform,
        chat_id=str(chat_id),
        name="home",
        thread_id=thread_id,
        user_id=user_id,
        scope_id=scope_id,
        chat_type=chat_type,
    )


def _dest(adapter, platform, home, thread_id):
    return asyncio.run(
        adapter.build_handoff_dest_source(
            platform=platform,
            home=home,
            new_thread_id=thread_id,
            effective_thread_id=thread_id or home.thread_id,
            profile_name=None,
        )
    )


def _discord_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter

    return DiscordAdapter.__new__(DiscordAdapter)


def _slack_adapter(*, chat_info=None, scope=None, extra=None, raise_lookup=False):
    from plugins.platforms.slack.adapter import SlackAdapter

    a = SlackAdapter.__new__(SlackAdapter)
    a.config = PlatformConfig(enabled=True, token="***", extra=dict(extra or {}))

    async def get_chat_info(chat_id):
        if raise_lookup:
            raise RuntimeError("slack api down")
        return chat_info or {"name": chat_id, "type": "group"}

    a.get_chat_info = get_chat_info
    a.scope_id_for_chat = lambda chat_id: scope
    from types import SimpleNamespace

    tspu = bool((extra or {}).get("thread_sessions_per_user", False))
    a._session_store = SimpleNamespace(
        resolve_session_scope=lambda source: (True, tspu)
    )
    return a


def test_discord_handoff_key_matches_organic_in_thread_key():
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"

    organic = build_session_key(
        SessionSource(
            platform=Platform.DISCORD,
            chat_id=thread_id,
            chat_type="thread",
            user_id="171164909650968576",
            thread_id=thread_id,
            parent_chat_id=parent_id,
        ),
        thread_sessions_per_user=False,
    )
    dest = _dest(_discord_adapter(), Platform.DISCORD, _home(Platform.DISCORD, parent_id), thread_id)
    handoff = build_session_key(dest, thread_sessions_per_user=False)

    assert handoff == organic
    assert handoff == f"agent:main:discord:thread:{thread_id}:{thread_id}"
    # The transport ref rides the source, keeping scope resolution
    # adapter-owned like any inbound source.
    assert getattr(dest, "_transport_adapter_ref", None) is not None


def test_slack_channel_handoff_key_matches_organic_thread_reply_key():
    channel_id = "C12345678"
    thread_ts = "1690000000.123456"

    organic = build_session_key(
        SessionSource(
            platform=Platform.SLACK,
            chat_id=channel_id,
            chat_type="group",
            user_id="U123456",
            thread_id=thread_ts,
            scope_id="T_TEAM",
        ),
        thread_sessions_per_user=False,
    )
    adapter = _slack_adapter(chat_info={"name": "general", "type": "group"}, scope="T_TEAM")
    dest = _dest(adapter, Platform.SLACK, _home(Platform.SLACK, channel_id), thread_ts)
    assert build_session_key(dest, thread_sessions_per_user=False) == organic
    # Legacy env-only home recorded no workspace: recovered from the adapter.
    assert dest.scope_id == "T_TEAM"


def test_slack_dm_home_handoff_keys_as_dm():
    """An IM home must key "dm" like organic IM replies — the adapter's
    reported chat type decides, never a hardcoded "group"."""
    adapter = _slack_adapter(chat_info={"name": "dm", "type": "dm"}, scope="T_TEAM")
    dest = _dest(adapter, Platform.SLACK, _home(Platform.SLACK, "D0DMDMDM"), "1690000000.123456")
    assert dest.chat_type == "dm"
    organic = SessionSource(
        platform=Platform.SLACK,
        chat_id="D0DMDMDM",
        chat_type="dm",
        user_id="U123",
        thread_id="1690000000.123456",
        scope_id="T_TEAM",
    )
    assert build_session_key(dest) == build_session_key(organic)


def test_slack_lookup_failure_falls_back_to_identity_prefix():
    """A transient get_chat_info failure must not change conversation
    identity: the Slack id prefix decides (D = IM, C = channel), whatever
    thread creation did — a DM home with a created thread stays "dm" like
    its organic replies, and a channel home without one stays "group"."""
    adapter = _slack_adapter(raise_lookup=True)
    dm_home = _home(Platform.SLACK, "D0DMDMDM", thread_id="1690000000.123456")
    assert _dest(adapter, Platform.SLACK, dm_home, None).chat_type == "dm"
    assert _dest(adapter, Platform.SLACK, dm_home, "1690000000.123456").chat_type == "dm"
    channel_home = _home(Platform.SLACK, "C12345678")
    assert _dest(adapter, Platform.SLACK, channel_home, None).chat_type == "group"
    assert _dest(adapter, Platform.SLACK, channel_home, "169.1").chat_type == "group"


def test_slack_per_user_threads_substitute_the_home_user():
    """Under thread_sessions_per_user the participant keys the session; the
    home channel's authenticated user replaces the placeholder, and its
    absence fails loudly rather than binding an unreachable transcript."""
    import pytest

    extra = {"group_sessions_per_user": True, "thread_sessions_per_user": True}
    adapter = _slack_adapter(chat_info={"name": "general", "type": "group"}, extra=extra)
    channel_id, thread_ts, user_id = "C12345678", "1690000000.123456", "U123456"

    dest = _dest(
        adapter, Platform.SLACK, _home(Platform.SLACK, channel_id, user_id=user_id), thread_ts
    )
    organic = build_session_key(
        SessionSource(
            platform=Platform.SLACK,
            chat_id=channel_id,
            chat_type="group",
            user_id=user_id,
            thread_id=thread_ts,
        ),
        thread_sessions_per_user=True,
    )
    assert build_session_key(dest, thread_sessions_per_user=True) == organic
    assert organic.endswith(f":{user_id}")

    with pytest.raises(RuntimeError, match="re-run /sethome"):
        _dest(adapter, Platform.SLACK, _home(Platform.SLACK, channel_id), thread_ts)


import pytest


@pytest.mark.parametrize(
    ("extra", "gateway_flag"),
    [
        # Adapter extra wins through the stamped transport ref.
        ({"group_sessions_per_user": True, "thread_sessions_per_user": True}, False),
        # Empty extra defers; the gateway-level flag decides.
        ({}, True),
    ],
    ids=["adapter-extra", "gateway-config"],
)
def test_slack_handoff_uses_resolved_thread_scope(tmp_path, extra, gateway_flag):
    """The REAL resolver decides participant substitution — reading this
    adapter's extra through the weakref the base hook stamps, else falling
    to the gateway config. One precedence contract, both directions."""
    from unittest.mock import patch

    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    adapter = _slack_adapter(chat_info={"name": "general", "type": "group"}, extra=extra)
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(
            sessions_dir=tmp_path,
            config=GatewayConfig(thread_sessions_per_user=gateway_flag),
        )
    store._db = None
    store._loaded = True
    adapter._session_store = store

    dest = _dest(
        adapter,
        Platform.SLACK,
        _home(Platform.SLACK, "C12345678", user_id="U123456"),
        "1690000000.123456",
    )
    assert dest.user_id == "U123456"


def test_discord_no_thread_shape_follows_conversation_identity():
    """Without an effective thread, identity decides: a guild home (workspace
    scope recorded) keys "group" like organic guild messages; a true DM home
    keys "dm". Thread-creation failure never demotes a guild conversation."""
    guild_home = _home(Platform.DISCORD, "111222333", scope_id="999888777")
    assert _dest(_discord_adapter(), Platform.DISCORD, guild_home, None).chat_type == "group"
    dm_home = _home(Platform.DISCORD, "444555666")
    assert _dest(_discord_adapter(), Platform.DISCORD, dm_home, None).chat_type == "dm"


def test_relay_fronted_discord_thread_keys_like_native():
    """A relay-fronted Discord home must key threads on the thread's own id
    like the native adapter — the relay applies the shared shape helpers for
    its logical lanes (its class never reaches sibling overrides)."""
    from gateway.relay.adapter import RelayAdapter

    a = RelayAdapter.__new__(RelayAdapter)
    a._transport = None
    dest = _dest(a, Platform.DISCORD, _home(Platform.DISCORD, "P1"), "T9")
    assert dest.chat_type == "thread"
    assert dest.chat_id == "T9"
    native = _dest(_discord_adapter(), Platform.DISCORD, _home(Platform.DISCORD, "P1"), "T9")
    assert build_session_key(dest) == build_session_key(native)


def test_relay_fronted_slack_thread_keys_like_native_without_chat_info():
    """A relay connector that cannot answer get_chat_info must still key a
    Slack thread handoff "group" (never "thread") and substitute the
    participant under per-user isolation — parity with the native adapter's
    shape default."""
    from types import SimpleNamespace

    from gateway.relay.adapter import RelayAdapter

    a = RelayAdapter.__new__(RelayAdapter)
    a._transport = None
    a._session_store = SimpleNamespace(
        resolve_session_scope=lambda source: (True, True)
    )
    channel_id, thread_ts, user_id = "C12345678", "1690000000.123456", "U123456"
    dest = _dest(
        a, Platform.SLACK, _home(Platform.SLACK, channel_id, user_id=user_id), thread_ts
    )
    assert dest.chat_type == "group"
    assert dest.user_id == user_id
    organic = build_session_key(
        SessionSource(
            platform=Platform.SLACK,
            chat_id=channel_id,
            chat_type="group",
            user_id=user_id,
            thread_id=thread_ts,
        ),
        thread_sessions_per_user=True,
    )
    assert build_session_key(dest, thread_sessions_per_user=True) == organic


def _telegram_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter.__new__(TelegramAdapter)


def _relay_adapter():
    from gateway.relay.adapter import RelayAdapter

    a = RelayAdapter.__new__(RelayAdapter)
    a._transport = None
    return a


@pytest.mark.parametrize(
    "adapter_factory", [_telegram_adapter, _relay_adapter], ids=["native", "relay"]
)
def test_telegram_handoff_shape(adapter_factory):
    """One shape table, both transports: private-chat topics key as DM topics
    with the real user id; forum/supergroup topics key "group"."""
    adapter = adapter_factory()
    private = _dest(adapter, Platform.TELEGRAM, _home(Platform.TELEGRAM, "123456789"), "77")
    assert private.chat_type == "dm"
    assert private.user_id == "123456789"
    group = _dest(adapter, Platform.TELEGRAM, _home(Platform.TELEGRAM, "-1001234567890"), "77")
    assert group.chat_type == "group"


def test_recorded_home_identity_beats_inference():
    """/sethome-recorded chat_type is canonical: an MPIM home (G… prefix,
    recorded "dm") keys dm without any lookup, and a Discord env home
    recorded "group" keys group with no workspace scope or lookup."""
    slack = _slack_adapter(raise_lookup=True)
    mpim = _dest(
        slack, Platform.SLACK, _home(Platform.SLACK, "G0MPIMPIM", chat_type="dm"), None
    )
    assert mpim.chat_type == "dm"

    discord_home = _home(Platform.DISCORD, "111222333", chat_type="group")
    dest = _dest(_discord_adapter(), Platform.DISCORD, discord_home, None)
    assert dest.chat_type == "group"


