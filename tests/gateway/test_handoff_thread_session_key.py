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


def _home(platform, chat_id, scope_id=None, user_id=None, thread_id=None):
    return HomeChannel(
        platform=platform,
        chat_id=str(chat_id),
        name="home",
        thread_id=thread_id,
        user_id=user_id,
        scope_id=scope_id,
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


def test_slack_lookup_failure_preserves_no_thread_dm_type():
    """A transient get_chat_info failure must not flip a no-thread DM-fallback
    destination to "group" — the pre-computed intent survives; only a
    thread-created destination normalizes to "group"."""
    adapter = _slack_adapter(raise_lookup=True)
    home = _home(Platform.SLACK, "D0DMDMDM", thread_id="1690000000.123456")
    no_thread = _dest(adapter, Platform.SLACK, home, None)
    assert no_thread.chat_type == "dm"
    threaded = _dest(adapter, Platform.SLACK, home, "1690000000.123456")
    assert threaded.chat_type == "group"


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


def test_slack_per_user_threads_honor_the_gateway_level_flag():
    """A deployment setting thread_sessions_per_user only at gateway level
    (platform extra carries no key) must still substitute the participant —
    the gate resolves extra-then-gateway-config like _create_adapter's
    seeding."""
    from types import SimpleNamespace

    from gateway.config import GatewayConfig

    adapter = _slack_adapter(chat_info={"name": "general", "type": "group"})
    adapter._session_store = SimpleNamespace(
        resolve_session_scope=lambda source: (True, True)
    )
    dest = _dest(
        adapter,
        Platform.SLACK,
        _home(Platform.SLACK, "C12345678", user_id="U123456"),
        "1690000000.123456",
    )
    assert dest.user_id == "U123456"


def test_telegram_private_chat_topic_keys_as_dm_topic():
    """A handoff-created topic in a private chat must use the DM-topic source
    shape (real user id == chat id) so the user's next message shares it."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    private_chat = "123456789"  # positive id = private chat
    dest = _dest(adapter, Platform.TELEGRAM, _home(Platform.TELEGRAM, private_chat), "77")
    assert dest.chat_type == "dm"
    assert dest.user_id == private_chat

    group_chat = "-1001234567890"
    dest = _dest(adapter, Platform.TELEGRAM, _home(Platform.TELEGRAM, group_chat), "77")
    # Organic supergroup topic replies arrive as "group" with the topic in
    # thread_id — a generic "thread" key would strand the transcript.
    assert dest.chat_type == "group"
