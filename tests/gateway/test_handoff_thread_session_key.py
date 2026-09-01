"""CLI-handoff destinations must key exactly like each adapter's ORGANIC
replies, or the next real message in the handed-off conversation derives a
different session key and spawns a fresh session.

Construction is one authoritative builder (``build_handoff_dest_source``)
with thin per-adapter shape hooks; these tests drive the real hooks with
stubbed IO — no mirrors of production logic. The parity contract is ONE
matrix: ``build_session_key(destination) == build_session_key(organic)``
across native and relay-fronted platforms; the focused tests below it pin
the failure policies and legacy fallbacks.
"""

import asyncio

import pytest

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


def _wire_store(adapter, *, tspu=False):
    from types import SimpleNamespace

    adapter._session_store = SimpleNamespace(
        resolve_session_scope=lambda source: (True, tspu)
    )
    return adapter


def _discord_adapter(*, chat_info=None, raise_lookup=False):
    from plugins.platforms.discord.adapter import DiscordAdapter

    a = DiscordAdapter.__new__(DiscordAdapter)

    async def get_chat_info(chat_id):
        if raise_lookup or chat_info is None:
            raise RuntimeError("discord api down")
        return chat_info

    a.get_chat_info = get_chat_info
    return _wire_store(a)


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
    return _wire_store(a, tspu=bool((extra or {}).get("thread_sessions_per_user", False)))


def _telegram_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return _wire_store(TelegramAdapter.__new__(TelegramAdapter))


def _signal_adapter():
    from gateway.platforms.signal import SignalAdapter

    return _wire_store(SignalAdapter.__new__(SignalAdapter))


def _relay_adapter(*, tspu=False):
    from gateway.relay.adapter import RelayAdapter

    a = RelayAdapter.__new__(RelayAdapter)
    a._transport = None
    return _wire_store(a, tspu=tspu)


# ── The parity contract: one matrix ─────────────────────────────────────────
#
# Each row: an adapter, a home, the thread outcome, the organic source the
# next real reply would produce, and the key flags — the destination and the
# organic source must derive byte-identical session keys.

_DISCORD_THREAD = "1523590238595846166"
_DISCORD_PARENT = "1523581766923845724"
_SLACK_TS = "1690000000.123456"

PARITY_CASES = [
    pytest.param(
        _discord_adapter,
        Platform.DISCORD,
        dict(chat_id=_DISCORD_PARENT),
        _DISCORD_THREAD,
        dict(
            chat_id=_DISCORD_THREAD,
            chat_type="thread",
            user_id="171164909650968576",
            thread_id=_DISCORD_THREAD,
            parent_chat_id=_DISCORD_PARENT,
        ),
        dict(thread_sessions_per_user=False),
        id="discord-thread-native",
    ),
    pytest.param(
        _relay_adapter,
        Platform.DISCORD,
        dict(chat_id=_DISCORD_PARENT),
        _DISCORD_THREAD,
        dict(
            chat_id=_DISCORD_THREAD,
            chat_type="thread",
            user_id="171164909650968576",
            thread_id=_DISCORD_THREAD,
            parent_chat_id=_DISCORD_PARENT,
        ),
        dict(thread_sessions_per_user=False),
        id="discord-thread-relay",
    ),
    pytest.param(
        lambda: _slack_adapter(chat_info={"name": "general", "type": "group"}, scope="T_TEAM"),
        Platform.SLACK,
        dict(chat_id="C12345678"),
        _SLACK_TS,
        dict(
            chat_id="C12345678",
            chat_type="group",
            user_id="U123456",
            thread_id=_SLACK_TS,
            scope_id="T_TEAM",
        ),
        dict(thread_sessions_per_user=False),
        id="slack-channel-thread-native",
    ),
    pytest.param(
        lambda: _slack_adapter(chat_info={"name": "dm", "type": "dm"}, scope="T_TEAM"),
        Platform.SLACK,
        dict(chat_id="D0DMDMDM"),
        _SLACK_TS,
        dict(
            chat_id="D0DMDMDM",
            chat_type="dm",
            user_id="U123",
            thread_id=_SLACK_TS,
            scope_id="T_TEAM",
        ),
        dict(),
        id="slack-dm-home-native",
    ),
    pytest.param(
        lambda: _relay_adapter(tspu=True),
        Platform.SLACK,
        dict(chat_id="C12345678", user_id="U123456"),
        _SLACK_TS,
        dict(
            chat_id="C12345678",
            chat_type="group",
            user_id="U123456",
            thread_id=_SLACK_TS,
        ),
        dict(thread_sessions_per_user=True),
        id="slack-per-user-thread-relay-no-chat-info",
    ),
    pytest.param(
        _telegram_adapter,
        Platform.TELEGRAM,
        dict(chat_id="123456789"),
        "77",
        dict(chat_id="123456789", chat_type="dm", user_id="123456789", thread_id="77"),
        dict(),
        id="telegram-private-topic-native",
    ),
    pytest.param(
        _relay_adapter,
        Platform.TELEGRAM,
        dict(chat_id="123456789"),
        "77",
        dict(chat_id="123456789", chat_type="dm", user_id="123456789", thread_id="77"),
        dict(),
        id="telegram-private-topic-relay",
    ),
    pytest.param(
        _telegram_adapter,
        Platform.TELEGRAM,
        dict(chat_id="-1001234567890", user_id="208214988"),
        "77",
        dict(
            chat_id="-1001234567890",
            chat_type="group",
            user_id="208214988",
            thread_id="77",
        ),
        dict(thread_sessions_per_user=False),
        id="telegram-supergroup-topic-native",
    ),
    pytest.param(
        _relay_adapter,
        Platform.TELEGRAM,
        dict(chat_id="-1001234567890", user_id="208214988"),
        "77",
        dict(
            chat_id="-1001234567890",
            chat_type="group",
            user_id="208214988",
            thread_id="77",
        ),
        dict(thread_sessions_per_user=False),
        id="telegram-supergroup-topic-relay",
    ),
    pytest.param(
        _signal_adapter,
        Platform.SIGNAL,
        dict(chat_id="grp.abc", user_id="+1555", chat_type="group"),
        None,
        dict(chat_id="grp.abc", chat_type="group", user_id="+1555"),
        dict(),
        id="signal-generic-group-no-override",
    ),
]


@pytest.mark.parametrize(
    ("adapter_factory", "platform", "home_kwargs", "thread_id", "organic_kwargs", "key_kwargs"),
    PARITY_CASES,
)
def test_handoff_key_matches_organic_reply_key(
    adapter_factory, platform, home_kwargs, thread_id, organic_kwargs, key_kwargs
):
    adapter = adapter_factory()
    home = _home(platform, **home_kwargs)
    dest = _dest(adapter, platform, home, thread_id)
    organic = SessionSource(platform=platform, **organic_kwargs)
    assert build_session_key(dest, **key_kwargs) == build_session_key(
        organic, **key_kwargs
    )
    # The transport ref rides every destination, keeping scope resolution
    # adapter-owned like any inbound source.
    assert getattr(dest, "_transport_adapter_ref", None) is not None


# ── Failure policies and legacy fallbacks ───────────────────────────────────


def test_slack_lookup_failure_falls_back_to_identity_prefix():
    """A transient get_chat_info failure must not change conversation
    identity: the Slack id prefix decides (D = IM, C = channel), whatever
    thread creation did."""
    adapter = _slack_adapter(raise_lookup=True)
    dm_home = _home(Platform.SLACK, "D0DMDMDM", thread_id=_SLACK_TS)
    assert _dest(adapter, Platform.SLACK, dm_home, None).chat_type == "dm"
    assert _dest(adapter, Platform.SLACK, dm_home, _SLACK_TS).chat_type == "dm"
    # Non-thread groups key per participant under the default scope, so the
    # home's recorded user rides along (generalized participant application).
    channel_home = _home(Platform.SLACK, "C12345678", user_id="U1")
    no_thread_group = _dest(adapter, Platform.SLACK, channel_home, None)
    assert no_thread_group.chat_type == "group"
    assert no_thread_group.user_id == "U1"
    assert _dest(adapter, Platform.SLACK, channel_home, "169.1").chat_type == "group"


def test_participant_policy_triangle():
    """Substitute when the home recorded a user; land-with-warning on a
    legacy non-thread home without one (run.py's thread-creation-failure
    contract); raise only for threaded per-user handoffs."""
    slack = _slack_adapter(chat_info={"name": "general", "type": "group"})
    landed = _dest(slack, Platform.SLACK, _home(Platform.SLACK, "C12345678"), None)
    assert landed.user_id == "system:handoff"

    tspu_extra = {"group_sessions_per_user": True, "thread_sessions_per_user": True}
    strict = _slack_adapter(chat_info={"name": "general", "type": "group"}, extra=tspu_extra)
    substituted = _dest(
        strict, Platform.SLACK, _home(Platform.SLACK, "C12345678", user_id="U123456"), _SLACK_TS
    )
    assert substituted.user_id == "U123456"
    with pytest.raises(RuntimeError, match="re-run /sethome"):
        _dest(strict, Platform.SLACK, _home(Platform.SLACK, "C12345678"), _SLACK_TS)


def test_participant_substitution_restores_user_id_alt():
    """build_session_key PREFERS user_id_alt (Signal UUID, Feishu union_id…),
    so the substitution must restore both recorded participant fields."""
    adapter = _signal_adapter()
    home = _home(Platform.SIGNAL, "grp.abc", user_id="+1555", chat_type="group")
    home.user_id_alt = "uuid-1234"
    dest = _dest(adapter, Platform.SIGNAL, home, None)
    assert dest.user_id == "+1555"
    assert dest.user_id_alt == "uuid-1234"
    organic = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="grp.abc",
        chat_type="group",
        user_id="+1555",
        user_id_alt="uuid-1234",
    )
    assert build_session_key(dest) == build_session_key(organic)


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
    if not extra:
        assert adapter.config.extra == {}  # the transport-ref branch must defer
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
        _SLACK_TS,
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


def test_discord_legacy_home_promotes_via_live_lookup():
    """A scope-less, non-recorded (legacy env-configured) Discord home
    resolves once through the live channel: a non-DM answer keys "group";
    a failed lookup keeps the "dm" default."""
    home = _home(Platform.DISCORD, "111222333")
    promoted = _dest(
        _discord_adapter(chat_info={"name": "general", "type": "channel"}),
        Platform.DISCORD,
        home,
        None,
    )
    assert promoted.chat_type == "group"
    kept = _dest(_discord_adapter(raise_lookup=True), Platform.DISCORD, home, None)
    assert kept.chat_type == "dm"


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
