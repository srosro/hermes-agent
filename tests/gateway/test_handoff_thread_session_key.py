"""Regression: CLI→Discord handoff must key a thread destination on the
thread's OWN id, matching how the platform adapter keys organic in-thread
messages.

Bug: the handoff built its destination ``SessionSource`` with
``chat_id = home.chat_id`` (the PARENT channel) while thread destinations use
``chat_type="thread"`` and ``thread_id = <thread>``. The Discord adapter,
however, builds organic in-thread messages with ``chat_id = <thread>`` (the
thread's own id). ``build_session_key`` therefore produced two different keys:

    handoff:  agent:main:discord:thread:{parent}:{thread}
    organic:  agent:main:discord:thread:{thread}:{thread}

So the next real user reply in the handoff thread resolved to a DIFFERENT
session_key and spawned a fresh session instead of continuing the handed-off
one (observed: a stray auto-titled session + a session_search fallback because
the new session had no prior context).

The fix is Discord-specific: Slack and Telegram adapters key organic thread
messages with ``chat_id = parent_channel``, so the parent channel is correct
for those platforms and the guard must NOT apply to them.
"""

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _organic_discord_thread_key(thread_id: str, parent_id: str, user_id: str) -> str:
    """Key the Discord adapter produces for a message typed inside a thread.

    Mirrors plugins/platforms/discord/adapter.py _handle_message: chat_id is
    the thread's own id, chat_type is "thread", thread_id is the thread id,
    parent_chat_id is the parent channel.
    """
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=str(thread_id),
        chat_type="thread",
        user_id=user_id,
        thread_id=str(thread_id),
        parent_chat_id=str(parent_id),
    )
    return build_session_key(source, thread_sessions_per_user=False)


def _organic_slack_thread_key(channel_id: str, thread_ts: str, user_id: str) -> str:
    """Key the Slack adapter produces for a message in a thread.

    Mirrors plugins/platforms/slack/adapter.py: chat_id is the parent channel,
    chat_type is "group", thread_id is the thread timestamp.
    """
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=str(channel_id),
        chat_type="group",
        user_id=user_id,
        thread_id=str(thread_ts),
    )
    return build_session_key(source, thread_sessions_per_user=False)


def _handoff_key(
    platform: Platform,
    home_chat_id: str,
    thread_id: str,
    scope_id: str | None = None,
    *,
    chat_info: dict | None = None,
    adapter_scope_id: str | None = None,
) -> str:
    """Key the REAL handoff construction produces.

    Drives GatewayRunner._build_handoff_dest_source (no mirror) with a stub
    adapter: ``chat_info`` is what the platform adapter would report for the
    home chat, ``adapter_scope_id`` its workspace scope for legacy homes.
    """
    import asyncio

    from gateway.config import HomeChannel
    from gateway.run import GatewayRunner

    class _StubAdapter:
        async def get_chat_info(self, chat_id):
            return chat_info or {"name": chat_id, "type": "group"}

        def scope_id_for_chat(self, chat_id):
            return adapter_scope_id

    runner = object.__new__(GatewayRunner)
    home = HomeChannel(
        platform=platform,
        chat_id=str(home_chat_id),
        name="home",
        scope_id=scope_id,
    )
    dest_source = asyncio.run(
        runner._build_handoff_dest_source(
            platform=platform,
            home=home,
            new_thread_id=str(thread_id),
            effective_thread_id=str(thread_id),
            profile_name=None,
            adapter=_StubAdapter(),
        )
    )
    return build_session_key(dest_source, thread_sessions_per_user=False)


def test_slack_dm_home_handoff_keys_as_dm():
    """An IM home must key "dm" like organic IM replies — the adapter's
    reported chat type decides, not a hardcoded "group"."""
    import asyncio

    from gateway.config import HomeChannel
    from gateway.run import GatewayRunner

    class _DmAdapter:
        async def get_chat_info(self, chat_id):
            return {"name": chat_id, "type": "dm"}

        def scope_id_for_chat(self, chat_id):
            return "T_TEAM"

    runner = object.__new__(GatewayRunner)
    home = HomeChannel(platform=Platform.SLACK, chat_id="D0DMDMDM", name="dm home")
    dest = asyncio.run(
        runner._build_handoff_dest_source(
            platform=Platform.SLACK,
            home=home,
            new_thread_id="1690000000.123456",
            effective_thread_id="1690000000.123456",
            profile_name=None,
            adapter=_DmAdapter(),
        )
    )
    assert dest.chat_type == "dm"
    # Legacy env-only home recorded no workspace: recovered from the adapter.
    assert dest.scope_id == "T_TEAM"
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
    destination to "group" — the pre-computed chat type survives; only a
    thread-created destination normalizes to "group"."""
    import asyncio

    from gateway.config import HomeChannel
    from gateway.run import GatewayRunner

    class _FailingAdapter:
        async def get_chat_info(self, chat_id):
            raise RuntimeError("slack api down")

        def scope_id_for_chat(self, chat_id):
            return None

    runner = object.__new__(GatewayRunner)
    home = HomeChannel(platform=Platform.SLACK, chat_id="D0DMDMDM", name="dm home")

    async def _build(new_thread_id):
        return await runner._build_handoff_dest_source(
            platform=Platform.SLACK,
            home=home,
            new_thread_id=new_thread_id,
            effective_thread_id=new_thread_id or "1690000000.123456",
            profile_name=None,
            adapter=_FailingAdapter(),
        )

    no_thread = asyncio.run(_build(None))
    assert no_thread.chat_type == "dm"
    threaded = asyncio.run(_build("1690000000.123456"))
    assert threaded.chat_type == "group"


def test_discord_handoff_key_matches_organic_in_thread_key():
    """For Discord, the handoff key must be byte-identical to the organic
    in-thread key — otherwise a reply in the handoff thread spawns a new session."""
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"
    user_id = "171164909650968576"

    organic = _organic_discord_thread_key(thread_id, parent_id, user_id)
    handoff = _handoff_key(Platform.DISCORD, parent_id, thread_id)

    assert handoff == organic, (
        f"handoff key {handoff!r} != organic in-thread key {organic!r}; "
        "a reply in the handoff thread would spawn a new session"
    )
    assert handoff == f"agent:main:discord:thread:{thread_id}:{thread_id}"


def test_discord_handoff_key_does_not_use_parent_channel():
    """The pre-fix bug: keying on the parent channel. Guard against regression."""
    parent_id = "1523581766923845724"
    thread_id = "1523590238595846166"

    handoff = _handoff_key(Platform.DISCORD, parent_id, thread_id)
    buggy = f"agent:main:discord:thread:{parent_id}:{thread_id}"

    assert handoff != buggy, "handoff regressed to keying on the parent channel"


def test_slack_handoff_key_matches_organic_thread_reply_key():
    """A Slack handoff destination must derive the exact key an organic reply
    in that thread uses: chat_id = parent channel, chat_type = "group"
    (adapter-native — a "thread" chat type binds a transcript no reply can
    reach), and the same workspace scope_id."""
    channel_id = "C12345678"
    thread_ts = "1690000000.123456"
    user_id = "U123456"

    organic = _organic_slack_thread_key(channel_id, thread_ts, user_id)
    handoff = _handoff_key(Platform.SLACK, channel_id, thread_ts)
    assert handoff == organic, (
        f"handoff key {handoff!r} != organic thread-reply key {organic!r}; "
        "a reply in the handed-off thread would spawn a new session"
    )
    # scope_id lands in the key (workspace-scoped session identity).
    scoped = _handoff_key(Platform.SLACK, channel_id, thread_ts, scope_id="T_TEAM")
    assert "T_TEAM" in scoped.split(":"), scoped
    assert scoped != handoff


def test_slack_handoff_key_matches_organic_under_per_user_threads():
    """Under thread_sessions_per_user the organic key carries the
    participant, and _process_handoff substitutes the home channel's
    authenticated user for the system:handoff placeholder — the keys must
    still be byte-identical."""
    channel_id = "C12345678"
    thread_ts = "1690000000.123456"
    user_id = "U123456"

    organic_source = SessionSource(
        platform=Platform.SLACK,
        chat_id=channel_id,
        chat_type="group",
        user_id=user_id,
        thread_id=thread_ts,
    )
    organic = build_session_key(organic_source, thread_sessions_per_user=True)
    handoff_source = SessionSource(
        platform=Platform.SLACK,
        chat_id=channel_id,
        chat_type="group",
        user_id=user_id,  # home.user_id substitution in _process_handoff
        user_name="Handoff",
        thread_id=thread_ts,
    )
    handoff = build_session_key(handoff_source, thread_sessions_per_user=True)
    assert handoff == organic
    assert organic.endswith(f":{user_id}")
