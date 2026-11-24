"""Conversations kept per user, and the audit log of turns."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from ptychodus.agent.store import ConversationStore, TurnOutcome, TurnRecord


@pytest_asyncio.fixture
async def store() -> AsyncIterator[ConversationStore]:
    store = ConversationStore('sqlite+aiosqlite:///:memory:')
    await store.open()
    yield store
    await store.close()


async def test_history_round_trips_and_titles_once(store: ConversationStore) -> None:
    conversation = await store.create_conversation('jdoe')
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hello')]),
        ModelResponse(parts=[TextPart('Hi.')]),
    ]

    await store.save_messages('jdoe', conversation.id, messages, title='Hello')
    await store.save_messages('jdoe', conversation.id, messages, title='Ignored')

    loaded = await store.load_messages('jdoe', conversation.id)
    assert [type(message) for message in loaded] == [ModelRequest, ModelResponse]
    assert loaded[1].parts == [TextPart('Hi.')]
    assert (await store.get_conversation('jdoe', conversation.id)).title == 'Hello'


async def test_a_user_never_reaches_another_user_s_conversation(store: ConversationStore) -> None:
    conversation = await store.create_conversation('jdoe', 'Mine')

    assert await store.list_conversations('mallory') == []

    with pytest.raises(LookupError):
        await store.load_messages('mallory', conversation.id)

    with pytest.raises(LookupError):
        await store.save_messages('mallory', conversation.id, [])

    with pytest.raises(LookupError):
        await store.delete_conversation('mallory', conversation.id)

    assert [c.id for c in await store.list_conversations('jdoe')] == [conversation.id]


async def test_conversations_list_most_recently_updated_first(store: ConversationStore) -> None:
    older = await store.create_conversation('jdoe', 'Older')
    newer = await store.create_conversation('jdoe', 'Newer')
    await store.save_messages('jdoe', older.id, [])

    assert [c.id for c in await store.list_conversations('jdoe')] == [older.id, newer.id]


def _turn(user: str, started_at: datetime, tokens: int) -> TurnRecord:
    return TurnRecord(
        conversation_id='c',
        user=user,
        model_id='m',
        started_at=started_at,
        finished_at=started_at,
        outcome=TurnOutcome.COMPLETED,
        input_tokens=tokens,
        output_tokens=1,
        tool_calls=('list_product',),
    )


async def test_tokens_are_counted_per_user_since_a_time(store: ConversationStore) -> None:
    now = datetime.now(timezone.utc)
    await store.record_turn(_turn('jdoe', now - timedelta(days=2), 1000))
    await store.record_turn(_turn('jdoe', now - timedelta(hours=1), 10))
    await store.record_turn(_turn('mallory', now, 500))

    assert await store.count_tokens_since('jdoe', now - timedelta(days=1)) == 11
    assert await store.count_tokens_since('nobody', now - timedelta(days=1)) == 0
    turns = await store.list_turns('jdoe')
    assert [turn.input_tokens for turn in turns] == [1000, 10]
    assert turns[0].tool_calls == ('list_product',)
