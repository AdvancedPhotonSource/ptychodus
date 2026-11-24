"""Each user's private conversations, and the audit log of every turn."""

from __future__ import annotations

import enum
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage
from sqlalchemy import JSON, DateTime, Enum, Integer, String, Text, delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

__all__ = [
    'ConversationStore',
    'ConversationSummary',
    'TurnOutcome',
    'TurnRecord',
]


class _Base(DeclarativeBase):
    pass


class TurnOutcome(enum.StrEnum):
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    REFUSED = 'refused'


class _Conversation(_Base):
    __tablename__ = 'conversations'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user: Mapped[str] = mapped_column(String(256), index=True)
    title: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    messages_json: Mapped[str] = mapped_column(Text, default='[]')


class _Turn(_Base):
    __tablename__ = 'turns'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Not a foreign key: the audit log outlives a deleted conversation.
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    user: Mapped[str] = mapped_column(String(256), index=True)
    model_id: Mapped[str] = mapped_column(String(256))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[TurnOutcome] = mapped_column(Enum(TurnOutcome, native_enum=False))
    error: Mapped[str] = mapped_column(Text, default='')
    requests: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[list[str]] = mapped_column(JSON, default=list)


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TurnRecord:
    """One turn as the audit log keeps it."""

    conversation_id: str
    user: str
    model_id: str
    started_at: datetime
    finished_at: datetime
    outcome: TurnOutcome
    error: str = ''
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[str, ...] = ()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _summarize(row: _Conversation) -> ConversationSummary:
    return ConversationSummary(row.id, row.title, row.created_at, row.updated_at)


async def _find(session: AsyncSession, user: str, conversation_id: str) -> _Conversation:
    row = await session.scalar(
        select(_Conversation).where(_Conversation.user == user, _Conversation.id == conversation_id)
    )

    if row is None:
        raise LookupError(conversation_id)

    return row


class ConversationStore:
    """Conversations keyed by user: a user never reads, changes or lists another's.

    Each conversation keeps its pydantic-ai message history, which is what the next
    turn resumes from and what the web UI redraws.
    """

    def __init__(self, database_url: str) -> None:
        if database_url.startswith('sqlite+aiosqlite:///:memory:'):
            # One shared connection, or each would open its own empty database.
            self._engine: AsyncEngine = create_async_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={'check_same_thread': False},
            )
        else:
            self._engine = create_async_engine(database_url)

        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def open(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def create_conversation(self, user: str, title: str = '') -> ConversationSummary:
        now = _now()
        row = _Conversation(
            id=str(uuid.uuid4()),
            user=user,
            title=title[:256],
            created_at=now,
            updated_at=now,
            messages_json='[]',
        )

        async with self._sessions.begin() as session:
            session.add(row)

        return _summarize(row)

    async def list_conversations(self, user: str) -> list[ConversationSummary]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(_Conversation)
                .where(_Conversation.user == user)
                .order_by(_Conversation.updated_at.desc())
            )
            return [_summarize(row) for row in rows]

    async def get_conversation(self, user: str, conversation_id: str) -> ConversationSummary:
        """Raises LookupError unless `user` has a conversation `conversation_id`."""
        async with self._sessions() as session:
            return _summarize(await _find(session, user, conversation_id))

    async def load_messages(self, user: str, conversation_id: str) -> list[ModelMessage]:
        """Raises LookupError unless `user` has a conversation `conversation_id`."""
        async with self._sessions() as session:
            row = await _find(session, user, conversation_id)
            return ModelMessagesTypeAdapter.validate_json(row.messages_json)

    async def save_messages(
        self,
        user: str,
        conversation_id: str,
        messages: Sequence[ModelMessage],
        *,
        title: str = '',
    ) -> None:
        """Replace the conversation's history, and give an untitled one `title`.

        Raises LookupError unless `user` has a conversation `conversation_id`.
        """
        async with self._sessions.begin() as session:
            row = await _find(session, user, conversation_id)
            row.messages_json = ModelMessagesTypeAdapter.dump_json(list(messages)).decode()
            row.updated_at = _now()

            if not row.title:
                row.title = title[:256]

    async def delete_conversation(self, user: str, conversation_id: str) -> None:
        """Raises LookupError unless `user` has a conversation `conversation_id`."""
        async with self._sessions.begin() as session:
            result = await session.execute(
                delete(_Conversation).where(
                    _Conversation.user == user, _Conversation.id == conversation_id
                )
            )

            if result.rowcount == 0:  # type: ignore[attr-defined]
                raise LookupError(conversation_id)

    async def record_turn(self, record: TurnRecord) -> None:
        async with self._sessions.begin() as session:
            session.add(
                _Turn(
                    id=str(uuid.uuid4()),
                    conversation_id=record.conversation_id,
                    user=record.user,
                    model_id=record.model_id,
                    started_at=record.started_at,
                    finished_at=record.finished_at,
                    outcome=record.outcome,
                    error=record.error,
                    requests=record.requests,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    tool_calls=list(record.tool_calls),
                )
            )

    async def list_turns(self, user: str) -> list[TurnRecord]:
        """The audit log of `user`'s turns, oldest first."""
        async with self._sessions() as session:
            rows = await session.scalars(
                select(_Turn).where(_Turn.user == user).order_by(_Turn.started_at)
            )
            return [
                TurnRecord(
                    conversation_id=row.conversation_id,
                    user=row.user,
                    model_id=row.model_id,
                    started_at=row.started_at,
                    finished_at=row.finished_at,
                    outcome=row.outcome,
                    error=row.error,
                    requests=row.requests,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    tool_calls=tuple(row.tool_calls),
                )
                for row in rows
            ]

    async def count_tokens_since(self, user: str, since: datetime) -> int:
        """The input and output tokens `user`'s turns started at or after `since` spent."""
        async with self._sessions() as session:
            total = await session.scalar(
                select(func.coalesce(func.sum(_Turn.input_tokens + _Turn.output_tokens), 0)).where(
                    _Turn.user == user, _Turn.started_at >= since
                )
            )
            return int(total or 0)
