import logging
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import overload

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .models import ChatMessage

logger = logging.getLogger(__name__)


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at          TEXT    NOT NULL,
        model_messages_json TEXT    NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id),
        role            TEXT    NOT NULL,
        content         TEXT    NOT NULL,
        created_at      TEXT    NOT NULL
    )
    """,
    'CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)',
)


class ConversationObserver(ABC):
    @abstractmethod
    def handle_message_appended(self, message: ChatMessage, index: int) -> None:
        pass

    @abstractmethod
    def handle_conversation_cleared(self) -> None:
        pass


class ConversationRepository(Sequence[ChatMessage]):
    """SQLite-backed conversation store.

    Holds one logical "current conversation" per app launch, created lazily on the
    first append. Mirrors the current conversation's messages in memory so Qt list
    reads stay O(1); writes hit SQLite. The connection is shared across the Qt
    thread and the asyncio.run-driven terminal; we never call concurrently because
    asyncio.run blocks the caller.
    """

    def __init__(self, database_path: str) -> None:
        self._conn = sqlite3.connect(
            database_path or ':memory:',
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute('PRAGMA foreign_keys = ON')
        for statement in _SCHEMA:
            self._conn.execute(statement)

        self._current_conversation_id: int | None = None
        self._cached_messages: list[ChatMessage] = []
        self._observers: list[ConversationObserver] = []

    @overload
    def __getitem__(self, index: int) -> ChatMessage: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ChatMessage]: ...

    def __getitem__(self, index: int | slice) -> ChatMessage | Sequence[ChatMessage]:
        return self._cached_messages[index]

    def __len__(self) -> int:
        return len(self._cached_messages)

    def add_observer(self, observer: ConversationObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def append(self, message: ChatMessage) -> None:
        if self._current_conversation_id is None:
            cursor = self._conn.execute(
                'INSERT INTO conversations(created_at) VALUES (?)',
                (datetime.now().astimezone().isoformat(),),
            )
            self._current_conversation_id = int(cursor.lastrowid or 0)

        self._conn.execute(
            'INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)',
            (
                self._current_conversation_id,
                message.role.value,
                message.content,
                message.created_at.isoformat(),
            ),
        )

        index = len(self._cached_messages)
        self._cached_messages.append(message)

        for observer in self._observers:
            observer.handle_message_appended(message, index)

    def clear(self) -> None:
        if self._current_conversation_id is not None:
            self._conn.execute(
                'DELETE FROM messages WHERE conversation_id = ?',
                (self._current_conversation_id,),
            )
            self._conn.execute(
                'DELETE FROM conversations WHERE id = ?',
                (self._current_conversation_id,),
            )

        self._current_conversation_id = None
        self._cached_messages.clear()

        for observer in self._observers:
            observer.handle_conversation_cleared()

    def load_model_messages(self) -> list[ModelMessage]:
        if self._current_conversation_id is None:
            return []
        row = self._conn.execute(
            'SELECT model_messages_json FROM conversations WHERE id = ?',
            (self._current_conversation_id,),
        ).fetchone()
        if row is None:
            return []
        return list(ModelMessagesTypeAdapter.validate_json(row[0]))

    def save_model_messages(self, messages: list[ModelMessage]) -> None:
        if self._current_conversation_id is None:
            logger.warning('save_model_messages called with no current conversation; ignoring')
            return
        blob = ModelMessagesTypeAdapter.dump_json(messages).decode()
        self._conn.execute(
            'UPDATE conversations SET model_messages_json = ? WHERE id = ?',
            (blob, self._current_conversation_id),
        )

    def close(self) -> None:
        self._conn.close()
