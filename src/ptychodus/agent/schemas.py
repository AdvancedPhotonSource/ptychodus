"""The agent service's JSON responses, outside the AG-UI stream."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

__all__ = [
    'AssistantItem',
    'ConversationDetail',
    'ConversationInfo',
    'SessionInfo',
    'ToolCallItem',
    'TranscriptItem',
    'UserItem',
]


class _Schema(BaseModel):
    # camelCase on the wire, as the AG-UI events the web UI reads beside these are.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class SessionInfo(_Schema):
    user: str
    provider: str
    model_id: str


class ConversationInfo(_Schema):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class UserItem(_Schema):
    kind: Literal['user'] = 'user'
    text: str


class AssistantItem(_Schema):
    kind: Literal['assistant'] = 'assistant'
    html: str
    """Sanitized HTML, rendered from the language model's Markdown."""


class ToolCallItem(_Schema):
    kind: Literal['tool_call'] = 'tool_call'
    tool_call_id: str
    name: str
    args: dict[str, Any]
    ok: bool | None = None
    """Whether the tool answered; None if no result was recorded."""


TranscriptItem = UserItem | AssistantItem | ToolCallItem


class ConversationDetail(_Schema):
    conversation: ConversationInfo
    items: list[TranscriptItem] = Field(default_factory=list)
