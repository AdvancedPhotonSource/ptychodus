"""What the language model is shown of a conversation's history."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

__all__ = [
    'elide_earlier_images',
]


# Any, because a tool return's content is whatever the tool returned, and the subclasses
# of ToolReturnPart each narrow it to a type of their own.
def _elide_binary(content: Any) -> Any:
    if isinstance(content, BinaryContent):
        return (
            f'[{content.media_type} from an earlier turn, not shown again; '
            'call the tool again to see it]'
        )

    if isinstance(content, list):
        return [_elide_binary(item) for item in content]

    return content


def _elide_request(message: ModelRequest) -> ModelRequest:
    parts = [
        replace(part, content=_elide_binary(part.content))
        if isinstance(part, ToolReturnPart)
        else part
        for part in message.parts
    ]
    return replace(message, parts=parts)


def elide_earlier_images(
    messages: list[ModelMessage], *, keep_turns: int = 1
) -> list[ModelMessage]:
    """`messages` with each image a tool returned before the last `keep_turns` turns
    replaced by a note, so a long conversation does not resend every image it rendered.

    A turn begins at each request carrying a user prompt. The messages are not modified;
    the elided ones are copies.
    """
    turn_starts = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]

    if len(turn_starts) <= keep_turns:
        return messages

    cutoff = turn_starts[-keep_turns] if keep_turns > 0 else len(messages)
    return [
        _elide_request(message) if isinstance(message, ModelRequest) and index < cutoff else message
        for index, message in enumerate(messages)
    ]
