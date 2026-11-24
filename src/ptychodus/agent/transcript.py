"""A stored conversation as the web UI redraws it."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .rendering import MessageRenderer
from .schemas import AssistantItem, ToolCallItem, TranscriptItem, UserItem

__all__ = [
    'describe_transcript',
    'prompt_text',
]


def prompt_text(part: UserPromptPart) -> str:
    """The text of a user prompt, leaving out anything that is not text."""
    if isinstance(part.content, str):
        return part.content

    return '\n'.join(item for item in part.content if isinstance(item, str))


def describe_transcript(
    messages: Sequence[ModelMessage], renderer: MessageRenderer
) -> list[TranscriptItem]:
    """The user prompts, assistant text (rendered as sanitized HTML) and tool calls of
    `messages`, in order. Tool results are reduced to whether each call was answered."""
    items: list[TranscriptItem] = []
    tool_calls: dict[str, ToolCallItem] = {}

    for message in messages:
        if isinstance(message, ModelResponse):
            text = ''.join(part.content for part in message.parts if isinstance(part, TextPart))

            if text.strip():
                items.append(AssistantItem(html=renderer.render(text)))

            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    item = ToolCallItem(
                        tool_call_id=part.tool_call_id,
                        name=part.tool_name,
                        args=part.args_as_dict(),
                    )
                    tool_calls[part.tool_call_id] = item
                    items.append(item)
        elif isinstance(message, ModelRequest):
            for request_part in message.parts:
                if isinstance(request_part, UserPromptPart):
                    items.append(UserItem(text=prompt_text(request_part)))
                elif isinstance(request_part, ToolReturnPart | RetryPromptPart):
                    call = tool_calls.get(request_part.tool_call_id)

                    if call is not None:
                        call.ok = isinstance(request_part, ToolReturnPart)

    return items
