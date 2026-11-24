"""The AG-UI event stream, with server-rendered HTML beside the raw text deltas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

from ag_ui.core import BaseEvent, CustomEvent
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolResultEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream

from .loop import AgentDeps
from .rendering import BlockRenderer, MessageRenderer

__all__ = [
    'RENDERED_TEXT_EVENT',
    'RenderingAdapter',
    'RenderingEventStream',
]

RENDERED_TEXT_EVENT = 'ptychodus.rendered_text'
"""The CUSTOM event carrying a text message's sanitized HTML: ``messageId``, ``html``,
``sourceLength`` (the UTF-16 length of the raw text it renders) and ``final``."""


def _omit_binary(content: object) -> object:
    if isinstance(content, BinaryContent):
        return {'kind': 'image-omitted', 'media_type': content.media_type}

    if isinstance(content, list):
        return [_omit_binary(item) for item in content]

    return content


def _rendered_text_event(message_id: str, rendered: tuple[str, int], *, final: bool) -> CustomEvent:
    html, source_length = rendered
    return CustomEvent(
        name=RENDERED_TEXT_EVENT,
        value={
            'messageId': message_id,
            'html': html,
            'sourceLength': source_length,
            'final': final,
        },
    )


@dataclass
class RenderingEventStream(AGUIEventStream[AgentDeps, str]):
    """AG-UI events with each text message rendered as it completes, block by block.

    After a delta completes a block, a `RENDERED_TEXT_EVENT` carries the HTML of the
    message's complete blocks; after the message ends, one carries all of it. A tool
    result is sent without its binary content, which the web UI fetches itself.
    """

    renderer: MessageRenderer = field(default_factory=MessageRenderer)
    _blocks: dict[str, BlockRenderer] = field(default_factory=dict, init=False)

    def _feed(self, delta: str) -> CustomEvent | None:
        message_id = self.message_id
        blocks = self._blocks.setdefault(message_id, BlockRenderer(self.renderer))
        rendered = blocks.feed(delta)
        return None if rendered is None else _rendered_text_event(message_id, rendered, final=False)

    async def handle_text_start(
        self, part: TextPart, follows_text: bool = False
    ) -> AsyncIterator[BaseEvent]:
        async for event in super().handle_text_start(part, follows_text):
            yield event

        rendered = self._feed(part.content)

        if rendered is not None:
            yield rendered

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        async for event in super().handle_text_delta(delta):
            yield event

        rendered = self._feed(delta.content_delta)

        if rendered is not None:
            yield rendered

    async def handle_text_end(
        self, part: TextPart, followed_by_text: bool = False
    ) -> AsyncIterator[BaseEvent]:
        async for event in super().handle_text_end(part, followed_by_text):
            yield event

        if followed_by_text:
            return

        blocks = self._blocks.pop(self.message_id, None)

        # The empty message opened before a tool call has nothing to render.
        if blocks is not None and blocks.text:
            yield _rendered_text_event(self.message_id, blocks.finish(), final=True)

    async def handle_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> AsyncIterator[BaseEvent]:
        part = event.part

        if isinstance(part, ToolReturnPart):
            event = replace(event, part=replace(part, content=_omit_binary(part.content)))

        async for result in super().handle_function_tool_result(event):
            yield result


@dataclass
class RenderingAdapter(AGUIAdapter[AgentDeps, str]):
    """The AG-UI adapter whose event stream is a `RenderingEventStream`."""

    def build_event_stream(self) -> RenderingEventStream:
        return RenderingEventStream(
            self.run_input,
            accept=self.accept,
            ag_ui_version=self.ag_ui_version,
            include_usage=self.include_usage,
            manage_system_prompt=self.manage_system_prompt,
            preserve_file_data=self.preserve_file_data,
        )
