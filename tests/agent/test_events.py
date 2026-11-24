"""The AG-UI stream's rendered-text events, driven hook by hook."""

from __future__ import annotations

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    RunAgentInput,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from pydantic_ai.messages import TextPart, TextPartDelta

from ptychodus.agent.events import RENDERED_TEXT_EVENT, RenderingEventStream


def _stream() -> RenderingEventStream:
    return RenderingEventStream(RunAgentInput(thread_id='t', run_id='r', messages=[], tools=[]))


async def test_adjacent_text_parts_render_as_one_message() -> None:
    stream = _stream()
    events: list[BaseEvent] = []

    events += [e async for e in stream.handle_text_start(TextPart('First paragraph.\n\n'))]
    events += [e async for e in stream.handle_text_end(TextPart(''), followed_by_text=True)]
    events += [e async for e in stream.handle_text_start(TextPart('Second'), follows_text=True)]
    events += [e async for e in stream.handle_text_delta(TextPartDelta(' paragraph.'))]
    events += [e async for e in stream.handle_text_end(TextPart(''), followed_by_text=False)]

    assert sum(isinstance(e, TextMessageStartEvent) for e in events) == 1
    assert sum(isinstance(e, TextMessageEndEvent) for e in events) == 1
    message_ids = {e.message_id for e in events if isinstance(e, TextMessageContentEvent)}
    assert len(message_ids) == 1
    rendered = [e for e in events if isinstance(e, CustomEvent)]
    assert all(e.name == RENDERED_TEXT_EVENT for e in rendered)
    assert [e.value['final'] for e in rendered] == [False, True]
    assert rendered[0].value['html'] == '<p>First paragraph.</p>\n'
    assert rendered[-1].value['html'] == '<p>First paragraph.</p>\n<p>Second paragraph.</p>\n'
    assert {e.value['messageId'] for e in rendered} == message_ids
    assert isinstance(events[-2], TextMessageEndEvent)
