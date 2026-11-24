"""What the language model is shown of earlier turns."""

from __future__ import annotations

from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ptychodus.agent.history import elide_earlier_images


def _turn(prompt: str) -> list[ModelMessage]:
    image = BinaryImage(data=b'\x89PNG', media_type='image/png')
    return [
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(parts=[ToolCallPart('render_probe', {}, tool_call_id=prompt)]),
        ModelRequest(parts=[ToolReturnPart('render_probe', [{'value_label': 'x'}, image], prompt)]),
        ModelResponse(parts=[TextPart('Done.')]),
    ]


def _render_content(message: ModelMessage) -> object:
    assert isinstance(message, ModelRequest)
    (part,) = message.parts
    assert isinstance(part, ToolReturnPart)
    return part.content


def test_images_before_the_latest_turn_become_notes() -> None:
    messages = [*_turn('first'), *_turn('second')]

    shown = elide_earlier_images(messages)

    earlier = _render_content(shown[2])
    assert isinstance(earlier, list)
    assert earlier[0] == {'value_label': 'x'}
    assert 'from an earlier turn' in earlier[1]
    latest = _render_content(shown[6])
    assert isinstance(latest, list) and isinstance(latest[1], BinaryImage)
    # The stored messages are untouched.
    stored = _render_content(messages[2])
    assert isinstance(stored, list) and isinstance(stored[1], BinaryImage)


def test_more_turns_can_keep_their_images() -> None:
    messages = [*_turn('first'), *_turn('second')]

    assert elide_earlier_images(messages, keep_turns=2) is messages
    only = _turn('only')
    assert elide_earlier_images(only) is only
