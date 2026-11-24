"""The agent service, on a scripted language model, against the real webapp's MCP server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any

import anyio
import pytest
from ag_ui.core import BaseEvent, RunStartedEvent
from fastapi.testclient import TestClient
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from ptychodus.agent.app import _audited, _TurnAudit
from ptychodus.agent.events import RENDERED_TEXT_EVENT
from ptychodus.agent.rendering import utf16_length
from ptychodus.agent.store import TurnOutcome, TurnRecord

from agent_fixtures import ScriptedLanguageModel, Webapp, headers, read_events, run_input

_TEXT = [
    'The probe is **Gaussian**, with $E = h\\nu$.\n',
    '\n$$\n\\int |P|^2 \\, dA\n$$\n',
    '\nIts width is 𝐱',
    ' pixels.',
]


def _scripted(webapp: Webapp) -> ScriptedLanguageModel:
    return ScriptedLanguageModel(
        [('list_product', {}), ('render_probe', {'uuid': str(webapp.product_uuid)})], _TEXT
    )


def _new_conversation(client: TestClient, user: str = 'jdoe') -> str:
    response = client.post('/agent/v1/conversations', headers=headers(user), json={})
    assert response.status_code == 201, response.text
    conversation_id: str = response.json()['id']
    return conversation_id


def _turn(
    client: TestClient, conversation_id: str, prompt: str, *, user: str = 'jdoe', **kwargs: Any
) -> list[dict[str, Any]]:
    response = client.post(
        f'/agent/v1/conversations/{conversation_id}/turns',
        headers=headers(user),
        json=run_input(conversation_id, prompt, **kwargs),
    )
    assert response.status_code == 200, response.text
    return read_events(response.text)


def _turns(client: TestClient, user: str) -> list[TurnRecord]:
    store = client.app.state.agent_service.store  # type: ignore[attr-defined]
    assert client.portal is not None
    turns: list[TurnRecord] = client.portal.call(store.list_turns, user)
    return turns


def test_a_turn_calls_the_webapp_and_streams_rendered_text(
    webapp: Webapp,
    agent_client: Callable[..., TestClient],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client)

    with caplog.at_level(logging.INFO):
        events = _turn(client, conversation_id, 'Describe the probe.')

    types = [event['type'] for event in events]
    assert types[0] == 'RUN_STARTED'
    assert types[-1] == 'RUN_FINISHED'
    assert [event['toolCallName'] for event in events if event['type'] == 'TOOL_CALL_START'] == [
        'list_product',
        'render_probe',
    ]

    # The render's scales reach the language model; its PNG does not cross SSE.
    render_result = [event for event in events if event['type'] == 'TOOL_CALL_RESULT'][-1]
    assert 'image-omitted' in render_result['content']
    assert 'value_label' in render_result['content']
    assert 'iVBOR' not in render_result['content']

    deltas = ''.join(e['delta'] for e in events if e['type'] == 'TEXT_MESSAGE_CONTENT')
    assert deltas == ''.join(_TEXT)
    rendered = [e for e in events if e['type'] == 'CUSTOM' and e['name'] == RENDERED_TEXT_EVENT]
    (message_id,) = {e['messageId'] for e in events if e['type'] == 'TEXT_MESSAGE_CONTENT'}
    assert {event['value']['messageId'] for event in rendered} == {message_id}
    end = next(
        index
        for index, event in enumerate(events)
        if event['type'] == 'TEXT_MESSAGE_END' and event['messageId'] == message_id
    )
    partial = [event for event in rendered if not event['value']['final']]
    final = [event for event in rendered if event['value']['final']]
    assert partial, 'no block was rendered before the message ended'
    assert all(events.index(event) < end for event in partial)
    assert len(final) == 1
    assert events.index(final[0]) > end
    assert '<math' in final[0]['value']['html']
    assert '<strong>Gaussian</strong>' in final[0]['value']['html']
    assert final[0]['value']['sourceLength'] == utf16_length(deltas)

    messages = [record.getMessage() for record in caplog.records]
    assert 'tool render_probe called by agent for jdoe' in messages

    (turn,) = _turns(client, 'jdoe')
    assert turn.outcome == TurnOutcome.COMPLETED
    assert turn.conversation_id == conversation_id
    assert turn.model_id == 'scripted'
    assert turn.requests == 3
    assert turn.tool_calls == ('list_product', 'render_probe')


def test_a_conversation_keeps_its_history_and_shows_earlier_images_as_notes(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    scripted = _scripted(webapp)
    client = agent_client(scripted.model())
    conversation_id = _new_conversation(client)
    _turn(client, conversation_id, 'Describe the probe.')

    detail = client.get(f'/agent/v1/conversations/{conversation_id}', headers=headers()).json()
    assert detail['conversation']['title'] == 'Describe the probe.'
    items = detail['items']
    assert [item['kind'] for item in items] == ['user', 'tool_call', 'tool_call', 'assistant']
    assert items[0]['text'] == 'Describe the probe.'
    assert items[2] == {
        'kind': 'tool_call',
        'toolCallId': items[2]['toolCallId'],
        'name': 'render_probe',
        'args': {'uuid': str(webapp.product_uuid)},
        'ok': True,
    }
    assert '<math' in items[3]['html']

    first_turn_requests = len(scripted.requests)
    _turn(client, conversation_id, 'And again?', message_id='u2')

    def render_returns(request_index: int) -> list[object]:
        return [
            part.content
            for message in scripted.requests[request_index]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'render_probe'
        ]

    # Within its turn the image is shown; on the next turn, only a note stands for it.
    (shown,) = render_returns(first_turn_requests - 1)
    assert isinstance(shown, list) and any(isinstance(item, BinaryContent) for item in shown)
    (elided,) = render_returns(first_turn_requests)
    assert isinstance(elided, list)
    assert not any(isinstance(item, BinaryContent) for item in elided)
    assert any('from an earlier turn' in str(item) for item in elided)

    listed = client.get('/agent/v1/conversations', headers=headers()).json()
    assert [conversation['id'] for conversation in listed] == [conversation_id]


def test_a_conversation_is_private_to_its_user(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client, 'jdoe')
    path = f'/agent/v1/conversations/{conversation_id}'
    mallory = headers('mallory')

    assert client.get(path, headers=mallory).status_code == 404
    assert client.delete(path, headers=mallory).status_code == 404
    response = client.post(
        f'{path}/turns', headers=mallory, json=run_input(conversation_id, 'Show me.')
    )
    assert response.status_code == 404
    assert client.get('/agent/v1/conversations', headers=mallory).json() == []
    assert client.get(path, headers=headers('jdoe')).status_code == 200


def test_a_request_without_the_user_header_is_refused_unless_a_development_user_is_set(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    assert client.get('/agent/v1/session').status_code == 401
    assert client.get('/agent/v1/session', headers=headers('a\tb')).status_code == 400
    assert client.get('/agent/v1/health').status_code == 200

    developer = agent_client(_scripted(webapp).model(), development_user='dev')
    assert developer.get('/agent/v1/session').json() == {
        'user': 'dev',
        'provider': 'injected',
        'modelId': 'scripted',
    }
    assert developer.get('/agent/v1/session', headers=headers()).json()['user'] == 'jdoe'


def test_a_turn_is_refused_while_another_runs_in_its_conversation(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client)
    client.app.state.agent_service.active_conversations.add(conversation_id)  # type: ignore[attr-defined]

    response = client.post(
        f'/agent/v1/conversations/{conversation_id}/turns',
        headers=headers(),
        json=run_input(conversation_id, 'Hello'),
    )

    assert response.status_code == 409
    path = f'/agent/v1/conversations/{conversation_id}'
    assert client.delete(path, headers=headers()).status_code == 409


def test_a_user_past_the_daily_token_limit_is_refused(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model(), user_daily_tokens_limit=1)
    conversation_id = _new_conversation(client)
    _turn(client, conversation_id, 'Describe the probe.')

    response = client.post(
        f'/agent/v1/conversations/{conversation_id}/turns',
        headers=headers(),
        json=run_input(conversation_id, 'Again', message_id='u2'),
    )

    assert response.status_code == 429
    assert [turn.outcome for turn in _turns(client, 'jdoe')] == [
        TurnOutcome.COMPLETED,
        TurnOutcome.REFUSED,
    ]
    assert _turns(client, 'someone-else') == []


def test_a_turn_past_its_request_limit_fails_and_leaves_the_conversation_unchanged(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model(), turn_request_limit=1)
    conversation_id = _new_conversation(client)

    events = _turn(client, conversation_id, 'Describe the probe.')

    assert events[-1]['type'] == 'RUN_ERROR'
    detail = client.get(f'/agent/v1/conversations/{conversation_id}', headers=headers()).json()
    assert detail['items'] == []
    (turn,) = _turns(client, 'jdoe')
    assert turn.outcome == TurnOutcome.FAILED
    assert 'request_limit' in turn.error


def test_a_turn_whose_stream_is_closed_early_is_recorded_as_cancelled(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client)
    service = client.app.state.agent_service  # type: ignore[attr-defined]
    service.active_conversations.add(conversation_id)

    async def endless() -> AsyncIterator[BaseEvent]:
        yield RunStartedEvent(thread_id=conversation_id, run_id='r')
        await anyio.sleep_forever()
        yield RunStartedEvent(thread_id=conversation_id, run_id='never')

    async def read_one_then_close() -> None:
        audit = _TurnAudit('jdoe', conversation_id, 'scripted', datetime.now(timezone.utc))
        stream = _audited(endless(), service, audit)
        await anext(stream)
        await stream.aclose()

    assert client.portal is not None
    client.portal.call(read_one_then_close)

    (turn,) = _turns(client, 'jdoe')
    assert turn.outcome == TurnOutcome.CANCELLED
    assert conversation_id not in service.active_conversations


def test_conversations_are_changed_only_by_json_requests(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    form = client.post(
        '/agent/v1/conversations', headers={**headers(), 'content-type': 'text/plain'}, content='{}'
    )
    assert form.status_code == 415

    conversation_id = _new_conversation(client)
    response = client.post(
        f'/agent/v1/conversations/{conversation_id}/turns',
        headers={**headers(), 'content-type': 'text/plain'},
        content='{}',
    )
    assert response.status_code == 415


def test_a_turn_must_name_its_conversation_as_the_thread(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client)
    path = f'/agent/v1/conversations/{conversation_id}/turns'

    other = client.post(path, headers=headers(), json=run_input('another-thread', 'Hi'))
    assert other.status_code == 400
    malformed = client.post(path, headers=headers(), json={'threadId': conversation_id})
    assert malformed.status_code == 422


def test_deleting_a_conversation_keeps_its_turns_in_the_audit_log(
    webapp: Webapp, agent_client: Callable[..., TestClient]
) -> None:
    client = agent_client(_scripted(webapp).model())
    conversation_id = _new_conversation(client)
    _turn(client, conversation_id, 'Describe the probe.')
    path = f'/agent/v1/conversations/{conversation_id}'

    assert client.delete(path, headers=headers()).status_code == 204
    assert client.get(path, headers=headers()).status_code == 404
    assert [turn.conversation_id for turn in _turns(client, 'jdoe')] == [conversation_id]
