"""The chat drawer in a real browser, with the agent service stubbed.

The webapp test server serves no `/agent/*`, so each test answers the panel's requests
itself, as the agent service would, while the panel draws render calls through the real
webapp. The drawer stays closed until opened, so no other page calls the agent service.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest
from playwright.sync_api import Page, Route, expect

from webapp_ui_fixtures import ServedPage

pytestmark = [
    pytest.mark.browser,
    pytest.mark.filterwarnings('ignore:Session-scoped fixtures defined as instance methods'),
]

OpenUI = Callable[[], ServedPage]

_CONVERSATION = {
    'id': 'c1',
    'title': '',
    'createdAt': '2026-11-24T00:00:00Z',
    'updatedAt': '2026-11-24T00:00:00Z',
}


def _sse(events: list[dict[str, Any]]) -> str:
    return ''.join(f'data: {json.dumps(event)}\n\n' for event in events)


def _turn_events(product: UUID) -> list[dict[str, Any]]:
    raw = 'The probe is **Gaussian**, $x^2$, width 𝐱.'
    html = (
        '<p>The probe is <strong>Gaussian</strong>, <span class="math inline">'
        '<math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math></span>, width 𝐱.</p>'
    )
    return [
        {'type': 'RUN_STARTED', 'threadId': 'c1', 'runId': 'r1'},
        {'type': 'TOOL_CALL_START', 'toolCallId': 't1', 'toolCallName': 'render_probe'},
        {'type': 'TOOL_CALL_ARGS', 'toolCallId': 't1', 'delta': json.dumps({'uuid': str(product)})},
        {'type': 'TOOL_CALL_END', 'toolCallId': 't1'},
        {'type': 'TOOL_CALL_RESULT', 'toolCallId': 't1', 'messageId': 'x', 'content': '[]'},
        {'type': 'TEXT_MESSAGE_START', 'messageId': 'm1', 'role': 'assistant'},
        {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': 'm1', 'delta': raw[:20]},
        {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': 'm1', 'delta': raw[20:]},
        {'type': 'TEXT_MESSAGE_END', 'messageId': 'm1'},
        {
            'type': 'CUSTOM',
            'name': 'ptychodus.rendered_text',
            # 𝐱 is two UTF-16 code units, which the panel's offsets count.
            'value': {'messageId': 'm1', 'html': html, 'sourceLength': len(raw) + 1, 'final': True},
        },
        {'type': 'RUN_FINISHED', 'threadId': 'c1', 'runId': 'r1'},
    ]


def _stub_agent(page: Page, turn: str) -> list[dict[str, Any]]:
    """Answers the panel's agent requests; returns the turn inputs it was sent."""
    inputs: list[dict[str, Any]] = []

    def handle(route: Route) -> None:
        request = route.request
        path = request.url.split('/agent/v1', 1)[1]

        if path == '/session':
            body = {'user': 'jdoe', 'provider': 'Test', 'modelId': 'scripted'}
            route.fulfill(json=body)
        elif path == '/conversations' and request.method == 'GET':
            route.fulfill(json=[])
        elif path == '/conversations' and request.method == 'POST':
            route.fulfill(status=201, json=_CONVERSATION)
        elif path == '/conversations/c1/turns':
            turn_input = request.post_data_json
            assert isinstance(turn_input, dict)
            inputs.append(turn_input)
            route.fulfill(content_type='text/event-stream', body=turn)
        else:
            route.fulfill(status=404, json={'detail': 'Not Found'})

    page.route('**/agent/v1/**', handle)
    return inputs


def test_a_turn_streams_into_the_drawer_with_rendered_text_and_the_render(
    open_ui: OpenUI, seed_product: Callable[..., UUID]
) -> None:
    product = seed_product(analyzable=True)
    ui = open_ui()
    page = ui.page
    inputs = _stub_agent(page, _sse(_turn_events(product)))
    ui.goto('#product')

    page.get_by_role('button', name='Agent', exact=True).click()
    expect(page.locator('#chat')).to_be_visible()
    expect(page.locator('.chat-model')).to_have_text('scripted · jdoe')

    page.get_by_label('Message').fill('Describe the probe.')
    page.get_by_label('Message').press('Enter')

    assistant = page.locator('.chat-assistant')
    expect(assistant.locator('strong')).to_have_text('Gaussian')
    expect(assistant.locator('math')).to_have_count(1)
    # The rendered HTML covers all the raw text, so none is left showing.
    expect(assistant.locator('.chat-raw')).to_have_text('')
    expect(page.locator('.chat-tool')).to_have_text('render_probe')
    expect(page.locator('.chat-figure img')).to_be_visible()
    link = page.locator('.chat-figure a')
    expect(link).to_have_attribute('href', f'#product/{product}/probe')
    expect(page.locator('.chat-user')).to_have_text('Describe the probe.')

    (sent,) = inputs
    assert sent['threadId'] == 'c1'
    assert [message['content'] for message in sent['messages']] == ['Describe the probe.']
    assert sent['messages'][0]['role'] == 'user'

    # A deep link from the conversation opens the viewer and leaves the drawer open.
    link.click()
    expect(page.locator('#chat')).to_be_visible()
    assert page.url.endswith(f'#product/{product}/probe')
    ui.assert_clean()


def test_the_drawer_says_when_the_agent_service_is_unavailable(open_ui: OpenUI) -> None:
    ui = open_ui()
    page = ui.page
    page.route('**/agent/v1/**', lambda route: route.fulfill(status=502, body='Bad Gateway'))
    ui.goto('#product')

    page.get_by_role('button', name='Agent', exact=True).click()

    expect(page.locator('.chat-notice')).to_contain_text('The agent service is unavailable')
    expect(page.get_by_label('Message')).to_be_disabled()
    ui.assert_clean(allow_status={502})
