"""The agent service: AG-UI turns over SSE, and each user's private conversations."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial

import anyio
import pydantic_ai
from ag_ui.core import BaseEvent, EventType, RunErrorEvent, ToolCallStartEvent, UserMessage
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import UsageLimits

from ptychodus.api.plugins import PluginRegistry

from .config import Settings, get_settings
from .endpoint import create_chat_model, resolve_endpoint
from .events import RenderingAdapter
from .loop import AgentDeps, build_agent
from .rendering import MessageRenderer
from .schemas import ConversationDetail, ConversationInfo, SessionInfo
from .store import ConversationStore, ConversationSummary, TurnOutcome, TurnRecord
from .transcript import describe_transcript

logger = logging.getLogger(__name__)

__all__ = [
    'create_app',
]

_PREFIX = '/agent/v1'


@dataclass
class _AgentService:
    settings: Settings
    agent: Agent[AgentDeps, str]
    store: ConversationStore
    provider_name: str
    model_id: str
    renderer: MessageRenderer = field(default_factory=MessageRenderer)
    active_conversations: set[str] = field(default_factory=set)

    def usage_limits(self) -> UsageLimits:
        return UsageLimits(
            request_limit=self.settings.turn_request_limit,
            tool_calls_limit=self.settings.turn_tool_calls_limit,
            total_tokens_limit=self.settings.turn_total_tokens_limit,
        )


@dataclass
class _TurnAudit:
    """What one turn has done so far, for its audit record."""

    user: str
    conversation_id: str
    model_id: str
    started_at: datetime
    outcome: TurnOutcome = TurnOutcome.CANCELLED
    error: str = ''
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)

    def record(self) -> TurnRecord:
        return TurnRecord(
            conversation_id=self.conversation_id,
            user=self.user,
            model_id=self.model_id,
            started_at=self.started_at,
            finished_at=datetime.now(timezone.utc),
            outcome=self.outcome,
            error=self.error,
            requests=self.requests,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            tool_calls=tuple(self.tool_calls),
        )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )


def _service(request: Request) -> _AgentService:
    service: _AgentService = request.app.state.agent_service
    return service


def _current_user(request: Request) -> str:
    """The user the proxy signed in, or the development user when there is no header.

    The header is believed as it arrives: only the proxy may be able to reach the service.
    """
    settings = _service(request).settings
    user = request.headers.get(settings.user_header, '').strip()

    if not user:
        if settings.development_user:
            return settings.development_user

        raise HTTPException(401, f'No {settings.user_header} header; sign in through the proxy.')

    if len(user) > 256 or not user.isprintable():
        raise HTTPException(400, f'The {settings.user_header} header is not a user name.')

    return user


def _describe(summary: ConversationSummary) -> ConversationInfo:
    return ConversationInfo(
        id=summary.id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
    )


def _require_json(request: Request) -> None:
    # A cross-site form can POST the CORS-safelisted types without a preflight.
    content_type = request.headers.get('content-type', '').split(';')[0].strip().lower()

    if content_type != 'application/json':
        raise HTTPException(415, 'Send application/json.')


def _new_prompt(messages: list[object]) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage) and isinstance(message.content, str):
            return message.content

    return ''


async def _complete_turn(
    service: _AgentService,
    audit: _TurnAudit,
    title: str,
    result: AgentRunResult[str],
) -> None:
    usage = result.usage
    audit.requests = usage.requests
    audit.input_tokens = usage.input_tokens
    audit.output_tokens = usage.output_tokens
    await service.store.save_messages(
        audit.user, audit.conversation_id, result.all_messages(), title=title
    )


def _observe(audit: _TurnAudit, event: BaseEvent) -> None:
    if isinstance(event, ToolCallStartEvent):
        audit.tool_calls.append(event.tool_call_name)
    elif isinstance(event, RunErrorEvent):
        audit.outcome = TurnOutcome.FAILED
        audit.error = event.message
    elif event.type == EventType.RUN_FINISHED and audit.outcome != TurnOutcome.FAILED:
        audit.outcome = TurnOutcome.COMPLETED


async def _audited(
    stream: AsyncIterator[BaseEvent], service: _AgentService, audit: _TurnAudit
) -> AsyncGenerator[BaseEvent, None]:
    """`stream`, recording the turn in the audit log however it ends.

    A turn whose stream is closed before the run finishes (the client disconnected or
    pressed Stop) is recorded as cancelled.
    """
    try:
        async for event in stream:
            _observe(audit, event)
            yield event
    finally:
        # Shielded, so a cancelled request still closes the run and leaves its record.
        with anyio.CancelScope(shield=True):
            if isinstance(stream, AsyncGenerator):
                await stream.aclose()

            service.active_conversations.discard(audit.conversation_id)
            record = audit.record()
            await service.store.record_turn(record)
            logger.info(
                'turn in conversation %s for %s %s: %d requests, %d+%d tokens, tools %s%s',
                record.conversation_id,
                record.user,
                record.outcome,
                record.requests,
                record.input_tokens,
                record.output_tokens,
                ', '.join(record.tool_calls) or 'none',
                f' ({record.error})' if record.error else '',
            )


def _resolve_language_model(settings: Settings) -> tuple[Model, str, str]:
    registry = PluginRegistry.load_plugins()
    provider = registry.language_model_providers.find_plugin(settings.provider)

    if provider is None:
        raise RuntimeError(
            f'No language-model provider "{settings.provider}"; choose one of '
            f'{registry.language_model_providers.stringify_plugin_names()}.'
        )

    endpoint = resolve_endpoint(settings, provider)
    return create_chat_model(endpoint), provider.display_name, endpoint.model_id


def create_app(settings: Settings | None = None, *, language_model: Model | None = None) -> FastAPI:
    """The agent service.

    `language_model` replaces the configured provider's, for testing.
    """
    settings = settings or get_settings()
    _configure_logging(settings.log_level)
    # The service owns its output; pydantic-ai's first-run banner would go to its log.
    pydantic_ai.BANNER_ENABLED = False

    if language_model is None:
        language_model, provider_name, model_id = _resolve_language_model(settings)
    else:
        provider_name, model_id = 'injected', language_model.model_name

    if settings.development_user:
        logger.warning(
            'Requests without %s are taken to be from %r. For development only.',
            settings.user_header,
            settings.development_user,
        )

    if settings.database_url.startswith('sqlite+aiosqlite:///:memory:'):
        logger.warning('Conversations are kept in memory and vanish on a restart.')

    store = ConversationStore(settings.database_url)
    service = _AgentService(
        settings=settings,
        agent=build_agent(language_model, settings),
        store=store,
        provider_name=provider_name,
        model_id=model_id,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await store.open()

        try:
            yield
        finally:
            await store.close()

    app = FastAPI(title='ptychodus-agent', lifespan=lifespan)
    app.state.agent_service = service

    @app.get(f'{_PREFIX}/health')
    async def get_health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get(f'{_PREFIX}/session', response_model=SessionInfo, response_model_by_alias=True)
    async def get_session(user: str = Depends(_current_user)) -> SessionInfo:
        return SessionInfo(user=user, provider=service.provider_name, model_id=service.model_id)

    @app.get(
        f'{_PREFIX}/conversations',
        response_model=list[ConversationInfo],
        response_model_by_alias=True,
    )
    async def list_conversations(user: str = Depends(_current_user)) -> list[ConversationInfo]:
        return [_describe(summary) for summary in await store.list_conversations(user)]

    @app.post(
        f'{_PREFIX}/conversations',
        status_code=201,
        response_model=ConversationInfo,
        response_model_by_alias=True,
        dependencies=[Depends(_require_json)],
    )
    async def create_conversation(user: str = Depends(_current_user)) -> ConversationInfo:
        return _describe(await store.create_conversation(user))

    @app.get(
        f'{_PREFIX}/conversations/{{conversation_id}}',
        response_model=ConversationDetail,
        response_model_by_alias=True,
    )
    async def get_conversation(
        conversation_id: str, user: str = Depends(_current_user)
    ) -> ConversationDetail:
        try:
            summary = await store.get_conversation(user, conversation_id)
            messages = await store.load_messages(user, conversation_id)
        except LookupError:
            raise HTTPException(404, 'No such conversation.') from None

        return ConversationDetail(
            conversation=_describe(summary),
            items=describe_transcript(messages, service.renderer),
        )

    @app.delete(f'{_PREFIX}/conversations/{{conversation_id}}', status_code=204)
    async def delete_conversation(
        conversation_id: str, user: str = Depends(_current_user)
    ) -> Response:
        if conversation_id in service.active_conversations:
            raise HTTPException(409, 'A turn is running in this conversation.')

        try:
            await store.delete_conversation(user, conversation_id)
        except LookupError:
            raise HTTPException(404, 'No such conversation.') from None

        return Response(status_code=204)

    @app.post(f'{_PREFIX}/conversations/{{conversation_id}}/turns')
    async def post_turn(
        conversation_id: str, request: Request, user: str = Depends(_current_user)
    ) -> Response:
        try:
            stored = await store.load_messages(user, conversation_id)
        except LookupError:
            raise HTTPException(404, 'No such conversation.') from None

        try:
            adapter = await RenderingAdapter.from_request(
                request, agent=service.agent, manage_message_history='server'
            )
        except ValidationError as exc:
            raise HTTPException(422, f'Not an AG-UI run input: {exc}') from exc

        if adapter.run_input.thread_id != conversation_id:
            raise HTTPException(400, 'The thread id is not this conversation.')

        started_at = datetime.now(timezone.utc)
        limit = settings.user_daily_tokens_limit

        if limit is not None:
            spent = await store.count_tokens_since(user, started_at - timedelta(days=1))

            if spent >= limit:
                await store.record_turn(
                    TurnRecord(
                        conversation_id=conversation_id,
                        user=user,
                        model_id=service.model_id,
                        started_at=started_at,
                        finished_at=started_at,
                        outcome=TurnOutcome.REFUSED,
                        error=f'daily token limit of {limit} reached',
                    )
                )
                raise HTTPException(
                    429, f'You have used your {limit} tokens for the last 24 hours.'
                )

        # No await separates the check from the claim, so two turns cannot both pass it.
        if conversation_id in service.active_conversations:
            raise HTTPException(409, 'A turn is already running in this conversation.')

        service.active_conversations.add(conversation_id)
        audit = _TurnAudit(user, conversation_id, service.model_id, started_at)
        stream = adapter.run_stream(
            deps=AgentDeps(user=user, conversation_id=conversation_id),
            message_history=stored,
            usage_limits=service.usage_limits(),
            on_complete=partial(
                _complete_turn, service, audit, _new_prompt(list(adapter.run_input.messages))
            ),
        )
        return adapter.streaming_response(_audited(stream, service, audit))

    return app
