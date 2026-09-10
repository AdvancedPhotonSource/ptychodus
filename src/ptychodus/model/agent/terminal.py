import asyncio
import logging
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .endpoint import get_api_key_for_base_url
from .models import ChatMessage, ChatRole
from .repository import ConversationRepository
from .settings import AgentSettings

logger = logging.getLogger(__name__)


class ChatTerminal:
    def __init__(self, settings: AgentSettings, repository: ConversationRepository) -> None:
        self._settings = settings
        self._repository = repository

    def clear_conversation(self) -> None:
        self._repository.clear()

    def send_message(self, content: str) -> None:
        if not content.strip():
            return

        self._repository.append(
            ChatMessage(
                role=ChatRole.USER,
                content=content,
                created_at=datetime.now().astimezone(),
            )
        )

        # NOTE: pydantic-ai uses Authorization: Bearer headers via the OpenAI SDK.
        # The Argo OpenAPI spec instead declares ?authorization= as a query param;
        # if the live endpoint rejects the Bearer header, pass a custom
        # httpx.AsyncClient(params={'authorization': key}) to OpenAIProvider.
        base_url = self._settings.base_url.get_value()
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=get_api_key_for_base_url(base_url),
        )
        model = OpenAIChatModel(self._settings.model.get_value(), provider=provider)
        model_settings = ModelSettings(
            temperature=self._settings.temperature.get_value(),
            top_p=self._settings.top_p.get_value(),
            max_tokens=self._settings.max_tokens.get_value(),
        )
        # TODO(mcp-followup): when settings.mcp_server_url is non-empty, pass
        # toolsets=[MCPServerStreamableHTTP(url=...)] so the agent can call
        # ptychodus_store tools (see src/ptychodus_store/mcp_server.py).
        agent = Agent(
            model,
            system_prompt=self._settings.system_prompt.get_value(),
            model_settings=model_settings,
        )

        prior = self._repository.load_model_messages()

        try:
            # asyncio.run blocks the Qt event loop for the duration of the call,
            # same as the previous requests.post. Move to a QThread if it stops
            # being acceptable for this developer-mode panel.
            result = asyncio.run(agent.run(content, message_history=prior))
        except Exception as exc:
            logger.exception('Chat request failed')
            self._repository.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=f'[error] {exc}',
                    created_at=datetime.now().astimezone(),
                )
            )
            return

        self._repository.save_model_messages(list(result.all_messages()))
        self._repository.append(
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content=str(result.output),
                created_at=datetime.now().astimezone(),
            )
        )
