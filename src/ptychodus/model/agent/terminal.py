import asyncio
import logging
from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

from .endpoint import create_chat_model, create_model_settings, get_mcp_server_url
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

        try:
            mcp_server_url = get_mcp_server_url(self._settings)
            agent = Agent(
                create_chat_model(
                    self._settings.base_url.get_value(), self._settings.model.get_value()
                ),
                instructions=self._settings.instructions.get_value(),
                model_settings=create_model_settings(self._settings),
                toolsets=[MCPToolset(mcp_server_url)] if mcp_server_url else [],
            )
            prior = self._repository.load_model_messages()

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
