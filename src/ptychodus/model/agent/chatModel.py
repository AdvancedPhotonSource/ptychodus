from typing import Any
import logging

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .argo import Argo

logger = logging.getLogger(__name__)


class ArgoChatModel(BaseChatModel):
    argo: Argo

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates a chat result from a prompt"""
        prompt: list[str] = []
        system: str | None = None

        for message in messages:
            if message.type == 'human':
                human = message.content
                prompt.append(human)
            elif message.type == 'system':
                if system is None:
                    system = message.content
                else:
                    logger.warning(f'IGNORED {message}')
            else:
                logger.warning(f'IGNORED {message}')

        logger.debug(f'SYSTEM: {system}')
        logger.debug(f'HUMAN: {prompt}')

        response = self.argo.invoke(prompt, system or '', stop or [])
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Uniquely identifies the type of the model. Used for logging."""
        return f'argo-{self.argo.model}'
