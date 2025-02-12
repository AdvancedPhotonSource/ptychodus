from collections.abc import Mapping
from typing import Any, cast
import json
import logging
import requests

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.output_parsers.openai_tools import make_invalid_tool_call, parse_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult

from .settings import ArgoSettings

__all__ = ['ChatArgo']

logger = logging.getLogger(__name__)


# This code is copied from langchain's OpenAI implementation and autoformatted
#   URL: https://github.com/langchain-ai/langchain.git
#   PATH: libs/partners/openai/lanchain_openai/chat_models/base.py
# BEGIN COPY


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        'type': 'function',
        'id': tool_call['id'],
        'function': {
            'name': tool_call['name'],
            'arguments': json.dumps(tool_call['args']),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        'type': 'function',
        'id': invalid_tool_call['id'],
        'function': {
            'name': invalid_tool_call['name'],
            'arguments': invalid_tool_call['args'],
        },
    }


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get('role')
    name = _dict.get('name')
    id_ = _dict.get('id')
    if role == 'user':
        return HumanMessage(content=_dict.get('content', ''), id=id_, name=name)
    elif role == 'assistant':
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get('content', '') or ''
        additional_kwargs: dict = {}
        if function_call := _dict.get('function_call'):
            additional_kwargs['function_call'] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get('tool_calls'):
            additional_kwargs['tool_calls'] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
        if audio := _dict.get('audio'):
            additional_kwargs['audio'] = audio
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role in ('system', 'developer'):
        if role == 'developer':
            additional_kwargs = {'__openai_role__': role}
        else:
            additional_kwargs = {}
        return SystemMessage(
            content=_dict.get('content', ''),
            name=name,
            id=id_,
            additional_kwargs=additional_kwargs,
        )
    elif role == 'function':
        return FunctionMessage(
            content=_dict.get('content', ''), name=cast(str, _dict.get('name')), id=id_
        )
    elif role == 'tool':
        additional_kwargs = {}
        if 'name' in _dict:
            additional_kwargs['name'] = _dict['name']
        return ToolMessage(
            content=_dict.get('content', ''),
            tool_call_id=cast(str, _dict.get('tool_call_id')),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get('content', ''), role=role, id=id_)  # type: ignore[arg-type]


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if isinstance(block, dict) and 'type' in block and block['type'] == 'tool_use':
                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: dict[str, Any] = {'content': _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get('name')) is not None:
        message_dict['name'] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict['role'] = message.role
    elif isinstance(message, HumanMessage):
        message_dict['role'] = 'user'
    elif isinstance(message, AIMessage):
        message_dict['role'] = 'assistant'
        if 'function_call' in message.additional_kwargs:
            message_dict['function_call'] = message.additional_kwargs['function_call']
        if message.tool_calls or message.invalid_tool_calls:
            message_dict['tool_calls'] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [_lc_invalid_tool_call_to_openai_tool_call(tc) for tc in message.invalid_tool_calls]
        elif 'tool_calls' in message.additional_kwargs:
            message_dict['tool_calls'] = message.additional_kwargs['tool_calls']
            tool_call_supported_props = {'id', 'type', 'function'}
            message_dict['tool_calls'] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict['tool_calls']
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if 'function_call' in message_dict or 'tool_calls' in message_dict:
            message_dict['content'] = message_dict['content'] or None

        if 'audio' in message.additional_kwargs:
            # openai doesn't support passing the data back - only the id
            # https://platform.openai.com/docs/guides/audio/multi-turn-conversations
            raw_audio = message.additional_kwargs['audio']
            audio = (
                {'id': message.additional_kwargs['audio']['id']} if 'id' in raw_audio else raw_audio
            )
            message_dict['audio'] = audio
    elif isinstance(message, SystemMessage):
        message_dict['role'] = message.additional_kwargs.get('__openai_role__', 'system')
    elif isinstance(message, FunctionMessage):
        message_dict['role'] = 'function'
    elif isinstance(message, ToolMessage):
        message_dict['role'] = 'tool'
        message_dict['tool_call_id'] = message.tool_call_id

        supported_props = {'content', 'role', 'tool_call_id'}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f'Got unknown type {message}')
    return message_dict


# END COPY


class ChatArgo(BaseChatModel):
    settings: ArgoSettings

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates a chat result from a prompt"""
        url = self.settings.chatEndpointURL.getValue()
        payload = {
            'user': self.settings.user.getValue(),
            'model': self.settings.chatModel.getValue(),
            'messages': [_convert_message_to_dict(m) for m in messages],
            'stop': stop or [],
            'temperature': self.settings.temperature.getValue(),
            'top_p': self.settings.top_p.getValue(),
            'max_tokens': self.settings.max_tokens.getValue(),
            'max_completion_tokens': self.settings.max_completion_tokens.getValue(),
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        logger.info(response)
        logger.info(f'Status Code: {response.status_code}')
        response_json = response.json()
        logger.info(f'JSON Response: {response_json}')
        response.raise_for_status()

        content = response_json['response']
        # FIXME _convert_dict_to_message
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Uniquely identifies the type of the model. Used for logging."""
        model = self.settings.chatModel.getValue()
        return f'argo-{model}'


class ArgoEmbeddings(Embeddings):
    def __init__(self, settings: ArgoSettings) -> None:
        self._settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of strings."""
        url = self._settings.embeddingsEndpointURL.getValue()
        payload = {
            'user': self._settings.user.getValue(),
            'model': self._settings.embeddingsModel.getValue(),
            'prompt': texts,
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        logger.info(response)
        logger.info(f'Status Code: {response.status_code}')
        response_json = response.json()
        logger.info(f'JSON Response: {response_json}')
        response.raise_for_status()

        return response_json['embedding']

    def embed_query(self, text: str) -> list[float]:
        """Generates an embedding for a single text query."""
        return self.embed_documents([text])[0]
