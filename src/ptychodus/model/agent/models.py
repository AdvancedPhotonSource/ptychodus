from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ChatRole(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str
    created_at: datetime
