from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatModelProperties:
    name: str

    @property
    def is_o_series_model(self) -> bool:
        return self.name.startswith('gpto')

    @property
    def accepts_system_prompt(self) -> bool:
        return not self.is_o_series_model

    @property
    def accepts_stop_sequence(self) -> bool:
        return not self.is_o_series_model

    @property
    def accepts_temperature(self) -> bool:
        return not self.is_o_series_model

    @property
    def accepts_top_p(self) -> bool:
        return not self.is_o_series_model

    @property
    def accepts_max_tokens(self) -> bool:
        return not self.is_o_series_model

    @property
    def accepts_max_completion_tokens(self) -> bool:
        return self.is_o_series_model


def list_argo_model_properties() -> Sequence[ChatModelProperties]:
    return [
        ChatModelProperties(name='gpt35'),
        ChatModelProperties(name='gpt35large'),
        ChatModelProperties(name='gpt4'),
        ChatModelProperties(name='gpt4large'),
        ChatModelProperties(name='gpt4o'),
        ChatModelProperties(name='gpt4olatest'),
        ChatModelProperties(name='gpt4turbo'),
        ChatModelProperties(name='gpto1'),
        ChatModelProperties(name='gpto1mini'),
        ChatModelProperties(name='gpto3mini'),
    ]
