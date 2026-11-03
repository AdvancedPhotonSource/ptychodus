from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ptychodus.api.language_model import (
    LanguageModelEndpoint,
    LanguageModelProvider,
    LanguageModelProviderError,
)
from ptychodus.api.plugins import Plugin

from ..declared import ParameterDescription, declare_in, describe, restore
from ..observer import Observable, Observer
from ..parameters import Parameter, StringParameter
from ..plugins import PluginChooser, PluginChooserParameter
from ..settings import SettingsRegistry
from .settings import AgentSettings


class LanguageModelProviderOptions:
    """The live options and model of one provider, persisted in its own settings group.

    The group is ``Agent<simple_name>``, so each provider remembers its own URL and
    model across a switch to another and back.
    """

    def __init__(self, registry: SettingsRegistry, plugin: Plugin[LanguageModelProvider]) -> None:
        self.plugin = plugin
        self._provider = plugin.strategy
        self._defaults = self._provider.get_default_options()
        self._described = describe(self._defaults)
        group = registry.create_group(f'Agent{plugin.simple_name}')
        self._parameters = declare_in(group, self._described)
        self.model = group.create_string_parameter('Model', self._provider.get_default_model())

    def describe_parameters(self) -> Sequence[ParameterDescription]:
        """What the provider declared, in declaration order, keyed into :meth:`parameters`."""
        return self._described

    def parameters(self) -> Mapping[str, Parameter[Any]]:
        return self._parameters

    def get_options(self) -> BaseModel:
        return restore(self._defaults, self._described, self._parameters)

    def list_models(self, *, timeout: float = 10.0) -> Sequence[str]:
        return self._provider.list_models(self.get_options(), timeout=timeout)

    def resolve_endpoint(self) -> LanguageModelEndpoint:
        model_id = self.model.get_value().strip()

        if not model_id:
            raise LanguageModelProviderError(f'Choose a {self.plugin.display_name} model.')

        return self._provider.resolve(self.get_options(), model_id)


class LanguageModelProviderSelection(Observable, Observer):
    """Which provider the agent talks to, and the options of every provider.

    Every provider's group is declared, not only the selected one's, so the settings
    file holds the same groups however the application was driven. Notifies when the
    selection moves.
    """

    def __init__(
        self,
        registry: SettingsRegistry,
        settings: AgentSettings,
        chooser: PluginChooser[LanguageModelProvider],
    ) -> None:
        super().__init__()
        self._chooser = chooser
        self._options = {
            plugin.simple_name: LanguageModelProviderOptions(registry, plugin) for plugin in chooser
        }
        self.provider = PluginChooserParameter(chooser, settings.provider)
        chooser.add_observer(self)

    def __iter__(self) -> Iterator[LanguageModelProviderOptions]:
        return iter(self._options.values())

    def get_current(self) -> LanguageModelProviderOptions:
        """Return the selected provider's options.

        Raises:
            LookupError: if no provider is registered.
        """
        return self._options[self._chooser.get_current_plugin().simple_name]

    def get_current_model(self) -> StringParameter:
        return self.get_current().model

    def resolve_endpoint(self) -> LanguageModelEndpoint:
        return self.get_current().resolve_endpoint()

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
