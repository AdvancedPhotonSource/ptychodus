"""Settings-backed parameter adapters over the plugin choosers."""

from __future__ import annotations
from collections.abc import Iterator
from typing import Generic, TypeVar

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import Parameter, StringParameter
from ptychodus.api.plugins import Plugin, PluginChooser

__all__ = [
    'PluginChooserParameter',
]

T = TypeVar('T')


class PluginChooserParameter(Parameter[str], Observer, Generic[T]):
    """Parameter[str] view of a PluginChooser whose value space is plugin display names.

    A chooser has two name spaces: the human-readable ``display_name`` a person
    chooses by and the ``simple_name`` persisted to settings. This adapter is the
    single place that translates between them. Its own value space is display
    names, so an editor bound to it round trips without knowing there are two.

    Passing *settings* additionally binds the chooser to that settings parameter: the
    persisted name selects a plugin at construction and is rewritten to its canonical
    simple name, and every later selection change writes the simple name back. A
    persisted name that matches no registered plugin is left alone rather than
    reconciled, because a plugin can be missing merely because its optional
    dependency failed to import this run.

    Note that :meth:`get_value` raises ``LookupError`` when no plugins are registered,
    so this must not be read before ``PluginRegistry.load_plugins()`` has run.
    """

    def __init__(self, chooser: PluginChooser[T], settings: StringParameter | None = None) -> None:
        super().__init__()
        self._chooser = chooser
        self._settings = settings
        self._selected: Plugin[T] | None = None
        self._suppress_notify = False

        if settings is not None:
            self._apply_settings()
            settings.add_observer(self)

        self._selected = chooser.get_current_plugin() if chooser else None
        chooser.add_observer(self)

    def choices(self) -> Iterator[str]:
        """Yield every value this parameter accepts, by display name."""
        for plugin in self._chooser:
            yield plugin.display_name

    def get_strategy(self) -> T:
        return self._chooser.get_current_plugin().strategy

    def get_value(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def set_value(self, value: str, *, notify: bool = True) -> None:
        # The chooser notifies us back through _update; suppress that relay rather
        # than notifying here, so a no-op selection stays silent either way. Only the
        # relay is suppressed: persistence must not depend on a view flag.
        self._suppress_notify = not notify

        try:
            self._chooser.set_current_plugin(value)
        finally:
            self._suppress_notify = False

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def copy(self) -> Parameter[str]:
        """Return an unbound view of the same chooser; the copy does not persist."""
        return PluginChooserParameter(self._chooser)

    def _apply_settings(self) -> None:
        settings = self._settings

        if settings is None:
            return

        name = settings.get_value()
        plugin = self._chooser.find_plugin(name)
        self._chooser.set_current_plugin(name)

        if plugin is not None:
            # Unconditional, so that a display name normalizes to its simple name even
            # when it already resolves to the current selection. ParameterBase guards
            # against no-op writes, so this does not notify unless the value changed.
            settings.set_value(plugin.simple_name)

    def _reconcile(self) -> None:
        """Settle the chooser and the settings parameter against each other.

        Observable notifications carry no payload, so the cached selection is what
        distinguishes "the selection moved" from "a plugin was registered". Only the
        former may write back; the latter must leave an unresolved setting intact.
        """
        settings = self._settings
        plugin = self._chooser.get_current_plugin() if self._chooser else None

        if plugin is None or settings is None:
            self._selected = plugin
            return

        if self._selected is None:
            # The chooser just gained its first plugin: the selection came into
            # existence rather than moving, so the fallback must not be persisted.
            self._selected = plugin

        if plugin is not self._selected:
            self._selected = plugin
            settings.set_value(plugin.simple_name)
            return

        # The selection held, so this was a registration. If it has just made the
        # persisted name resolvable, honor it now — set_current_plugin re-enters
        # here through the chooser's notification to finish the write-back.
        desired = self._chooser.find_plugin(settings.get_value())

        if desired is not None and desired is not plugin:
            self._apply_settings()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._apply_settings()
        elif observable is self._chooser:
            self._reconcile()

            if not self._suppress_notify:
                self.notify_observers()
