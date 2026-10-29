"""Plugin selection over the registry's read-only catalogs, and its settings adapter."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar
import logging

from ptychodus.api.plugins import Plugin, PluginCatalog

from .observer import Observable, Observer
from .parameters import Parameter, StringParameter

__all__ = [
    'PluginChooser',
    'PluginChooserParameter',
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PluginChooser(Iterable[Plugin[T]], Observable):
    """Observable selection of one plugin from a catalog.

    The selection starts at the catalog's first plugin and moves only through
    :meth:`set_current_plugin`, notifying observers when it does. The catalog's
    membership must be settled before it is wrapped: a chooser does not see later
    registrations. Wrap each catalog once and share the chooser, since two choosers
    over one catalog hold two independent selections.
    """

    def __init__(self, catalog: PluginCatalog[T]) -> None:
        super().__init__()
        self._catalog = catalog
        self._current: Plugin[T] | None = catalog[0] if catalog else None

    def find_plugin(self, name: str) -> Plugin[T] | None:
        """Return the plugin matching *name* (case-insensitive simple or display name), or None."""
        return self._catalog.find_plugin(name)

    def get_strategy_by_name(self, name: str) -> T:
        """Return the strategy of the plugin matching *name*; raise LookupError otherwise."""
        return self._catalog.get_strategy_by_name(name)

    def get_current_plugin(self) -> Plugin[T]:
        """Return the currently selected plugin."""
        if self._current is None:
            raise LookupError('No plugins registered')
        return self._current

    def set_current_plugin(self, name: str) -> None:
        """Select the plugin matching *name* (case-insensitive simple or display name).

        An unrecognized name logs a warning and leaves the selection unchanged, but
        still notifies observers so that a bound view resynchronizes to the selection
        the chooser actually holds.
        """
        plugin = self._catalog.find_plugin(name)

        if plugin is None:
            registered_plugins = ', '.join(f'"{other.simple_name}"' for other in self._catalog)
            logger.warning(
                f'Invalid plugin name "{name}". Registered plugins: {registered_plugins}.'
            )
            self.notify_observers()
            return

        if plugin is not self._current:
            self._current = plugin
            self.notify_observers()

    def __iter__(self) -> Iterator[Plugin[T]]:
        return iter(self._catalog)

    def __bool__(self) -> bool:
        return bool(self._catalog)


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

    Note that :meth:`get_value` raises ``LookupError`` when the chooser's catalog is
    empty.
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
        """Persist the chooser's selection if it moved.

        Observable notifications carry no payload, so the cached selection is what
        distinguishes "the selection moved" from a notification that left it in
        place, such as the one an unrecognized name sends. Only the former may write
        back; the latter must leave an unresolved setting intact.
        """
        plugin = self._chooser.get_current_plugin() if self._chooser else None

        if plugin is not self._selected:
            self._selected = plugin

            if plugin is not None and self._settings is not None:
                self._settings.set_value(plugin.simple_name)

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._apply_settings()
        elif observable is self._chooser:
            self._reconcile()

            if not self._suppress_notify:
                self.notify_observers()
