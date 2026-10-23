"""A plugin chooser of file handlers, presented the way a file dialog needs it."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Generic, TypeVar

from ptychodus.api.files import FileHandler
from ptychodus.api.plugins import Plugin, PluginChooser

__all__ = [
    'FileTypeChooser',
]

H = TypeVar('H', bound=FileHandler)


class FileTypeChooser(Generic[H]):
    """The file types a dialog can offer, and the one it opens on.

    A file type is named by its filter, which a handler builds from the format it
    describes, so the string a dialog hands back selects the handler that produced
    it. Both of a chooser's name spaces resolve, so a name persisted in settings
    selects the same file type a dialog would.
    """

    def __init__(self, chooser: PluginChooser[H]) -> None:
        self._chooser = chooser

    def get_filters(self) -> Sequence[str]:
        """Return every file type's filter, in the order the chooser holds them."""
        return [plugin.display_name for plugin in self._chooser]

    def get_filter(self) -> str:
        """Return the filter of the currently selected file type."""
        return self._chooser.get_current_plugin().display_name

    def select(self, file_type: str | None = None) -> Plugin[H]:
        """Select the file type named by *file_type* and return it.

        None keeps the current selection, which is what a caller means when it has no
        file type to impose. An unrecognized name warns and keeps the selection, so
        the file is handled by whichever type was selected before.
        """
        if file_type is not None:
            self._chooser.set_current_plugin(file_type)

        return self._chooser.get_current_plugin()
