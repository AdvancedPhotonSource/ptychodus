"""Common bases for the file reader and writer plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

__all__ = [
    'FileHandler',
    'FileReader',
    'FileWriter',
]

T = TypeVar('T')


class FileHandler(ABC):
    """Plugin interface for anything that reads or writes a file format.

    A subclass states the format in two parts, a human-readable description and the
    set of extensions it handles, and inherits the Qt-style name filter built from
    them. Extensions carry a leading dot so that they compare directly against
    :attr:`pathlib.PurePath.suffix`.
    """

    @abstractmethod
    def get_file_description(self) -> str:
        """Return the human-readable format name, without extensions."""
        pass

    @abstractmethod
    def get_file_extensions(self) -> set[str]:
        """Return the handled file extensions, each with a leading dot."""
        pass

    def get_file_filter(self) -> str:
        """Return the Qt name filter naming this format and the extensions it handles.

        Extensions are sorted so that the filter is stable from run to run: it is
        also the name a plugin registers under and the one settings resolve against.
        """
        globs = ' '.join(f'*{extension}' for extension in sorted(self.get_file_extensions()))
        return f'{self.get_file_description()} ({globs})'


class FileReader(FileHandler, Generic[T]):
    """Plugin interface for reading one kind of object from a file."""

    @abstractmethod
    def read(self, file_path: Path) -> T:
        """Read one object from file."""
        pass


class FileWriter(FileHandler, Generic[T]):
    """Plugin interface for writing one kind of object to a file."""

    @abstractmethod
    def write(self, file_path: Path, data: T, /) -> None:
        """Write one object to file.

        The signature is positional-only, because every implementation names the
        payload for its own domain -- ``product``, ``probes``, ``object_``. The name
        here is therefore not part of the contract, and marking it so makes a keyword
        call a type error rather than one that fails against every implementation.
        """
        pass
