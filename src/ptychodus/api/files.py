"""Common bases for the file reader and writer plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar

__all__ = [
    'FileFormat',
    'FileHandler',
    'FileReader',
    'FileWriter',
]

T = TypeVar('T')


def _build_file_filter(description: str, extensions: Collection[str]) -> str:
    """Return the Qt name filter naming *description* and the *extensions* it handles.

    Extensions are sorted so that the filter is stable from run to run: it is also
    the name a plugin registers under and the one settings resolve against.
    """
    globs = ' '.join(f'*{extension}' for extension in sorted(extensions))
    return f'{description} ({globs})'


class FileFormat(Enum):
    """A file format common to several handlers, naming itself once.

    A handler whose format is one of these declares the member rather than repeating
    its description and extensions; a handler that reads a common format out of an
    instrument's own layout keeps its own description and takes only the extensions.
    Anything that needs the Qt name filter without holding a handler reads it from
    ``file_filter``, so the string it shows cannot drift from the one a handler
    registers under.

    A member is named for the simple name its handlers register under, so that the
    member, the registration and the settings key that persists it all read the same.
    """

    CSV = ('Comma-Separated Values Files', frozenset({'.csv'}))
    # A CXI file is an HDF5 file, so it carries the HDF5 extensions as well as its own.
    CXI = ('Coherent X-ray Imaging Files', frozenset({'.cxi', '.h5', '.hdf5'}))
    HDF5 = ('Hierarchical Data Format 5 Files', frozenset({'.h5', '.hdf5'}))
    INI = ('Initialization Files', frozenset({'.ini'}))
    NPY = ('NumPy Binary Files', frozenset({'.npy'}))
    NPZ = ('NumPy Zipped Archive', frozenset({'.npz'}))
    TIFF = ('Tagged Image File Format Files', frozenset({'.tif', '.tiff'}))

    def __init__(self, description: str, extensions: frozenset[str]) -> None:
        self.description = description
        self._extensions = extensions

    @property
    def extensions(self) -> set[str]:
        """Return the handled file extensions, each with a leading dot.

        The set is a fresh copy, so a handler can return it directly without its
        caller being able to mutate the format.
        """
        return set(self._extensions)

    @property
    def file_filter(self) -> str:
        """Return the Qt name filter naming this format and the extensions it handles."""
        return _build_file_filter(self.description, self._extensions)


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
        """Return the Qt name filter naming this format and the extensions it handles."""
        return _build_file_filter(self.get_file_description(), self.get_file_extensions())


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
