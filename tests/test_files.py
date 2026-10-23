"""Unit tests for ptychodus.api.files.FileHandler and FileFormat.

The base contributes exactly one piece of behavior: composing a Qt name filter from
a description and a set of extensions. These tests pin that composition, including
the sort that keeps the filter -- and so the name a plugin registers under -- stable
however the extension set happens to iterate. FileFormat states that same pair once
for a format several handlers share, so the tests also pin that a member composes
its filter exactly as a handler declaring it would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.files import FileFormat, FileHandler, FileReader, FileWriter


class _Handler(FileHandler):
    def __init__(self, description: str, extensions: set[str]) -> None:
        self._description = description
        self._extensions = extensions

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return self._extensions


def test_single_extension() -> None:
    handler = _Handler('NumPy Binary Files', {'.npy'})
    assert handler.get_file_filter() == 'NumPy Binary Files (*.npy)'


def test_multiple_extensions_are_sorted() -> None:
    handler = _Handler('Tagged Image File Format Files', {'.tiff', '.tif'})
    assert handler.get_file_filter() == 'Tagged Image File Format Files (*.tif *.tiff)'


def test_filter_is_independent_of_set_iteration_order() -> None:
    forward = _Handler('Ptychodus Product Files', {'.h5', '.hdf5'})
    reverse = _Handler('Ptychodus Product Files', {'.hdf5', '.h5'})
    assert forward.get_file_filter() == reverse.get_file_filter()
    assert forward.get_file_filter() == 'Ptychodus Product Files (*.h5 *.hdf5)'


@pytest.mark.parametrize(
    'description, extensions, expected',
    [
        ('APS 4-ID-B,G,H POLAR Files', {'.hdf'}, 'APS 4-ID-B,G,H POLAR Files (*.hdf)'),
        (
            'APS 33-ID-C VelociProbe Files - Laser Interferometer',
            {'.txt'},
            'APS 33-ID-C VelociProbe Files - Laser Interferometer (*.txt)',
        ),
        (
            'Coherent X-ray Imaging Files, SHARP layout',
            {'.cxi'},
            'Coherent X-ray Imaging Files, SHARP layout (*.cxi)',
        ),
    ],
)
def test_punctuation_in_description_is_preserved(
    description: str, extensions: set[str], expected: str
) -> None:
    assert _Handler(description, extensions).get_file_filter() == expected


def test_a_reader_must_still_describe_itself() -> None:
    """The format contract is not waived by implementing only the payload half."""

    class _Reader(FileReader[int]):
        def get_file_extensions(self) -> set[str]:
            return {'.npy'}

        def read(self, file_path: Path) -> int:
            return 0

    with pytest.raises(TypeError, match='get_file_description'):
        _Reader()  # type: ignore[abstract]


def test_a_writer_must_still_describe_itself() -> None:
    class _Writer(FileWriter[int]):
        def get_file_extensions(self) -> set[str]:
            return {'.npy'}

        def write(self, file_path: Path, value: int) -> None:
            pass

    with pytest.raises(TypeError, match='get_file_description'):
        _Writer()  # type: ignore[abstract]


@pytest.mark.parametrize('file_format', list(FileFormat))
def test_every_format_declares_extensions_with_a_leading_dot(file_format: FileFormat) -> None:
    """A dotless extension compares false against every PurePath.suffix."""
    extensions = file_format.extensions
    assert extensions
    assert all(extension.startswith('.') for extension in sorted(extensions))


@pytest.mark.parametrize('file_format', list(FileFormat))
def test_a_format_filters_exactly_as_a_handler_declaring_it(file_format: FileFormat) -> None:
    """The two paths to a filter agree, so a dialog naming a format directly and a
    handler registering under it cannot show different text for the same format."""
    handler = _Handler(file_format.description, file_format.extensions)
    assert file_format.file_filter == handler.get_file_filter()


def test_a_format_hands_out_a_copy_of_its_extensions() -> None:
    """A handler returns this set to its caller, so mutating it must not move the format."""
    extensions = FileFormat.CXI.extensions
    extensions.add('.bogus')
    assert '.bogus' not in FileFormat.CXI.extensions


def test_cxi_accepts_the_hdf5_extensions() -> None:
    """A CXI file is an HDF5 file, and instruments name it either way."""
    assert FileFormat.HDF5.extensions <= FileFormat.CXI.extensions
    assert FileFormat.CXI.file_filter == 'Coherent X-ray Imaging Files (*.cxi *.h5 *.hdf5)'
