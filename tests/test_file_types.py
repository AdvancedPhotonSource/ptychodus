"""Unit tests for ptychodus.model.file_types.FileTypeChooser.

The chooser is the one place the model layer turns a PluginChooser of file
handlers into what a file dialog needs: the filters to offer, the one to open on,
and a selection made by whichever name came back. These tests pin that surface,
including the forgiving behavior an unrecognized name has always had.
"""

from __future__ import annotations

from pathlib import Path
import logging

import pytest

from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginChooser
from ptychodus.model.file_types import FileTypeChooser


class _Reader(ObjectFileReader):
    def __init__(self, description: str, extension: str) -> None:
        self._description = description
        self._extension = extension

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return {self._extension}

    def read(self, file_path: Path) -> Object:
        raise NotImplementedError


@pytest.fixture
def file_types() -> FileTypeChooser[ObjectFileReader]:
    chooser: PluginChooser[ObjectFileReader] = PluginChooser()
    chooser.register_plugin(_Reader('Alpha Files', '.alpha'), simple_name='alpha')
    chooser.register_plugin(_Reader('Beta Files', '.beta'), simple_name='beta')
    return FileTypeChooser(chooser)


def test_filters_are_the_handlers_own(file_types: FileTypeChooser[ObjectFileReader]) -> None:
    assert list(file_types.get_filters()) == ['Alpha Files (*.alpha)', 'Beta Files (*.beta)']


def test_filters_are_a_sequence_a_dialog_can_take_directly(
    file_types: FileTypeChooser[ObjectFileReader],
) -> None:
    """QFileDialog.setNameFilters wants a Sequence, so the caller needs no comprehension."""
    filters = file_types.get_filters()

    assert len(filters) == 2
    assert filters[0] == 'Alpha Files (*.alpha)'


def test_filter_tracks_the_selection(file_types: FileTypeChooser[ObjectFileReader]) -> None:
    assert file_types.get_filter() == 'Alpha Files (*.alpha)'

    file_types.select('Beta Files (*.beta)')

    assert file_types.get_filter() == 'Beta Files (*.beta)'


def test_select_returns_the_plugin_it_moved_to(
    file_types: FileTypeChooser[ObjectFileReader],
) -> None:
    plugin = file_types.select('Beta Files (*.beta)')

    assert plugin.simple_name == 'beta'
    assert plugin.strategy.get_file_description() == 'Beta Files'


def test_select_resolves_a_simple_name_too(
    file_types: FileTypeChooser[ObjectFileReader],
) -> None:
    """A name persisted in settings picks the same file type a dialog filter would."""
    assert file_types.select('beta').simple_name == 'beta'


def test_select_none_keeps_the_current_selection(
    file_types: FileTypeChooser[ObjectFileReader],
) -> None:
    """None is what a caller means by "no file type to impose"."""
    file_types.select('beta')

    assert file_types.select(None).simple_name == 'beta'
    assert file_types.select().simple_name == 'beta'


def test_select_an_unknown_name_warns_and_keeps_the_selection(
    file_types: FileTypeChooser[ObjectFileReader], caplog: pytest.LogCaptureFixture
) -> None:
    """Pinned deliberately: the file is then handled by whichever type was selected.

    Preserved from before this chooser existed rather than endorsed -- changing it
    is a user-visible decision, because a stale name in a settings file currently
    loads with a warning instead of failing.
    """
    file_types.select('beta')

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.plugins'):
        plugin = file_types.select('nonesuch')

    assert 'Invalid plugin name' in caplog.text
    assert plugin.simple_name == 'beta'
