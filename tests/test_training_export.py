"""ProcessingAPI's training-data export: any registered format, whatever is selected."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptychodus.api.plugins import PluginCatalog, PluginList
from ptychodus.api.reconstruct import (
    NullReconstructor,
    ReconstructInput,
    TrainingDataFileWriter,
)
from ptychodus.model.processing.api import ProcessingAPI
from ptychodus.plugins.cxi_file import CXITrainingDataFileWriter


class _StubWriter(TrainingDataFileWriter):
    def __init__(self, description: str, extension: str) -> None:
        self._description = description
        self._extension = extension
        self.written: list[tuple[Path, ReconstructInput]] = []

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return {self._extension}

    def write(self, file_path: Path, parameters: ReconstructInput) -> None:
        self.written.append((file_path, parameters))


def _writers() -> tuple[PluginCatalog[TrainingDataFileWriter], dict[str, _StubWriter]]:
    stubs = {
        'CXI': _StubWriter('CXI Files', '.cxi'),
        'Native': _StubWriter('Native Files', '.npz'),
    }
    writers = PluginList[TrainingDataFileWriter]()

    for simple_name, stub in stubs.items():
        writers.register_plugin(stub, simple_name=simple_name)

    return writers, stubs


def _processing_api(
    writers: PluginCatalog[TrainingDataFileWriter], *, library: str = 'Native'
) -> ProcessingAPI:
    algorithm_parameter = MagicMock()
    # Not trainable: export must not depend on what is selected being able to train.
    algorithm_parameter.get_current_reconstructor.return_value = NullReconstructor('None')
    algorithm_parameter.get_current_parts.return_value = (library, 'Algorithm')
    return ProcessingAPI(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        algorithm_parameter,
        {},
        MagicMock(),
        MagicMock(),
        writers,
    )


def test_every_registered_writer_is_offered() -> None:
    writers, _ = _writers()
    api = _processing_api(writers)

    assert api.get_training_data_file_filters() == {
        'CXI Files (*.cxi)': 'CXI',
        'Native Files (*.npz)': 'Native',
    }


def test_the_default_is_the_selected_library_s_own_format() -> None:
    writers, _ = _writers()

    assert _processing_api(writers, library='native').get_default_training_data_file_type() == (
        'Native'
    )


@pytest.mark.parametrize('library', ['pty-chi', ''])
def test_a_library_without_a_format_defaults_to_cxi(library: str) -> None:
    writers, _ = _writers()

    assert _processing_api(writers, library=library).get_default_training_data_file_type() == (
        'CXI'
    )


def test_the_named_writer_writes_the_reconstruct_input(tmp_path: Path) -> None:
    writers, stubs = _writers()
    api = _processing_api(writers)
    reconstruct_input = MagicMock()

    with patch.object(api, 'get_reconstruct_input', return_value=reconstruct_input):
        api.export_training_data(tmp_path / 'exported.cxi', product_index=0, file_type='cxi')

    assert stubs['CXI'].written == [(tmp_path / 'exported.cxi', reconstruct_input)]
    assert stubs['Native'].written == []


def test_no_file_type_writes_the_default(tmp_path: Path) -> None:
    writers, stubs = _writers()
    api = _processing_api(writers, library='Native')

    with patch.object(api, 'get_reconstruct_input', return_value=MagicMock()):
        api.export_training_data(tmp_path / 'exported.npz', product_index=0)

    assert len(stubs['Native'].written) == 1
    assert stubs['CXI'].written == []


def test_an_unknown_file_type_is_refused(tmp_path: Path) -> None:
    writers, _ = _writers()
    api = _processing_api(writers)

    with pytest.raises(LookupError, match='TIFF'):
        api.export_training_data(tmp_path / 'x', product_index=0, file_type='TIFF')


def test_the_cxi_writer_adds_its_suffix(tmp_path: Path) -> None:
    reconstruct_input = MagicMock()

    with patch('ptychodus.plugins.cxi_file.save_cxi_reconstruct_input') as save:
        CXITrainingDataFileWriter().write(tmp_path / 'exported', reconstruct_input)
        CXITrainingDataFileWriter().write(tmp_path / 'kept.h5', reconstruct_input)

    assert [call.args[0] for call in save.call_args_list] == [
        tmp_path / 'exported.cxi',
        tmp_path / 'kept.h5',
    ]
