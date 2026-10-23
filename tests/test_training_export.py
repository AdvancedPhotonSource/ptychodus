"""ProcessingAPI's training-data export: the reconstructor's own format or CXI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ptychodus.api.reconstruct import SubprocessReconstructor
from ptychodus.model.processing.api import ProcessingAPI

_CXI_FILTER = 'Coherent X-ray Imaging Files (*.cxi *.h5 *.hdf5)'


def _reconstructor(
    *, is_trainable: bool, file_filter: str = 'Native Files (*.npz)'
) -> SubprocessReconstructor:
    return SubprocessReconstructor(
        name='test',
        dump_reconstruct_options=MagicMock(),
        is_trainable=is_trainable,
        training_data_file_filter=file_filter,
        export_training_data=MagicMock(),
    )


def _processing_api(reconstructor: SubprocessReconstructor) -> ProcessingAPI:
    algorithm_parameter = MagicMock()
    algorithm_parameter.get_current_reconstructor.return_value = reconstructor
    api = ProcessingAPI(
        MagicMock(), MagicMock(), MagicMock(), algorithm_parameter, {}, MagicMock(), MagicMock()
    )
    return api


def test_filters_offer_the_native_format_first_and_then_cxi() -> None:
    api = _processing_api(_reconstructor(is_trainable=True))

    filters = api.get_training_data_file_filters()

    assert list(filters.items()) == [('Native Files (*.npz)', None), (_CXI_FILTER, 'CXI')]


def test_a_reconstructor_without_a_native_format_still_offers_cxi() -> None:
    api = _processing_api(_reconstructor(is_trainable=True, file_filter=''))

    assert api.get_training_data_file_filters() == {_CXI_FILTER: 'CXI'}


def test_a_reconstructor_that_cannot_train_offers_nothing() -> None:
    api = _processing_api(_reconstructor(is_trainable=False))

    assert api.get_training_data_file_filters() == {}


def test_the_native_format_is_exported_by_the_reconstructor(tmp_path: Path) -> None:
    reconstructor = _reconstructor(is_trainable=True)
    api = _processing_api(reconstructor)
    reconstruct_input = MagicMock()
    options = MagicMock()

    with (
        patch.object(api, 'get_reconstruct_input', return_value=reconstruct_input),
        patch.object(api, '_options_for', return_value=options),
    ):
        api.export_training_data(tmp_path / 'native.npz', product_index=0)

    export = reconstructor._export_training_data  # noqa: SLF001
    assert isinstance(export, MagicMock)
    export.assert_called_once_with(tmp_path / 'native.npz', reconstruct_input, options)


def test_cxi_is_written_with_its_suffix(tmp_path: Path) -> None:
    api = _processing_api(_reconstructor(is_trainable=True))
    reconstruct_input = MagicMock()

    with (
        patch.object(api, 'get_reconstruct_input', return_value=reconstruct_input),
        patch('ptychodus.model.processing.api.save_cxi_reconstruct_input') as save,
    ):
        api.export_training_data(tmp_path / 'exported', product_index=0, file_type='cxi')

    save.assert_called_once_with(tmp_path / 'exported.cxi', reconstruct_input)


def test_an_unknown_file_type_is_refused(tmp_path: Path) -> None:
    api = _processing_api(_reconstructor(is_trainable=True))

    with pytest.raises(ValueError, match='Unknown training data file type'):
        api.export_training_data(tmp_path / 'x', product_index=0, file_type='TIFF')
