"""Unit tests for the fluorescence api data structures."""

from __future__ import annotations

import numpy
import pytest

from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset


def _make_dataset(*names: str, shape: tuple[int, int] = (4, 5)) -> FluorescenceDataset:
    element_maps = [ElementMap(name, numpy.zeros(shape, dtype=numpy.float32)) for name in names]
    return FluorescenceDataset(
        element_maps=element_maps,
        counts_per_second_path='/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec',
        channel_names_path='/MAPS/XRF_Analyzed/NNLS/Channel_Names',
    )


def test_element_map_nbytes_is_the_array_size() -> None:
    array = numpy.zeros((4, 5), dtype=numpy.float32)
    assert ElementMap('Fe', array).nbytes == array.nbytes


def test_dataset_nbytes_sums_its_element_maps() -> None:
    dataset = _make_dataset('Fe', 'Cu', 'Zn')
    assert dataset.nbytes == sum(element_map.nbytes for element_map in dataset.element_maps)
    assert dataset.nbytes == 3 * 4 * 5 * 4


def test_empty_dataset_has_no_bytes() -> None:
    assert _make_dataset().nbytes == 0


def test_dataset_is_sized() -> None:
    assert len(_make_dataset('Fe', 'Cu')) == 2
    assert len(_make_dataset()) == 0


def test_dataset_supports_integer_indexing() -> None:
    dataset = _make_dataset('Fe', 'Cu')
    assert dataset[0].name == 'Fe'
    assert dataset[-1].name == 'Cu'

    with pytest.raises(IndexError):
        dataset[2]


def test_dataset_supports_slicing() -> None:
    dataset = _make_dataset('Fe', 'Cu', 'Zn')
    assert [element_map.name for element_map in dataset[1:]] == ['Cu', 'Zn']


def test_dataset_is_iterable() -> None:
    dataset = _make_dataset('Fe', 'Cu')
    assert [element_map.name for element_map in dataset] == ['Fe', 'Cu']


def test_dataset_supports_containment_and_index() -> None:
    dataset = _make_dataset('Fe', 'Cu')
    assert dataset[1] in dataset
    assert dataset.index(dataset[1]) == 1


def test_dataset_repr_names_its_maps() -> None:
    assert repr(_make_dataset('Fe', 'Cu')) == 'FluorescenceDataset(2 maps: Fe, Cu)'
