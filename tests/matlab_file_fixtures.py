"""Write a ``-v7.3`` MATLAB file, so the HDF5 reading path can be tested without MATLAB.

MATLAB switched ``save`` to an HDF5 container at ``-v7.3`` and fold_slice has used that
flag since 2023, but no such file is checked in and producing one needs a MATLAB
licence. This builds the same container from the format description: dimensions stored
in reverse, a complex array as a two-field compound, a struct as a group, a cell as a
dataset of object references, and the 512-byte text header in front of the payload.

Three details are modelled rather than observed, and are the first thing to check
against a real file: the names and order of the complex compound's fields, that a
scalar struct is a plain group holding one dataset per field, and the cell layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy

__all__ = [
    'write_matlab_v73',
]

# What MATLAB calls each storage type. A complex array declares the class of its
# components, so both precisions appear twice.
_MATLAB_CLASSES: dict[numpy.dtype[Any], str] = {
    numpy.dtype(numpy.float64): 'double',
    numpy.dtype(numpy.complex128): 'double',
    numpy.dtype(numpy.float32): 'single',
    numpy.dtype(numpy.complex64): 'single',
    numpy.dtype(numpy.int64): 'int64',
    numpy.dtype(numpy.int32): 'int32',
    numpy.dtype(numpy.bool_): 'logical',
}

_USERBLOCK_SIZE = 512


def _as_matlab_array(value: Any) -> numpy.ndarray:
    """Shape a value the way MATLAB would hold it, before the dimensions are reversed.

    MATLAB has no rank below two, so a scalar is a 1x1 and a flat sequence is a row.
    """
    values = numpy.asarray(value)

    if values.ndim == 0:
        return values.reshape(1, 1)

    if values.ndim == 1:
        return values.reshape(1, -1)

    return values


def _write_value(h5_group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, Mapping):
        subgroup = h5_group.create_group(key)
        subgroup.attrs['MATLAB_class'] = numpy.bytes_(b'struct')

        for subkey, subvalue in value.items():
            _write_value(subgroup, subkey, subvalue)

        return

    if isinstance(value, str):
        codes = numpy.array([[ord(character) for character in value]], dtype=numpy.uint16)
        dataset = h5_group.create_dataset(key, data=codes.transpose())
        dataset.attrs['MATLAB_class'] = numpy.bytes_(b'char')
        dataset.attrs['MATLAB_int_decode'] = numpy.uint8(2)
        return

    if isinstance(value, list):
        _write_cell(h5_group, key, value)
        return

    values = _as_matlab_array(value)
    matlab_class = _MATLAB_CLASSES[values.dtype]

    if numpy.iscomplexobj(values):
        component_dtype = numpy.dtype(f'<f{values.dtype.itemsize // 2}')
        compound_dtype = numpy.dtype([('real', component_dtype), ('imag', component_dtype)])
        values = numpy.ascontiguousarray(values).view(compound_dtype)

    dataset = h5_group.create_dataset(key, data=numpy.ascontiguousarray(values.transpose()))
    dataset.attrs['MATLAB_class'] = numpy.bytes_(matlab_class.encode())


def _write_cell(h5_group: h5py.Group, key: str, items: list[Any]) -> None:
    """Store a cell array as a dataset of references into the file's ``#refs#`` group."""
    h5_file = h5_group.file
    refs = h5_file.require_group('#refs#')
    references = []

    for item in items:
        name = str(len(refs))
        _write_value(refs, name, item)
        references.append(refs[name].ref)

    data = numpy.array(references, dtype=h5py.ref_dtype).reshape(1, -1)
    dataset = h5_group.create_dataset(key, data=data.transpose())
    dataset.attrs['MATLAB_class'] = numpy.bytes_(b'cell')


def write_matlab_v73(file_path: Path, variables: Mapping[str, Any]) -> Path:
    """Write *variables* as a ``-v7.3`` file, and return the path.

    Accepts the same mapping ``scipy.io.savemat`` does, so one workspace can be written
    both ways and the two read back compared.
    """
    with h5py.File(file_path, 'w', userblock_size=_USERBLOCK_SIZE) as h5_file:
        for key, value in variables.items():
            _write_value(h5_file, key, value)

    header = b'MATLAB 7.3 MAT-file, Platform: PCWIN64, Created by: ptychodus tests'

    with open(file_path, 'r+b') as binary_file:
        binary_file.write(header.ljust(_USERBLOCK_SIZE, b'\x00'))

    return file_path
