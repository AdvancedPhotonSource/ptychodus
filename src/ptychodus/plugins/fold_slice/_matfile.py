"""Read a MATLAB ``.mat`` file of either vintage into one dict shape.

MATLAB has written two unrelated container formats under the same extension. Up to
``-v7`` a ``.mat`` is a bespoke binary stream that ``scipy.io.loadmat`` parses; from
``-v7.3`` it is HDF5, which that function refuses outright. Both are in circulation
because the switch is per-``save`` and old files do not migrate.

This module hides that split behind one call, so a reader sees the shape
``scipy.io.loadmat(..., simplify_cells=True)` produces whichever vintage it was
handed: structs as mappings, cells as lists, arrays squeezed, and a one-element
array as a plain Python scalar.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import logging

import h5py
import numpy
import scipy.io

logger = logging.getLogger(__name__)

__all__ = [
    'open_matlab_file',
]

# MATLAB stores a complex array as a compound of two same-width floating fields.
# Unverified against a MATLAB-written file: the names and their order are taken from
# the format description rather than observed, so an unrecognized compound is skipped
# rather than guessed at.
_COMPLEX_FIELD_NAMES: tuple[str, str] = ('real', 'imag')

# Bookkeeping groups holding the targets of object references and the serialized
# objects behind them. They are not workspace variables and never appear in what
# loadmat returns.
_PRIVATE_GROUP_NAMES = frozenset({'#refs#', '#subsystem#'})


def _is_metadata_key(key: str) -> bool:
    """Whether a name is the parser's own bookkeeping rather than a workspace variable."""
    return key.startswith('__') and key.endswith('__')


def _read_matlab_class(item: h5py.HLObject) -> str:
    """The MATLAB type name an object declares, or the empty string for the file root."""
    matlab_class = item.attrs.get('MATLAB_class', b'')

    if isinstance(matlab_class, bytes):
        return matlab_class.decode('utf-8', errors='replace')

    return str(matlab_class)


def _normalize(values: numpy.ndarray) -> Any:
    """Put an array into the orientation and rank ``loadmat`` would have produced.

    h5py hands back the dimensions in the reverse of MATLAB's order, so reversing all
    of them restores the original. Squeezing afterwards matches `squeeze_me`, and the
    order matters: a MATLAB row vector arrives as a column, which only reverses into a
    row while it still has both axes.
    """
    values = values.transpose()

    if values.size == 1:
        # `.item()` rather than a constructor because the Python type has to follow
        # the stored one -- a double becomes float, a logical bool, an int32 int --
        # which is what a one-element array does under `simplify_cells`.
        return values.reshape(()).item()

    return numpy.squeeze(values)


def _combine_complex(values: numpy.ndarray) -> numpy.ndarray | None:
    """Recombine a complex compound into a complex array, or None if it is not one."""
    if values.dtype.names != _COMPLEX_FIELD_NAMES:
        return None

    real = values['real']
    complex_dtypes: dict[numpy.dtype[Any], numpy.dtype[Any]] = {
        numpy.dtype(numpy.float32): numpy.dtype(numpy.complex64),
        numpy.dtype(numpy.float64): numpy.dtype(numpy.complex128),
    }
    complex_dtype = complex_dtypes.get(real.dtype)

    if complex_dtype is not None and values.dtype.itemsize == 2 * real.dtype.itemsize:
        # The two fields are adjacent and equally wide, which is bit-identical to a
        # complex of the same precision. Viewing avoids a temporary the size of the
        # probe or the object.
        return values.view(complex_dtype)

    return real + 1j * values['imag']


def _decode_char(values: numpy.ndarray) -> str | None:
    """Decode a MATLAB character array, or None when the codes are not characters."""
    try:
        return ''.join(chr(int(code)) for code in numpy.ravel(values))
    except (TypeError, ValueError):
        return None


def _dereference(h5_file: h5py.File, reference: h5py.Reference) -> Any:
    """Follow one object reference, or None for a null one."""
    if not reference:
        return None

    return _convert(h5_file, h5_file[reference])


def _convert_cell(h5_file: h5py.File, dataset: h5py.Dataset) -> Any:
    """Resolve a cell array to a list, or to its single element."""
    references = numpy.ravel(dataset[()].transpose())
    items = [_dereference(h5_file, reference) for reference in references]

    if len(items) == 1:
        return items[0]

    return items


def _convert_dataset(h5_file: h5py.File, dataset: h5py.Dataset) -> Any:
    """Convert one stored value to what ``loadmat`` would have returned for it."""
    matlab_class = _read_matlab_class(dataset)

    if 'MATLAB_empty' in dataset.attrs:
        # The body holds the dimensions rather than data, already in MATLAB's order.
        return numpy.empty(tuple(int(length) for length in numpy.ravel(dataset[()])))

    if h5py.check_ref_dtype(dataset.dtype) is not None:
        return _convert_cell(h5_file, dataset)

    values = dataset[()]

    if values.dtype.names is not None:
        combined = _combine_complex(values)

        if combined is None:
            logger.debug('Skipping "%s": unrecognized compound %s.', dataset.name, values.dtype)
            return None

        values = combined
    elif matlab_class == 'char':
        text = _decode_char(values)

        if text is not None:
            return text
    elif matlab_class == 'logical':
        values = values.astype(bool)

    return _normalize(values)


def _convert(h5_file: h5py.File, item: h5py.HLObject) -> Any:
    """Convert a stored group or dataset, or None for a type this does not model."""
    if isinstance(item, h5py.Group):
        matlab_class = _read_matlab_class(item)

        if matlab_class in ('', 'struct'):
            return _MatGroup(h5_file, item)

        logger.debug('Skipping "%s": unmodelled MATLAB class "%s".', item.name, matlab_class)
        return None

    return _convert_dataset(h5_file, item)


class _MatGroup(Mapping[str, Any]):
    """A MATLAB struct, or the file root, backed by an open HDF5 group.

    Conversion happens on access rather than up front. A reconstruction file stores
    its whole object and probe a second time inside `outputs`, alongside several
    serialized MATLAB objects, so materializing everything would double peak memory
    to produce values nothing asks for.
    """

    def __init__(self, h5_file: h5py.File, group: h5py.Group) -> None:
        self._h5_file = h5_file
        self._group = group

    def __getitem__(self, key: str) -> Any:
        return _convert(self._h5_file, self._group[key])

    def __iter__(self) -> Iterator[str]:
        for key in self._group:
            if key not in _PRIVATE_GROUP_NAMES:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


@contextmanager
def open_matlab_file(file_path: Path) -> Iterator[Mapping[str, Any]]:
    """Open a ``.mat`` file of either vintage as a mapping of workspace variables.

    The vintage is decided by the HDF5 signature rather than by letting the older
    parser fail, because the signature is a property of the file while the failure is
    a property of whichever library version is installed. The search accounts for the
    512-byte header MATLAB puts in front of the HDF5 payload.

    Yields:
        The workspace, as a mapping. It is valid only inside the context: a ``-v7.3``
        file is read lazily and its values come from a handle closed on exit.
    """
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, 'r') as h5_file:
            yield _MatGroup(h5_file, h5_file)
    else:
        contents = scipy.io.loadmat(file_path, simplify_cells=True)
        # The older parser reports the file header and the global list alongside the
        # workspace. Dropping them is what makes the two backends agree on iteration
        # and length, not only on lookup.
        yield {key: value for key, value in contents.items() if not _is_metadata_key(key)}
