from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Final

import h5py
import numpy

from .common import BYTES_PER_MEGABYTE
from .diffraction import (
    BadPixels,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
)


class StandardFileLayout(str, Enum):
    DIFFRACTION = 'diffraction.h5'
    FLUORESCENCE_IN = 'fluorescence-in.h5'
    FLUORESCENCE_OUT = 'fluorescence-out.h5'
    PRODUCT_IN = 'product-in.h5'
    PRODUCT_OUT = 'product-out.h5'
    SETTINGS = 'settings.ini'


class AssembledDiffractionData:
    PATTERNS_KEY: Final[str] = 'patterns'
    INDEXES_KEY: Final[str] = 'indexes'
    BAD_PIXELS_KEY: Final[str] = 'bad_pixels'

    def __init__(
        self, indexes: DiffractionIndexes, patterns: DiffractionPatterns, bad_pixels: BadPixels
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._bad_pixels = bad_pixels

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if bad_pixels.ndim != 2:
            raise ValueError(
                f'Unexpected number of dimensions for bad pixels! (actual={bad_pixels.ndim} expected=2)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if patterns.shape[1:] != bad_pixels.shape:
            raise ValueError('Patterns shape does not match bad pixels shape!')

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=int),
            patterns=numpy.zeros((1, 1, 1), dtype=int),
            bad_pixels=numpy.zeros((1, 1), dtype=bool),
        )

    def read_from_file(self, file_path: Path) -> None:
        with h5py.File(file_path, 'r') as h5_file:
            h5_indexes = h5_file[self.INDEXES_KEY]

            if not isinstance(h5_indexes, h5py.Dataset):
                raise ValueError('Indexes are not a dataset!')

            h5_patterns = h5_file[self.PATTERNS_KEY]

            if not isinstance(h5_patterns, h5py.Dataset):
                raise ValueError('Patterns are not a dataset!')

            h5_bad_pixels = h5_file[self.BAD_PIXELS_KEY]

            if not isinstance(h5_bad_pixels, h5py.Dataset):
                raise ValueError('Bad pixels are not a dataset!')

            self._indexes = h5_indexes[()]
            self._patterns = h5_patterns[()]  # TODO support memmap
            self._bad_pixels = h5_bad_pixels[()]

    def write_to_file(self, file_path: Path, compression: str = 'lzf') -> None:
        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset(self.INDEXES_KEY, data=self._indexes, compression=compression)
            h5_file.create_dataset(self.PATTERNS_KEY, data=self._patterns, compression=compression)
            h5_file.create_dataset(
                self.BAD_PIXELS_KEY, data=self._bad_pixels, compression=compression
            )

    def get_patterns_shape(self) -> tuple[int, int, int]:
        return self._patterns.shape

    def get_patterns_dtype(self) -> DiffractionPatternDType:
        return self._patterns.dtype

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._patterns[index]

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = self._indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = self._patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            bad_pixels=data._bad_pixels,
        )

    def get_assembled_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_assembled_patterns(self) -> DiffractionPatterns:
        return self._patterns[self._indexes >= 0]

    def get_assembled_pattern_counts(self) -> DiffractionPatternCounts:
        good_pixels = numpy.logical_not(self._bad_pixels)
        assembled_patterns = self.get_assembled_patterns()
        pattern_counts = numpy.sum(assembled_patterns[:, good_pixels], axis=-1)
        return pattern_counts

    def get_average_pattern(self) -> DiffractionPattern:
        assembled_patterns = self.get_assembled_patterns()
        return numpy.mean(assembled_patterns, axis=0)

    def __str__(self) -> str:
        number, height, width = self._patterns.shape
        dtype = str(self._patterns.dtype)
        size_MB = self._patterns.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{number} x {height}H x {width}W {dtype} [{size_MB:.2f}MB]'
