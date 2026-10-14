"""Hand-built diffraction arrays and datasets shared by the assemble-layer tests.

``test_assemble`` and ``test_diffraction_summary`` both drive the pure assembly
layer over synthetic arrays, so the fakes live here rather than being forked
per module. The two differ only in frame shape, which is a parameter below.

Every fake implements :class:`DiffractionArray` on its real signature,
``read_region`` keyword included. That matters: ``mypy`` runs over
``src/ptychodus`` and ``scripts`` but not ``tests``, so an override that quietly
drops the keyword is caught by nothing until a crop path calls it.
"""

from __future__ import annotations

from collections.abc import Sequence
import threading
import time

import numpy

from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionArray,
    DiffractionDatasetLayoutNode,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatterns,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry

GEOMETRY = PixelGeometry(width_m=75e-6, height_m=75e-6)

# Long enough that a correctly released gate never reaches it, short enough that a
# test which forgets to release still ends. It is a deadlock guard, not a
# synchronization mechanism -- see BlockingArray.timed_out.
GATE_TIMEOUT_SEC = 10.0


def make_array(
    label: str, first_index: int, num_patterns: int, fill: int, *, frame_shape: tuple[int, int]
) -> SimpleDiffractionArray:
    """A constant-valued array whose indexes run from `first_index`."""
    height, width = frame_shape
    patterns = numpy.full((num_patterns, height, width), fill, dtype=numpy.int32)
    indexes = numpy.arange(first_index, first_index + num_patterns, dtype=numpy.intp)
    return SimpleDiffractionArray(label, indexes, patterns)


def _resolve_pattern_dtype(
    arrays: Sequence[DiffractionArray], pattern_dtype: numpy.typing.DTypeLike | None
) -> numpy.dtype:
    if pattern_dtype is not None:
        return numpy.dtype(pattern_dtype)

    return arrays[0].get_patterns().dtype if arrays else numpy.dtype(numpy.uint16)


def make_dataset(
    arrays: Sequence[DiffractionArray],
    frame_shape: tuple[int, int],
    bad_pixels: BadPixels | None = None,
    num_patterns_per_array: Sequence[int] | None = None,
    pixel_geometry: PixelGeometry | None = GEOMETRY,
    exposure_time_s: float | None = None,
    pattern_dtype: numpy.typing.DTypeLike | None = None,
) -> SimpleDiffractionDataset:
    """Build a dataset over `arrays`, with `frame_shape` as the detector extent.

    `pattern_dtype` is read off `arrays[0]` when it is not given. Pass it
    explicitly whenever `arrays[0]` is a :class:`BlockingArray`, so that building
    the metadata does not trip the gate before the test has armed it.
    """
    height, width = frame_shape
    metadata = DiffractionMetadata(
        num_patterns_per_array=(
            [a.get_num_patterns() for a in arrays]
            if num_patterns_per_array is None
            else list(num_patterns_per_array)
        ),
        pattern_dtype=_resolve_pattern_dtype(arrays, pattern_dtype),
        detector_extent=ImageExtent(width_px=width, height_px=height),
        detector_pixel_geometry=pixel_geometry,
        exposure_time_s=exposure_time_s,
    )
    return SimpleDiffractionDataset(
        metadata, DiffractionDatasetLayoutNode.create_root(), arrays, bad_pixels
    )


class FailingArray(DiffractionArray):
    """An array whose read raises, to exercise the error and skip paths."""

    def __init__(self, label: str, error: BaseException, num_patterns: int = 2) -> None:
        self._label = label
        self._error = error
        self._num_patterns = num_patterns

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return numpy.arange(self._num_patterns, dtype=numpy.intp)

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        raise self._error

    def get_num_patterns(self) -> int:
        return self._num_patterns


class BlockingArray(DiffractionArray):
    """Parks `inner`'s read on `gate`, to hold a worker at a known point in the fan-out.

    `entered` is set once the read is parked, so a test can wait until a worker is
    definitely inside this array before acting on that fact. `release_delay_sec`
    then keeps the worker busy for a beat after the gate opens, giving whoever
    opened it time to finish before this worker moves on to the next array.
    """

    def __init__(
        self,
        inner: DiffractionArray,
        gate: threading.Event,
        *,
        release_delay_sec: float = 0.0,
    ) -> None:
        self._inner = inner
        self._gate = gate
        self._release_delay_sec = release_delay_sec
        self.entered = threading.Event()
        self.timed_out = False

    def get_label(self) -> str:
        return self._inner.get_label()

    def get_indexes(self) -> DiffractionIndexes:
        return self._inner.get_indexes()

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        self.entered.set()

        # A test whose gate never opens should fail on `timed_out` rather than hang.
        if not self._gate.wait(timeout=GATE_TIMEOUT_SEC):
            self.timed_out = True

        if self._release_delay_sec > 0.0:
            time.sleep(self._release_delay_sec)

        return self._inner.get_patterns(read_region=read_region)

    def get_num_patterns(self) -> int:
        return self._inner.get_num_patterns()
