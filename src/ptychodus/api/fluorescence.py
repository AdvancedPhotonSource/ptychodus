"""Fluorescence data structures and enhancement plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from pydantic import BaseModel
from scipy.sparse.linalg import LinearOperator
import numpy

from .typing import RealArrayType
from .files import FileReader, FileWriter
from .interpolate import BarycentricArrayInterpolator
from .product import Product
from .reconstruct import DeviceEnvironment


class VSPILinearOperator(LinearOperator):
    def __init__(self, product: Product) -> None:
        """
        M: number of XRF positions
        N: number of ptychography object pixels
        P: number of XRF channels

        A[M,N] * X[N,P] = B[M,P]
        """
        object_geometry = product.object_.get_geometry()
        M = len(product.probe_positions)  # noqa: N806
        N = object_geometry.height_px * object_geometry.width_px  # noqa: N806
        super().__init__(float, (M, N))
        self._product = product

    def _matvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = x.reshape((object_geometry.height_px, object_geometry.width_px))
        AX = numpy.zeros(len(self._product.probe_positions))  # noqa: N806
        interpolator = BarycentricArrayInterpolator(object_array)

        for index, (scan_point, probe) in enumerate(self._product.iter_position_probes()):
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            patch = interpolator.get_patch(
                object_point.x_px,
                object_point.y_px,
                psf.shape[-1],
                psf.shape[-2],
            )
            AX[index] = numpy.sum(psf * patch)

        return AX

    def _rmatvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = numpy.zeros((object_geometry.height_px, object_geometry.width_px))
        interpolator = BarycentricArrayInterpolator(object_array)

        for index, (scan_point, probe) in enumerate(self._product.iter_position_probes()):
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            interpolator.add_patch(
                object_point.x_px,
                object_point.y_px,
                x[index] * psf,
            )

        HX = object_array.flatten()  # noqa: N806

        return HX


@dataclass(frozen=True)
class ElementMap:
    """2D spatial map of fluorescence signal for a single element in counts per second."""

    name: str
    counts_per_second: RealArrayType

    @property
    def nbytes(self) -> int:
        return self.counts_per_second.nbytes


@dataclass(frozen=True)
class FluorescenceDataset(Sequence[ElementMap]):
    """Collection of element maps with metadata."""

    _element_maps: Sequence[ElementMap]
    counts_per_second_path: str
    channel_names_path: str

    # TODO need to communicate association between element map pixels and scan order.
    #      integer-valued, same shape as counts_per_second
    # scan_indexes: IntegerArray

    @property
    def nbytes(self) -> int:
        return sum(element_map.nbytes for element_map in self._element_maps)

    @overload
    def __getitem__(self, index: int) -> ElementMap: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ElementMap]: ...

    def __getitem__(self, index: int | slice) -> ElementMap | Sequence[ElementMap]:
        return self._element_maps[index]

    def __len__(self) -> int:
        return len(self._element_maps)

    def __repr__(self) -> str:
        names = ', '.join(element_map.name for element_map in self._element_maps)
        return f'{type(self).__name__}({len(self)} maps: {names})'


@dataclass(frozen=True)
class FluorescenceEnhancerInput:
    """All data required to start a fluorescence enhancement: dataset and ptychography product."""

    dataset: FluorescenceDataset
    product: Product


@dataclass(frozen=True)
class FluorescenceEnhancerOutput:
    """Enhancement result yielded at each step: updated dataset and progress.

    ``progress_goal`` is what ``progress`` counts towards, reported by the run
    rather than predicted before it: how far an enhancement has to go depends on
    the options it was handed and the dataset it is working through. Zero means
    unknown, which a progress bar reads as indeterminate.
    """

    dataset: FluorescenceDataset
    progress: int = 0
    progress_goal: int = 0


class FluorescenceEnhancer(ABC):
    """An algorithm that enhances a fluorescence dataset using ptychography, and its options.

    Holds no settings of its own and is configured per call, so two callers can
    drive the same one with different values. It does not name itself either;
    the name a person selects it by is the one it was registered under.

    An enhancer knows nothing about where its option values come from. Its
    options may state what a field *is* -- its unit, its bounds -- but never
    where a value for it is found.
    """

    @abstractmethod
    def get_default_options(self) -> BaseModel:
        """Return this enhancer's options, populated with their defaults.

        A description, not live state: plain pydantic carrying values, types,
        ranges, enums and the words to put beside each field. A caller edits a
        copy and passes it back as ``options``, and its type is what an incoming
        ``options`` is validated against.

        An instance rather than the class, because a field declared with a
        ``default_factory`` reports its class-level default as undefined. Only an
        instance carries the value such a field actually starts at.
        """

    @abstractmethod
    def enhance(
        self, input_: FluorescenceEnhancerInput, *, options: BaseModel
    ) -> Iterator[FluorescenceEnhancerOutput]:
        pass


class SubprocessFluorescenceEnhancer(FluorescenceEnhancer):
    """An enhancer whose work runs in a spawned child rather than in this process.

    Carries only where that child may run. Placement says nothing about the
    algorithm, so it is neither declared in the options nor passed through
    :meth:`enhance`; whatever owns the device selection attaches it after
    construction.
    """

    def __init__(self) -> None:
        super().__init__()
        # Where the work runs, which is the spawning process's to decide and
        # not the backend's. Replaced after construction by whatever owns the
        # selection; until then a child sees every device this process does.
        self.device_environment: DeviceEnvironment = dict


class FluorescenceFileReader(FileReader[FluorescenceDataset]):
    """Plugin interface for reading fluorescence datasets."""


class FluorescenceFileWriter(FileWriter[FluorescenceDataset]):
    """Plugin interface for writing fluorescence datasets."""
