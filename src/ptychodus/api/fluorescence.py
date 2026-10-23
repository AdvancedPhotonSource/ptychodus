"""Fluorescence data structures and enhancement plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from scipy.sparse.linalg import LinearOperator
import numpy

from .typing import RealArrayType
from .files import FileReader, FileWriter
from .interpolate import BarycentricArrayInterpolator
from .product import Product


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
    """Enhancement result yielded at each step: updated dataset and progress."""

    dataset: FluorescenceDataset
    progress: int = 0


class FluorescenceEnhancer(ABC):
    """Abstract interface for algorithms that enhance fluorescence datasets using ptychography."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_progress_goal(self) -> int:
        pass

    @abstractmethod
    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        pass


class FluorescenceFileReader(FileReader[FluorescenceDataset]):
    """Plugin interface for reading fluorescence datasets."""


class FluorescenceFileWriter(FileWriter[FluorescenceDataset]):
    """Plugin interface for writing fluorescence datasets."""


class UpscalingStrategy(ABC):
    """Remaps element concentrations from the regular scan grid to the upscaled grid using ptychography-corrected probe positions."""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass


class DeconvolutionStrategy(ABC):
    """Deconvolves the probe kernel from the accumulated array to enhance the element map."""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass
