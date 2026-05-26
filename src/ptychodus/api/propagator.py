"""Wavefield propagation models and associated parameter containers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from .common import ComplexArrayType, RealArrayType
from .geometry import PixelGeometry


def intensity(wavefield: ComplexArrayType) -> RealArrayType:
    """Return the element-wise intensity (|wavefield|²) of a complex array."""
    return numpy.square(numpy.absolute(wavefield))


@dataclass(frozen=True)
class PropagatorParameters:
    wavelength_m: float
    """Illumination wavelength in meters."""
    width_px: int
    """Number of pixels in the x-direction."""
    height_px: int
    """Number of pixels in the y-direction."""
    pixel_width_m: float
    """Source-plane pixel width in meters."""
    pixel_height_m: float
    """Source-plane pixel height in meters."""
    propagation_distance_m: float
    """Propagation distance in meters."""

    @property
    def dx(self) -> float:
        """Pixel width in wavelengths."""
        return self.pixel_width_m / self.wavelength_m

    @property
    def pixel_aspect_ratio(self) -> float:
        """Pixel aspect ratio (width / height)."""
        return self.pixel_width_m / self.pixel_height_m

    @property
    def z(self) -> float:
        """Propagation distance in wavelengths."""
        return self.propagation_distance_m / self.wavelength_m

    @property
    def fresnel_number(self) -> float:
        """Fresnel number."""
        return numpy.square(self.dx) / numpy.absolute(self.z)

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        JJ, II = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        XX = II - self.width_px // 2  # noqa: N806
        YY = JJ - self.height_px // 2  # noqa: N806
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = numpy.meshgrid(fy, fx, indexing='ij')  # noqa: N806
        return FY, FX


class Propagator(ABC):
    """Abstract interface for free-space wavefield propagators."""

    @abstractmethod
    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        pass


class AngularSpectrumPropagator(Propagator):
    """Exact propagator using the angular-spectrum transfer function; valid for all Fresnel numbers."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * numpy.pi * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)
        tf = numpy.exp(i2piz * numpy.sqrt(1 - ratio))

        self._transfer_function = numpy.where(ratio < 1, tf, 0)

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(Propagator):
    """Fresnel propagator using a paraxial transfer function in the frequency domain."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * numpy.pi * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)

        self._transfer_function = numpy.exp(i2piz * (1 - ratio / 2))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(Propagator):
    """Fresnel propagator using the direct Fresnel transform; changes pixel size between planes."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ipi = 1j * numpy.pi

        Fr = parameters.fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = Fr / (1j * ar)  # noqa: N806
        C1 = numpy.exp(2j * numpy.pi * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0
        self._B = numpy.exp(ipi * Fr * (numpy.square(XX) + numpy.square(YY / ar)))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))
        else:
            return self._B * fftshift(ifft2(ifftshift(wavefield * self._A)))


class FraunhoferPropagator(Propagator):
    """Far-field (Fraunhofer) propagator; valid when the Fresnel number is much less than one."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ipi = 1j * numpy.pi

        Fr = parameters.fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = Fr / (1j * ar)  # noqa: N806
        C1 = numpy.exp(2j * numpy.pi * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield)))
        else:
            return fftshift(ifft2(ifftshift(wavefield * self._A)))


@dataclass(frozen=True)
class PropagatedProbe:
    """Stack of probe wavefields at evenly-spaced free-space propagation distances,
    produced by :func:`propagate_probe`.

    Stores the complex wavefield as ``(num_steps, num_incoherent_modes, height_px,
    width_px)``. Per-step intensity (incoherent-mode sum of ``|wf|^2``) and the three
    orthogonal projections used by the GUI are derived lazily.
    """

    wavefield: ComplexArrayType
    begin_coordinate_m: float
    end_coordinate_m: float
    pixel_geometry: PixelGeometry

    @property
    def num_steps(self) -> int:
        return self.wavefield.shape[0]

    @property
    def num_incoherent_modes(self) -> int:
        return self.wavefield.shape[1]

    @property
    def height_px(self) -> int:
        return self.wavefield.shape[2]

    @property
    def width_px(self) -> int:
        return self.wavefield.shape[3]

    @property
    def intensity(self) -> RealArrayType:
        """Per-step intensity image: ``sum_modes |wavefield|^2``,
        shape ``(num_steps, height_px, width_px)``. Recomputed on each access;
        cache in a local if calling repeatedly in a hot loop."""
        return numpy.sum(intensity(self.wavefield), axis=1)

    def get_xy_projection(self, step: int) -> RealArrayType:
        return self.intensity[step]

    def get_zx_projection(self) -> RealArrayType:
        ints = self.intensity
        sz = ints.shape[-2]
        cut_l = ints[:, (sz - 1) // 2, :]
        cut_r = ints[:, sz // 2, :]
        return numpy.transpose(numpy.add(cut_l, cut_r) / 2)

    def get_zy_projection(self) -> RealArrayType:
        ints = self.intensity
        sz = ints.shape[-1]
        cut_l = ints[:, :, (sz - 1) // 2]
        cut_r = ints[:, :, sz // 2]
        return numpy.transpose(numpy.add(cut_l, cut_r) / 2)

    def save_npz(self, file_path: Path) -> None:
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            wavefield=self.wavefield,
            intensity=self.intensity,
            begin_coordinate_m=self.begin_coordinate_m,
            end_coordinate_m=self.end_coordinate_m,
            pixel_height_m=self.pixel_geometry.height_m,
            pixel_width_m=self.pixel_geometry.width_m,
        )


def propagate_probe(
    wavefield: ComplexArrayType,
    *,
    pixel_geometry: PixelGeometry,
    wavelength_m: float,
    begin_coordinate_m: float,
    end_coordinate_m: float,
    num_steps: int,
) -> PropagatedProbe:
    """Propagate a multi-mode probe through a slab of free space using the
    angular-spectrum propagator at ``num_steps`` evenly-spaced distances in
    ``[begin_coordinate_m, end_coordinate_m]``.

    Args:
        wavefield: Complex source-plane wavefield, shape ``(num_incoherent_modes,
            height_px, width_px)``. Each incoherent mode is propagated independently
            and the result preserves the mode axis.
        pixel_geometry: Source-plane pixel geometry (assumed constant across modes
            and propagation distances).
        wavelength_m: Illumination wavelength in meters.
        begin_coordinate_m: Smallest propagation distance in the output stack.
        end_coordinate_m: Largest propagation distance in the output stack.
        num_steps: Number of evenly-spaced steps along z.
    """
    if wavefield.ndim != 3:
        raise ValueError(
            f'wavefield must be 3-dimensional (modes, height, width); got ndim={wavefield.ndim}.'
        )

    num_modes, height_px, width_px = wavefield.shape

    propagated = numpy.zeros((num_steps, num_modes, height_px, width_px), dtype=wavefield.dtype)
    distance_m = numpy.linspace(begin_coordinate_m, end_coordinate_m, num_steps)

    for idx, z_m in enumerate(distance_m):
        params = PropagatorParameters(
            wavelength_m=wavelength_m,
            width_px=width_px,
            height_px=height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            propagation_distance_m=float(z_m),
        )
        propagator = AngularSpectrumPropagator(params)
        for mode in range(num_modes):
            propagated[idx, mode, :, :] = propagator.propagate(wavefield[mode, :, :])

    return PropagatedProbe(
        wavefield=propagated,
        begin_coordinate_m=begin_coordinate_m,
        end_coordinate_m=end_coordinate_m,
        pixel_geometry=pixel_geometry,
    )
