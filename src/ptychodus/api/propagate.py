"""Wavefield propagation models and associated parameter containers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy
import numpy.typing

from .constants import TWO_PI_J
from .typing import ComplexArrayType, RealArrayType
from .geometry import ImageExtent, PixelGeometry

_InexactT = TypeVar('_InexactT', bound=numpy.inexact[Any])


def intensity(wavefield: ComplexArrayType) -> RealArrayType:
    """Return the element-wise intensity (``|wavefield|²``) of a complex array."""
    return numpy.square(numpy.absolute(wavefield))


def _cut_central_row(
    array: numpy.typing.NDArray[_InexactT],
) -> numpy.typing.NDArray[_InexactT]:
    """Average the two central rows of a ``(num_steps, height, width)`` stack, then
    transpose to ``(width, num_steps)``.

    For an odd height the two indices coincide and nothing is averaged. The average is
    taken on whatever was passed in, so an intensity stack averages powers while a
    complex wavefield averages phasors.
    """
    size = array.shape[-2]
    cut_lower = array[:, (size - 1) // 2, :]
    cut_upper = array[:, size // 2, :]
    return numpy.transpose(numpy.add(cut_lower, cut_upper) / 2)


def _cut_central_column(
    array: numpy.typing.NDArray[_InexactT],
) -> numpy.typing.NDArray[_InexactT]:
    """Average the two central columns of a ``(num_steps, height, width)`` stack, then
    transpose to ``(height, num_steps)``. See :func:`_cut_central_row`."""
    size = array.shape[-1]
    cut_lower = array[:, :, (size - 1) // 2]
    cut_upper = array[:, :, size // 2]
    return numpy.transpose(numpy.add(cut_lower, cut_upper) / 2)


def compute_far_field_pixel_geometry(
    pixel_geometry: PixelGeometry,
    extent: ImageExtent,
    *,
    wavelength_m: float,
    propagation_distance_m: float,
) -> PixelGeometry:
    """Pixel geometry of the conjugate plane under the Fraunhofer relation
    ``dx_out = lambda |z| / (N dx_in)``.

    The single-FFT propagators (:class:`FresnelTransformPropagator`,
    :class:`FraunhoferPropagator`) map a plane of pitch ``dx_in`` onto a plane of this
    pitch. The relation is its own inverse: applying it twice returns *pixel_geometry*.

    Raises:
        ZeroDivisionError: when the input plane has zero extent along either axis.
            Callers that must degrade gracefully catch it rather than receiving an
            invented sentinel.
    """
    # Python-float arithmetic throughout: a numpy intermediate would yield inf instead
    # of the ZeroDivisionError that callers such as ProductGeometry rely on.
    numerator_m2 = wavelength_m * abs(propagation_distance_m)
    return PixelGeometry(
        width_m=numerator_m2 / (extent.width_px * pixel_geometry.width_m),
        height_m=numerator_m2 / (extent.height_px * pixel_geometry.height_m),
    )


@dataclass(frozen=True)
class PropagatorParameters:
    """Geometric parameters for a wavefield propagator: wavelength, extent, pixel size, and distance."""

    wavelength_m: float
    """Illumination wavelength in meters."""
    width_px: int
    """Number of pixels in the x-direction."""
    height_px: int
    """Number of pixels in the y-direction."""
    pixel_width_m: float
    """Upstream-plane pixel width in meters.

    For the single-FFT propagators this is the plane at the smaller z: the *input*
    plane when ``propagation_distance_m`` is positive, and the *output* plane when it
    is negative. The opposite plane has pitch ``lambda |z| / (width_px * pixel_width_m)``
    -- see :func:`compute_far_field_pixel_geometry`. Both grids are the same for the
    pitch-preserving propagators (:class:`AngularSpectrumPropagator`,
    :class:`FresnelTransferFunctionPropagator`), so the distinction is moot there.
    """
    pixel_height_m: float
    """Upstream-plane pixel height in meters; see :attr:`pixel_width_m`."""
    propagation_distance_m: float
    """Propagation distance in meters. Negative propagates backward, which for the
    single-FFT propagators is the exact inverse of the forward operator built from the
    same parameters."""

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
    def pixel_fresnel_number(self) -> float:
        """Signed per-pixel Fresnel number ``dx^2 / (lambda z)``.

        Signed because the propagator phase terms must conjugate when the propagation
        direction reverses; take :func:`numpy.absolute` where only the magnitude is
        meant. Width-only because the propagator algebra carries the y-axis separately
        through :attr:`pixel_aspect_ratio`.

        Distinct from ``ProductGeometry.fresnel_number``, which is the full-aperture
        number ``W H / (lambda z)``; the two differ by a factor of ``width_px *
        height_px * (dy / dx)``.
        """
        return numpy.square(self.dx) / self.z

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

        i2piz = TWO_PI_J * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)
        tf = numpy.exp(i2piz * numpy.sqrt(numpy.maximum(1 - ratio, 0.0)))

        # ifftshift matches the centered frequency grid to the unshifted spectrum in propagate
        self._transfer_function = ifftshift(numpy.where(ratio < 1, tf, 0))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(Propagator):
    """Fresnel propagator using a paraxial transfer function in the frequency domain."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = TWO_PI_J * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)

        # ifftshift matches the centered frequency grid to the unshifted spectrum in propagate
        self._transfer_function = ifftshift(numpy.exp(i2piz * (1 - ratio / 2)))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(Propagator):
    """Fresnel propagator using the direct Fresnel transform; changes pixel size between planes.

    The output plane has pitch ``lambda |z| / (N dx)`` -- see
    :func:`compute_far_field_pixel_geometry`. A negative propagation distance selects the
    exact inverse of the forward operator built from the same parameters, in which case
    :attr:`PropagatorParameters.pixel_width_m` describes the *output* plane.
    Retains the input quadratic phase that :class:`FraunhoferPropagator` drops, so it
    stays accurate outside the far field.

    The dropped/retained quadratic phase spans ``X_max = N / 2``, so the controlling
    quantity for far-field validity is ``N^2 * pixel_fresnel_number``, not the pixel
    Fresnel number alone. For N=256 that makes the honest condition ``Fr << 1.5e-5``.
    """

    def __init__(self, parameters: PropagatorParameters) -> None:
        if parameters.propagation_distance_m == 0.0:
            raise ValueError(
                'FresnelTransformPropagator requires a nonzero propagation distance; '
                'the output pixel size lambda*z/(N*dx) vanishes at z=0. Use '
                'AngularSpectrumPropagator, which is the identity there.'
            )

        ipi = 1j * numpy.pi

        # Signed: C2 and _B are pure phases that must conjugate when the direction
        # reverses, which is what makes the backward branch the forward branch's inverse.
        Fr = parameters.pixel_fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        # Magnitude only: a signed amplitude prefactor would flip the sign of the
        # recovered field on the backward branch.
        C0 = numpy.absolute(Fr) / (1j * ar)  # noqa: N806
        C1 = numpy.exp(TWO_PI_J * parameters.z)  # noqa: N806
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
    """Far-field (Fraunhofer) propagator: :class:`FresnelTransformPropagator` with the
    input quadratic phase ``exp(i pi Fr (X^2 + Y^2))`` dropped.

    The dropped/retained quadratic phase spans ``X_max = N / 2``, so the controlling
    quantity for far-field validity is ``N^2 * pixel_fresnel_number``, not the pixel
    Fresnel number alone. For N=256 that makes the honest condition ``Fr << 1.5e-5``.

    Shares the pitch and direction conventions of :class:`FresnelTransformPropagator`.
    """

    def __init__(self, parameters: PropagatorParameters) -> None:
        if parameters.propagation_distance_m == 0.0:
            raise ValueError(
                'FraunhoferPropagator requires a nonzero propagation distance; '
                'the output pixel size lambda*z/(N*dx) vanishes at z=0. Use '
                'AngularSpectrumPropagator, which is the identity there.'
            )

        ipi = 1j * numpy.pi

        # Signed phase, magnitude-only prefactor -- see FresnelTransformPropagator.
        Fr = parameters.pixel_fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = numpy.absolute(Fr) / (1j * ar)  # noqa: N806
        C1 = numpy.exp(TWO_PI_J * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield)))
        else:
            return fftshift(ifft2(ifftshift(wavefield * self._A)))


def choose_propagator(parameters: PropagatorParameters) -> tuple[Propagator, PixelGeometry]:
    """Select the correctly-sampled propagator and report the plane its output lives on.

    The single-FFT and transfer-function families sample opposite regimes, and each
    carries an implicit output grid, so the choice cannot be hidden inside either class:
    swapping silently would return a field on a plane the caller did not ask for. The
    returned :class:`PixelGeometry` is therefore part of the answer.

    Selection is per-axis with the conservative outcome, so an anisotropic geometry is
    never aliased on its narrow axis:

    - far-field pitch <= source pitch on *both* axes -> :class:`AngularSpectrumPropagator`
      on the source grid;
    - otherwise -> :class:`FresnelTransformPropagator` on the far-field grid.

    For square pixels this is the scalar condition ``|pixel_fresnel_number| <= 1 / N``.
    The two methods are numerically interchangeable at that crossover (measured
    agreement 2.5e-08); away from it they disagree by tens of percent, but that is grid
    disagreement rather than physics.

    At ``z = 0`` the far-field pitch is zero, so this selects angular spectrum on the
    source grid -- the identity for any geometry without evanescent modes. Zero distance
    is thus resolved by construction rather than by special case.

    Two propagators are never selected. :class:`FraunhoferPropagator` is
    :class:`FresnelTransformPropagator` with the input quadratic phase dropped, so the
    latter is strictly more accurate for one extra multiply.
    :class:`FresnelTransferFunctionPropagator` is the paraxial approximation of angular
    spectrum over the same grid.
    """
    source_pixel_geometry = PixelGeometry(
        width_m=parameters.pixel_width_m, height_m=parameters.pixel_height_m
    )
    extent = ImageExtent(width_px=parameters.width_px, height_px=parameters.height_px)
    far_field_pixel_geometry = compute_far_field_pixel_geometry(
        source_pixel_geometry,
        extent,
        wavelength_m=parameters.wavelength_m,
        propagation_distance_m=parameters.propagation_distance_m,
    )

    if (
        far_field_pixel_geometry.width_m <= source_pixel_geometry.width_m
        and far_field_pixel_geometry.height_m <= source_pixel_geometry.height_m
    ):
        return AngularSpectrumPropagator(parameters), source_pixel_geometry

    return FresnelTransformPropagator(parameters), far_field_pixel_geometry


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
        return _cut_central_row(self.intensity)

    def get_zy_projection(self) -> RealArrayType:
        return _cut_central_column(self.intensity)

    def get_xy_wavefield(self, step: int, mode: int) -> ComplexArrayType:
        """Complex wavefield of a single incoherent mode at one propagation step,
        shape ``(height_px, width_px)``.

        The mode-summed counterpart is :meth:`get_xy_projection`. Summing over
        mutually incoherent modes is only meaningful in intensity, so the per-mode
        accessors are the only way to reach the phase.
        """
        return self.wavefield[step, mode]

    def get_zx_wavefield(self, mode: int) -> ComplexArrayType:
        """Complex ZX plane of a single incoherent mode, shape ``(width_px, num_steps)``."""
        return _cut_central_row(self.wavefield[:, mode])

    def get_zy_wavefield(self, mode: int) -> ComplexArrayType:
        """Complex ZY plane of a single incoherent mode, shape ``(height_px, num_steps)``."""
        return _cut_central_column(self.wavefield[:, mode])

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
