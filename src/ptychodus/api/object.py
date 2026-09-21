"""Object (transmission function) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import logging
import math

import numpy
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

from .typing import ComplexArrayType, RealArrayType
from .constants import format_length
from .fourier import fourier_shift_2d
from .geometry import PixelGeometry
from .probe import ProbeGeometry
from .probe_positions import ProbePosition, calculate_scan_geometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectCenter:
    """Physical center coordinates of the object array in meters."""

    x_m: float
    y_m: float

    def copy(self) -> ObjectCenter:
        return ObjectCenter(
            x_m=float(self.x_m),
            y_m=float(self.y_m),
        )


@dataclass(frozen=True)
class ObjectPosition:
    """Position expressed in object pixel coordinates."""

    index: int
    x_px: float
    y_px: float


@dataclass(frozen=True)
class ObjectTransverseCoordinates:
    """2D Cartesian coordinate arrays for the transverse plane of the object, in meters."""

    x_m: RealArrayType
    y_m: RealArrayType


@dataclass(frozen=True)
class ObjectGeometry:
    """Spatial geometry of the object: size, pixel scale, and center."""

    width_px: int
    height_px: int
    pixel_width_m: float
    pixel_height_m: float
    center_x_m: float
    center_y_m: float

    @property
    def width_m(self) -> float:
        return self.width_px * self.pixel_width_m

    @property
    def height_m(self) -> float:
        return self.height_px * self.pixel_height_m

    @property
    def minimum_x_m(self) -> float:
        return self.center_x_m - self.width_m / 2.0

    @property
    def minimum_y_m(self) -> float:
        return self.center_y_m - self.height_m / 2.0

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.pixel_width_m,
            height_m=self.pixel_height_m,
        )

    def get_center(self) -> ObjectCenter:
        return ObjectCenter(
            x_m=self.center_x_m,
            y_m=self.center_y_m,
        )

    def get_transverse_coordinates(self) -> ObjectTransverseCoordinates:
        Y, X = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        x_px = X - (self.width_px - 1) / 2
        y_px = Y - (self.height_px - 1) / 2
        return ObjectTransverseCoordinates(
            x_m=x_px * self.pixel_width_m, y_m=y_px * self.pixel_height_m
        )

    def map_coordinates_object_to_probe(self, position: ObjectPosition) -> ProbePosition:
        # Centered-pixel convention: the world center sits at pixel index (N-1)/2,
        # matching get_transverse_coordinates above.
        rx_px = (self.width_px - 1) / 2
        ry_px = (self.height_px - 1) / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_m = self.center_x_m + dx_m * (position.x_px - rx_px)
        y_m = self.center_y_m + dy_m * (position.y_px - ry_px)

        return ProbePosition(position.index, x_m, y_m)

    def map_coordinates_probe_to_object(self, position: ProbePosition) -> ObjectPosition:
        rx_px = (self.width_px - 1) / 2
        ry_px = (self.height_px - 1) / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_px = (position.x_m - self.center_x_m) / dx_m + rx_px
        y_px = (position.y_m - self.center_y_m) / dy_m + ry_px

        return ObjectPosition(position.index, x_px, y_px)

    def contains(self, geometry: ObjectGeometry) -> bool:
        dx = self.center_x_m - geometry.center_x_m
        dy = self.center_y_m - geometry.center_y_m
        dw = self.width_m - geometry.width_m
        dh = self.height_m - geometry.height_m
        return abs(dx) <= dw and abs(dy) <= dh

    def __str__(self) -> str:
        return (
            f'{self.width_px} x {self.height_px} px around '
            f'({format_length(self.center_x_m)}, {format_length(self.center_y_m)})'
        )


class ObjectGeometryProvider(ABC):
    """Interface for classes that provide object geometry."""

    @abstractmethod
    def get_probe_positions(self) -> Sequence[ProbePosition]:
        pass

    @abstractmethod
    def get_object_geometry(self) -> ObjectGeometry:
        pass


class Object:
    """Complex transmission function stored as a (layers, height, width) array with spatial metadata."""

    def __init__(
        self,
        array: ComplexArrayType | None,
        pixel_geometry: PixelGeometry | None,
        center: ObjectCenter | None,
        layer_spacing_m: Sequence[float] = [],
    ) -> None:
        if array is None:
            self._array: ComplexArrayType = numpy.zeros((1, 0, 0), dtype=complex)
        elif numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, ...]
                case 3:
                    self._array = array
                case _:
                    raise ValueError('Object must be 2- or 3-dimensional ndarray.')
        else:
            raise TypeError('Object must be a complex-valued ndarray')

        self._pixel_geometry = pixel_geometry
        self._center = center
        self._layer_spacing_m = layer_spacing_m

        expected_layers = self._array.shape[-3]
        actual_layers = len(layer_spacing_m) + 1

        if actual_layers != expected_layers:
            raise ValueError(f'Expected {expected_layers} layers; got {actual_layers}!')

    def copy(self) -> Object:
        return Object(
            array=self._array.copy(),
            pixel_geometry=None if self._pixel_geometry is None else self._pixel_geometry.copy(),
            center=None if self._center is None else self._center.copy(),
            layer_spacing_m=list(self._layer_spacing_m),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    @property
    def width_px(self) -> int:
        return self._array.shape[-1]

    @property
    def height_px(self) -> int:
        return self._array.shape[-2]

    @property
    def num_layers(self) -> int:
        return self._array.shape[-3]

    def get_pixel_geometry(self) -> PixelGeometry:
        if self._pixel_geometry is None:
            raise ValueError('Missing object pixel geometry!')

        return self._pixel_geometry

    def get_center(self) -> ObjectCenter:
        if self._center is None:
            raise ValueError('Missing object center!')

        return self._center

    def get_geometry(self) -> ObjectGeometry:
        pixel_geometry = self.get_pixel_geometry()
        center = self.get_center()

        return ObjectGeometry(
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            center_x_m=center.x_m,
            center_y_m=center.y_m,
        )

    def get_layer(self, number: int) -> ComplexArrayType:
        return self._array[number, :, :]

    def get_layers_flattened(self) -> ComplexArrayType:
        return numpy.prod(self._array, axis=-3)

    @property
    def layer_spacing_m(self) -> Sequence[float]:
        return self._layer_spacing_m

    def get_total_thickness_m(self) -> float:
        return sum(self._layer_spacing_m)

    def __repr__(self) -> str:
        return f'{self._array.dtype}{self._array.shape}'


class RegistrationQuantity(Enum):
    """Quantity derived from a complex object array and handed to cross-correlation.

    Each member records the two properties that decide when it is usable.
    :attr:`is_shift_exact` marks a quantity that is a *linear* functional of the
    object and therefore exactly shift-equivariant, so the recovered sub-pixel
    shift is limited only by ``upsample_factor``; every pointwise-nonlinear
    quantity distorts sub-pixel interpolation and biases the estimate a little.
    :attr:`is_ramp_invariant` marks a quantity that does not change when a linear
    phase ramp is applied to the object, which is what makes it usable to
    bootstrap registration *before* the ramp ambiguity has been standardized
    away.

    Beyond those two flags: ``AMPLITUDE`` is the historical behavior and is nearly
    blind for a phase-contrast sample, whose transmission modulus is almost flat.
    ``PHASOR`` and ``PHASE`` discard amplitude contrast entirely and are undefined
    where the object vanishes. ``VARIATION`` -- the magnitude of the mean-subtracted
    complex logarithmic gradient -- buys its ramp invariance at the cost of a
    sub-pixel bias of order 0.1 px from the rectifying magnitude.
    """

    COMPLEX = auto()
    VARIATION = auto()
    PHASOR = auto()
    PHASE = auto()
    AMPLITUDE = auto()

    @property
    def is_ramp_invariant(self) -> bool:
        return self is RegistrationQuantity.VARIATION

    @property
    def is_shift_exact(self) -> bool:
        return self is RegistrationQuantity.COMPLEX

    def extract(self, array: ComplexArrayType) -> numpy.ndarray:
        """Map a complex object array onto the quantity used for registration."""
        values = numpy.asarray(array)

        match self:
            case RegistrationQuantity.COMPLEX:
                return values
            case RegistrationQuantity.AMPLITUDE:
                return numpy.absolute(values)
            case RegistrationQuantity.PHASE:
                return numpy.angle(values)
            case RegistrationQuantity.PHASOR:
                magnitude = numpy.absolute(values)
                return numpy.divide(
                    values,
                    magnitude,
                    out=numpy.zeros_like(values, dtype=numpy.complex128),
                    where=magnitude > 0.0,
                )
            case RegistrationQuantity.VARIATION:
                gradient_y, gradient_x = _logarithmic_gradient(values)
                return numpy.hypot(
                    numpy.absolute(gradient_y - gradient_y.mean()),
                    numpy.absolute(gradient_x - gradient_x.mean()),
                )


def _logarithmic_gradient(values: ComplexArrayType) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Forward differences of ``log(O)`` along each axis, computed wrap-free.

    ``log(O[i + 1] / O[i])`` is the log-amplitude difference plus the phase
    difference wrapped into ``(-pi, pi]`` per pair, so no unwrapping is needed.
    A global complex scale cancels in the ratio; a linear phase ramp adds a
    constant, which the caller removes by mean subtraction. The trailing
    row/column is edge-replicated so both results keep the input shape.
    """
    magnitude = numpy.absolute(values)
    floor = 1.0e-6 * (magnitude.mean() or 1.0)
    safe = numpy.where(magnitude > floor, values, floor)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        diff_x = numpy.log(safe[:, 1:] / safe[:, :-1])
        diff_y = numpy.log(safe[1:, :] / safe[:-1, :])

    gradient_x = numpy.concatenate([diff_x, diff_x[:, -1:]], axis=1)
    gradient_y = numpy.concatenate([diff_y, diff_y[-1:, :]], axis=0)
    return gradient_y, gradient_x


def _remove_gaussian_background(image: numpy.ndarray, sigma_px: float) -> numpy.ndarray:
    if sigma_px <= 0.0:
        return image

    if numpy.iscomplexobj(image):
        background = gaussian_filter(image.real, sigma_px) + 1j * gaussian_filter(
            image.imag, sigma_px
        )
    else:
        background = gaussian_filter(image, sigma_px)

    return image - background


def estimate_object_alignment_shift(
    reference_object: Object,
    moving_object: Object,
    *,
    upsample_factor: int = 100,
    registration_quantity: RegistrationQuantity = RegistrationQuantity.COMPLEX,
    high_pass_sigma_px: float = 0.0,
) -> tuple[float, float]:
    """Estimate the sub-pixel shift ``(dy, dx)``, in pixels, that aligns moving to reference.

    Both objects must already share a common ``(height, width)``; use
    :func:`center_crop_object` first if they do not. Layers are collapsed with
    :meth:`Object.get_layers_flattened` before registration.

    Args:
        reference_object: The object whose array indices define the target frame.
        moving_object: The object to be registered onto ``reference_object``.
        upsample_factor: Sub-pixel precision passed to
            ``skimage.registration.phase_cross_correlation``.
        registration_quantity: Which quantity derived from the complex object to
            correlate. See :class:`RegistrationQuantity`.
        high_pass_sigma_px: Sigma, in pixels, of a Gaussian background
            subtracted from the quantity before correlation, to suppress a
            slowly varying transmission or illumination envelope that differs
            between the two reconstructions. Disabled (``0.0``) by default:
            ``phase_cross_correlation`` normalizes the cross-power spectrum by
            its own modulus, which already whitens the spectrum, and stacking a
            small-sigma high-pass on top of that leaves only noise. Useful
            values are a sizeable fraction of the array, not a few pixels.

    Returns:
        ``(shift_y_px, shift_x_px)``, the translation to apply to
        ``moving_object`` so that it lands on ``reference_object``.
    """
    reference_flat = reference_object.get_layers_flattened()
    moving_flat = moving_object.get_layers_flattened()

    if reference_flat.shape != moving_flat.shape:
        raise ValueError(
            f'Arrays must have same shape; got {reference_flat.shape} vs {moving_flat.shape}!'
        )

    reference_image = _remove_gaussian_background(
        registration_quantity.extract(reference_flat), high_pass_sigma_px
    )
    moving_image = _remove_gaussian_background(
        registration_quantity.extract(moving_flat), high_pass_sigma_px
    )

    shift_yx, _, _ = phase_cross_correlation(
        reference_image, moving_image, upsample_factor=upsample_factor
    )

    return float(shift_yx[0]), float(shift_yx[1])


def shift_object(obj: Object, *, shift_y_px: float, shift_x_px: float) -> Object:
    """Translate every layer of ``obj`` by ``(shift_y_px, shift_x_px)`` and update its center.

    The translation is applied as a Fourier phase ramp so the complex phase
    survives the interpolation, and the returned object's :class:`ObjectCenter`
    is offset by ``-shift * pixel_size`` so world coordinates of the content are
    preserved.
    """
    if shift_y_px == 0.0 and shift_x_px == 0.0:
        return obj

    pixel_geometry = obj.get_pixel_geometry()
    old_center = obj.get_center()

    return Object(
        array=fourier_shift_2d(obj.get_array(), dx=shift_x_px, dy=shift_y_px),
        pixel_geometry=pixel_geometry.copy(),
        center=ObjectCenter(
            x_m=old_center.x_m - shift_x_px * pixel_geometry.width_m,
            y_m=old_center.y_m - shift_y_px * pixel_geometry.height_m,
        ),
        layer_spacing_m=list(obj.layer_spacing_m),
    )


def center_crop_object(obj: Object, target_h: int, target_w: int) -> Object:
    """Center-crop ``obj`` to ``(target_h, target_w)`` and update its center.

    Uses an asymmetric-toward-higher-index bias for odd differences
    (``start = delta // 2``), so an even shape difference preserves the object
    center exactly and an odd difference shifts it by half a pixel — the shift
    is absorbed into :class:`ObjectCenter` to keep the world-coordinate frame
    correct. All layers are sliced together.
    """
    array = obj.get_array()
    h_current = array.shape[-2]
    w_current = array.shape[-1]
    delta_h = h_current - target_h
    delta_w = w_current - target_w

    if delta_h < 0 or delta_w < 0:
        raise ValueError(
            f'Center-crop target ({target_h}, {target_w}) exceeds source '
            f'shape ({h_current}, {w_current})!'
        )

    if delta_h == 0 and delta_w == 0:
        return obj

    h_start = delta_h // 2
    w_start = delta_w // 2
    cropped_array = array[..., h_start : h_start + target_h, w_start : w_start + target_w].copy()

    pixel_geometry = obj.get_pixel_geometry()
    old_center = obj.get_center()
    # ObjectGeometry places pixel i at world offset (i - (N-1)/2) * pixel from center.
    # Cropping to [start : start + N_new] moves the array-center pixel by
    # (start - delta / 2) in original pixel units.
    center_shift_y_px = h_start - delta_h / 2.0
    center_shift_x_px = w_start - delta_w / 2.0
    new_center = ObjectCenter(
        x_m=old_center.x_m + center_shift_x_px * pixel_geometry.width_m,
        y_m=old_center.y_m + center_shift_y_px * pixel_geometry.height_m,
    )

    return Object(
        array=cropped_array,
        pixel_geometry=pixel_geometry.copy(),
        center=new_center,
        layer_spacing_m=list(obj.layer_spacing_m),
    )


def align_objects(
    reference_object: Object,
    moving_object: Object,
    *,
    upsample_factor: int = 100,
    registration_quantity: RegistrationQuantity = RegistrationQuantity.COMPLEX,
    high_pass_sigma_px: float = 0.0,
) -> tuple[Object, Object]:
    """Sub-pixel align ``moving_object`` to ``reference_object`` on a common shape.

    If the two inputs disagree on ``(height, width)``, both are first
    center-cropped to their common (min-in-each-axis) shape. Each object's
    center is updated so the crop preserves world coordinates: even shape
    differences preserve the center exactly; odd differences shift it by half
    a pixel, absorbed into :class:`ObjectCenter`. A sub-pixel translation
    between the cropped pair is then estimated by
    :func:`estimate_object_alignment_shift` and applied to every layer of the
    complex moving array via a Fourier phase ramp so the complex phase is
    preserved across the interpolation.

    The returned aligned moving object's ``center`` is offset from the
    (cropped) moving center by ``-shift_yx * pixel_size`` (in meters). This
    preserves the world-coordinate mapping of every probe position that was
    previously valid against ``moving_object``: a probe at world coordinate
    ``W`` that addressed a particular piece of content in ``moving_object``
    will, after alignment, address that same content at its new array index in
    the returned object. Both returned objects share the common shape and pixel
    grid so downstream elementwise math (e.g. :func:`ptychodus.api.xmcd.estimate_xmcd`)
    can be applied directly.

    The geometric-center crop assumes the two arrays' array-centers correspond
    to approximately the same physical point (the regime of pty-chi
    probe-position rounding and padding mismatches, which are on the order of
    one to a few pixels). Systematic integer-pixel offsets larger than
    ``ceil(delta / 2)`` in either axis are outside the recovery envelope of
    this scheme because the geometric crop discards the very content the
    correlator would need to see.

    Args:
        reference_object: The reconstruction whose array indices the result is
            aligned to.
        moving_object: The reconstruction to be re-registered. Must share
            ``reference_object``'s pixel geometry; flattened array shapes may
            differ and will be trimmed to their common shape.
        upsample_factor: Sub-pixel precision passed to
            ``phase_cross_correlation``. Higher values find finer shifts at
            roughly linear cost.
        registration_quantity: Which quantity derived from the complex object to
            correlate. The default is the only :class:`RegistrationQuantity` that is
            shift-exact, and it avoids ``AMPLITUDE``, which is nearly
            featureless for a phase-contrast sample.
        high_pass_sigma_px: Gaussian-background sigma in pixels, ``0.0`` to
            disable. See :func:`estimate_object_alignment_shift`.

    Returns:
        ``(cropped_reference, aligned_moving)``: both objects sharing the
        common ``(height, width)`` on the same pixel grid, with centers
        reflecting any crop-driven and shift-driven world-coordinate updates.
    """
    reference_pixel_geometry = reference_object.get_pixel_geometry()
    moving_pixel_geometry = moving_object.get_pixel_geometry()
    if reference_pixel_geometry != moving_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: reference {reference_pixel_geometry} '
            f'vs moving {moving_pixel_geometry}!'
        )

    common_h = min(reference_object.height_px, moving_object.height_px)
    common_w = min(reference_object.width_px, moving_object.width_px)
    cropped_reference = center_crop_object(reference_object, common_h, common_w)
    cropped_moving = center_crop_object(moving_object, common_h, common_w)

    shift_y_px, shift_x_px = estimate_object_alignment_shift(
        cropped_reference,
        cropped_moving,
        upsample_factor=upsample_factor,
        registration_quantity=registration_quantity,
        high_pass_sigma_px=high_pass_sigma_px,
    )
    logger.info(f'align_objects sub-pixel shift (y, x) = {(shift_y_px, shift_x_px)} px')

    aligned_moving = shift_object(cropped_moving, shift_y_px=shift_y_px, shift_x_px=shift_x_px)

    return cropped_reference, aligned_moving


def compute_object_geometry(
    positions: Iterable[ProbePosition],
    probe_geometry: ProbeGeometry,
    *,
    padding_px: int = 0,
) -> ObjectGeometry:
    """Size an object canvas covering the scan bounding box plus a probe-extent border on each side, with optional additional padding in pixels.

    Raises :class:`ValueError` when ``positions`` is empty. Pass a re-iterable
    sequence: :func:`calculate_scan_geometry` iterates twice.
    """
    scan = calculate_scan_geometry(positions)

    if scan is None:
        raise ValueError('Probe positions are empty; cannot compute an object geometry.')

    core_width_px = math.ceil(
        (scan.width_m + probe_geometry.width_m) / probe_geometry.pixel_width_m
    )
    core_height_px = math.ceil(
        (scan.height_m + probe_geometry.height_m) / probe_geometry.pixel_height_m
    )

    return ObjectGeometry(
        width_px=core_width_px + 2 * padding_px,
        height_px=core_height_px + 2 * padding_px,
        pixel_width_m=probe_geometry.pixel_width_m,
        pixel_height_m=probe_geometry.pixel_height_m,
        center_x_m=scan.center_x_m,
        center_y_m=scan.center_y_m,
    )


class ObjectFileReader(ABC):
    """Plugin interface for reading objects."""

    @abstractmethod
    def read(self, file_path: Path) -> Object:
        """Read an object from file."""
        pass


class ObjectFileWriter(ABC):
    """Plugin interface for writing objects."""

    @abstractmethod
    def write(self, file_path: Path, object_: Object) -> None:
        """Write an object to file."""
        pass
