"""Object (transmission function) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Final
import logging
import math

import numpy
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from skimage.restoration import unwrap_phase

from .typing import ComplexArrayType, RealArrayType
from .constants import format_length
from .fourier import fourier_shift_2d
from .geometry import PixelGeometry
from .probe import ProbeGeometry
from .probe_positions import ProbePosition, calculate_scan_geometry

logger = logging.getLogger(__name__)

# A true 2-pi wrap draws a contour of large adjacent-pixel jumps across a layer, while
# ordinary high-contrast structure produces only scattered ones.
_WRAPPED_PHASE_JUMP_FRACTION: Final[float] = 0.002


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


def _with_layers(obj: Object, array: ComplexArrayType, layer_spacing_m: Sequence[float]) -> Object:
    """Rebuild *obj* around a new layer stack, carrying its spatial metadata through.

    Mirrors :meth:`Object.copy` in tolerating absent pixel geometry and center, so the
    layer transforms work on an object straight out of a file reader.
    """
    return Object(
        array=array,
        pixel_geometry=None if obj._pixel_geometry is None else obj._pixel_geometry.copy(),
        center=None if obj._center is None else obj._center.copy(),
        layer_spacing_m=list(layer_spacing_m),
    )


def _layer_depths_m(layer_spacing_m: Sequence[float]) -> RealArrayType:
    """Depth of each layer, measured from the entrance layer at zero."""
    return numpy.concatenate(([0.0], numpy.cumsum(numpy.asarray(layer_spacing_m, dtype=float))))


def _warn_if_phase_appears_wrapped(phase_rad: RealArrayType) -> None:
    """Warn when a layer's phase looks wrapped, which invalidates its principal logarithm.

    A true 2-pi wrap draws a contour of adjacent-pixel jumps larger than pi across the
    layer, so the fraction of such jumps separates wrapping from ordinary high-contrast
    structure, which produces only scattered ones.
    """
    for index, layer in enumerate(phase_rad):
        jumps = 0
        total = 0

        for axis in (-2, -1):
            difference = numpy.diff(layer, axis=axis)
            jumps += int(numpy.count_nonzero(numpy.abs(difference) > numpy.pi))
            total += difference.size

        if total > 0 and jumps / total > _WRAPPED_PHASE_JUMP_FRACTION:
            logger.warning(
                'Layer %d looks phase-wrapped (%.1f%% of neighboring pixels jump by more than '
                'pi); its principal logarithm is not the true optical path. Pass '
                'unwrap_phase_rad=True to unwrap it first.',
                index,
                100.0 * jumps / total,
            )


def _layer_log_transmission(
    array: ComplexArrayType, *, unwrap_phase_rad: bool, amplitude_floor: float
) -> ComplexArrayType:
    """Per-layer complex logarithm, the additive form of the layer product.

    Layers compose multiplicatively, so summing this over the layer axis gives the total
    optical path. Doing it per layer is what keeps the total free of phase wrapping: each
    layer's own phase lies in ``(-pi, pi]`` whenever the multislice decomposition is
    valid, and the sum then accumulates past ``+/-pi`` without ever being wrapped.
    Unwrapping is needed only for a single layer thick enough to wrap by itself.

    Args:
        array: Complex layer stack, shape ``(layers, height, width)``.
        unwrap_phase_rad: Unwrap each layer's phase spatially before taking its logarithm.
        amplitude_floor: Smallest magnitude used in the logarithm. A padded object carries
            a zero-amplitude border, whose logarithm would otherwise be ``-inf`` and would
            poison any later difference of cumulative sums.
    """
    magnitude = numpy.maximum(numpy.abs(array), amplitude_floor)

    if unwrap_phase_rad:
        phase_rad = numpy.stack([unwrap_phase(numpy.angle(layer)) for layer in array])
    else:
        phase_rad = numpy.angle(array)
        _warn_if_phase_appears_wrapped(phase_rad)

    return numpy.log(magnitude) + 1j * phase_rad


def compute_uniform_layer_spacing_m(total_thickness_m: float, num_layers: int) -> list[float]:
    """Spacing that distributes ``num_layers`` layers evenly over ``total_thickness_m``.

    The layers are phase screens at depths ``0`` through ``total_thickness_m``, so there
    are ``num_layers - 1`` gaps of ``total_thickness_m / (num_layers - 1)`` and the
    returned list sums to ``total_thickness_m`` exactly. This is the convention
    :meth:`Object.get_total_thickness_m` reports and the one the reconstruction backends
    propagate: they step between consecutive screens and traverse the whole stack.

    A non-positive thickness yields the right number of zero gaps rather than an empty
    list, so the result always satisfies the layer-count invariant of :class:`Object`.

    Raises:
        ValueError: If ``num_layers`` is less than one.
    """
    if num_layers < 1:
        raise ValueError(f'Need at least one layer; got {num_layers}.')

    if num_layers == 1:
        return []

    if total_thickness_m <= 0.0:
        return [0.0] * (num_layers - 1)

    return [total_thickness_m / (num_layers - 1)] * (num_layers - 1)


class LayerFillMode(Enum):
    """Content of a layer inserted by :func:`resize_object_layers`."""

    GEOMETRIC_MEAN = auto()
    """The slab whose repetition reproduces the whole object; thickens by one slab."""

    EDGE = auto()
    """A copy of the exit layer; thickens by one slab."""

    VACUUM = auto()
    """Unit transmission, the only fill that leaves the layer product unchanged."""


def select_object_layers(
    obj: Object, indexes: Sequence[int], *, drop_out_of_range: bool = False
) -> Object:
    """Keep only the layers named by ``indexes``.

    The output spacing is read off the input depth grid, so a non-contiguous selection
    still records the true distance between the layers it kept.

    Args:
        obj: Source object.
        indexes: Strictly increasing layer indexes to keep.
        drop_out_of_range: Silently discard indexes outside the layer stack instead of
            raising. An entirely out-of-range selection still raises, because an object
            must keep at least one layer.

    Raises:
        ValueError: If ``indexes`` is empty, or is not strictly increasing.
        IndexError: If an index falls outside the stack and ``drop_out_of_range`` is off.
    """
    num_layers = obj.num_layers
    selection = [int(index) for index in indexes]

    if drop_out_of_range:
        selection = [index for index in selection if 0 <= index < num_layers]
    else:
        for index in selection:
            if not 0 <= index < num_layers:
                raise IndexError(f'Layer {index} is outside a stack of {num_layers} layer(s).')

    if not selection:
        raise ValueError('Layer selection is empty; an object must keep at least one layer.')

    if any(later <= earlier for earlier, later in zip(selection, selection[1:])):
        raise ValueError(f'Layer selection must be strictly increasing; got {selection}.')

    depths_m = _layer_depths_m(obj.layer_spacing_m)
    spacing_m = [
        float(depths_m[later] - depths_m[earlier])
        for earlier, later in zip(selection, selection[1:])
    ]

    return _with_layers(obj, obj.get_array()[selection], spacing_m)


def homogenize_object_layers(
    obj: Object,
    *,
    num_layers: int | None = None,
    layer_spacing_m: Sequence[float] | None = None,
    unwrap_phase_rad: bool = False,
    amplitude_floor: float = 1.0e-12,
) -> Object:
    """Spread the object's whole transmission evenly over ``num_layers`` identical layers.

    Each output layer is the ``num_layers``-th root of the layer product, so
    :meth:`Object.get_layers_flattened` is invariant and the result is the homogeneous
    slab stack with the same projected transmission. The root is taken through the
    summed per-layer logarithm rather than through the phase of the product, which is
    what keeps a total phase outside ``(-pi, pi]`` correct.

    Args:
        obj: Source object.
        num_layers: Output layer count. Defaults to the input count.
        layer_spacing_m: Output spacing. Required when the layer count changes and the
            result has more than one layer, because the output depth grid does not follow
            from the input one.
        unwrap_phase_rad: Unwrap each input layer's phase before summing. Needed only when
            a single layer is itself thick enough to wrap.
        amplitude_floor: Smallest magnitude used in the logarithm.

    Raises:
        ValueError: If ``num_layers`` is less than one, if the layer count changes without
            an explicit ``layer_spacing_m``, or if that spacing has the wrong length.
    """
    count = obj.num_layers if num_layers is None else int(num_layers)

    if count < 1:
        raise ValueError(f'Need at least one layer; got {count}.')

    if layer_spacing_m is not None:
        spacing_m: Sequence[float] = layer_spacing_m
    elif count == obj.num_layers:
        spacing_m = obj.layer_spacing_m
    elif count == 1:
        spacing_m = []
    else:
        raise ValueError(
            f'Homogenizing {obj.num_layers} layer(s) into {count} needs an explicit '
            'layer_spacing_m; the output depth grid does not follow from the input one.'
        )

    if len(spacing_m) != count - 1:
        raise ValueError(f'Expected {count - 1} layer spacing(s) for {count} layers.')

    log_transmission = _layer_log_transmission(
        obj.get_array(), unwrap_phase_rad=unwrap_phase_rad, amplitude_floor=amplitude_floor
    ).sum(axis=0)
    layer = numpy.exp(log_transmission / count)

    return _with_layers(obj, numpy.repeat(layer[numpy.newaxis], count, axis=0), spacing_m)


def scale_object_phase(
    obj: Object, scaling: float, *, unwrap_phase_rad: bool | None = None
) -> Object:
    """Multiply the phase of every layer by ``scaling``, leaving the amplitude alone.

    Args:
        obj: Source object.
        scaling: Factor applied to the phase.
        unwrap_phase_rad: Whether to unwrap each layer's phase first. The default decides
            from ``scaling``: an integer factor needs no unwrapping, because the wrapped
            and true phases differ by ``2 pi k`` and an integer multiple of that is again
            a whole number of turns, so it cancels. A fractional factor does not cancel
            and is unwrapped. Pass a bool to force the choice.
    """
    if scaling == 1.0:
        return obj

    if unwrap_phase_rad is None:
        unwrap_phase_rad = not float(scaling).is_integer()

    array = obj.get_array()

    if unwrap_phase_rad:
        phase_rad = numpy.stack([unwrap_phase(numpy.angle(layer)) for layer in array])
    else:
        phase_rad = numpy.angle(array)

    scaled = numpy.abs(array) * numpy.exp(1j * phase_rad * scaling)

    return _with_layers(obj, scaled, obj.layer_spacing_m)


def resize_object_layers(
    obj: Object,
    num_layers: int,
    *,
    fill_mode: LayerFillMode = LayerFillMode.VACUUM,
    layer_spacing_m: Sequence[float] | None = None,
) -> Object:
    """Grow or shrink the layer stack to ``num_layers`` without interpolating.

    Shrinking to one layer takes the layer product, so the whole transmission survives;
    shrinking to more than one center-crops the stack and discards the outermost layers.
    Growing inserts ``fill_mode`` layers alternately after and before the existing stack.
    Only :attr:`LayerFillMode.VACUUM` leaves the layer product unchanged; the other fills
    thicken the object by one slab each. Use :func:`resample_object_layers` to change the
    layer count at a fixed thickness.

    Args:
        obj: Source object.
        num_layers: Output layer count.
        fill_mode: Content of each inserted layer.
        layer_spacing_m: Output spacing. Defaults to slicing the input spacing when
            shrinking and to repeating the adjacent gap when growing.

    Raises:
        ValueError: If ``num_layers`` is less than one, or ``layer_spacing_m`` has the
            wrong length.
    """
    if num_layers < 1:
        raise ValueError(f'Need at least one layer; got {num_layers}.')

    array = obj.get_array()
    count = obj.num_layers

    if num_layers == count:
        if layer_spacing_m is None or list(layer_spacing_m) == list(obj.layer_spacing_m):
            return obj

        resized = array
        spacing_m: Sequence[float] = layer_spacing_m
    elif num_layers == 1:
        resized = obj.get_layers_flattened()[numpy.newaxis]
        spacing_m = [] if layer_spacing_m is None else layer_spacing_m
    elif num_layers < count:
        start = (count - num_layers) // 2
        resized = array[start : start + num_layers]
        spacing_m = (
            list(obj.layer_spacing_m)[start : start + num_layers - 1]
            if layer_spacing_m is None
            else layer_spacing_m
        )
    else:
        match fill_mode:
            case LayerFillMode.GEOMETRIC_MEAN:
                filler = numpy.exp(
                    _layer_log_transmission(
                        array, unwrap_phase_rad=False, amplitude_floor=1.0e-12
                    ).mean(axis=0)
                )
            case LayerFillMode.EDGE:
                filler = array[-1]
            case LayerFillMode.VACUUM:
                filler = numpy.ones_like(array[0])

        layers = list(array)
        gaps = list(obj.layer_spacing_m)
        back_gap_m = gaps[-1] if gaps else 0.0
        front_gap_m = gaps[0] if gaps else 0.0

        for index in range(num_layers - count):
            if index % 2 == 0:
                layers.append(filler)
                gaps.append(back_gap_m)
            else:
                layers.insert(0, filler)
                gaps.insert(0, front_gap_m)

        resized = numpy.stack(layers)
        spacing_m = gaps if layer_spacing_m is None else layer_spacing_m

    if len(spacing_m) != num_layers - 1:
        raise ValueError(f'Expected {num_layers - 1} layer spacing(s) for {num_layers} layers.')

    return _with_layers(obj, resized, spacing_m)


def _voronoi_boundaries_m(depths_m: RealArrayType) -> RealArrayType:
    """Slab boundaries around a set of layer depths.

    Each interior layer owns the interval between its midpoints with its two neighbors;
    the entrance and exit layers own half-slabs, because they sit on the ends of the
    stack rather than at the center of one. The boundaries therefore span exactly the
    same range as the depths and there is one more of them than there are layers.
    """
    return numpy.concatenate(([depths_m[0]], (depths_m[:-1] + depths_m[1:]) / 2.0, [depths_m[-1]]))


def resample_object_layers(
    obj: Object,
    layer_spacing_m: Sequence[float],
    *,
    preserve_total_transmission: bool = True,
    unwrap_phase_rad: bool = False,
    amplitude_floor: float = 1.0e-12,
) -> Object:
    """Resample the layer stack onto the depth grid implied by ``layer_spacing_m``.

    The output has ``1 + len(layer_spacing_m)`` layers. Resampling runs on the cumulative
    complex optical path sampled at slab boundaries, and each output layer is a
    difference of the interpolated path, so the layers still compose multiplicatively and
    no layer's phase is ever wrapped. Because those differences telescope, the layer
    product is preserved exactly whenever the output grid spans the input one -- to
    floating-point round-off, with no renormalization step.

    Interpolation is monotone cubic (PCHIP) on that cumulative path. Monotonicity is the
    property that matters: it prevents the difference from handing a layer a phase step
    that runs against the local trend, or an amplitude above the material's own.

    Args:
        obj: Source object.
        layer_spacing_m: Output spacing, one entry per gap.
        preserve_total_transmission: Stretch the output depth grid onto the input span
            before evaluating, making the layer product exactly invariant. Disable to
            evaluate the output boundaries at their own physical depths, letting the
            transmission follow the change in thickness.
        unwrap_phase_rad: Unwrap each input layer's phase before accumulating. Needed only
            when a single layer is itself thick enough to wrap.
        amplitude_floor: Smallest magnitude used in the logarithm.
    """
    # An unchanged depth grid must be a bit-exact no-op: callers rebuild objects on every
    # settings notification, and a round trip through log and exp would drift each time.
    if list(layer_spacing_m) == list(obj.layer_spacing_m):
        return obj

    num_layers = 1 + len(layer_spacing_m)
    source_depths_m = _layer_depths_m(obj.layer_spacing_m)
    source_span_m = float(source_depths_m[-1])
    target_depths_m = _layer_depths_m(layer_spacing_m)
    target_span_m = float(target_depths_m[-1])

    # A single source layer, a stack with no thickness, or a single output layer leaves no
    # depth axis to resample along; spreading the transmission evenly is the whole of it.
    degenerate = obj.num_layers == 1 or source_span_m <= 0.0 or num_layers == 1

    if degenerate or (preserve_total_transmission and target_span_m <= 0.0):
        return homogenize_object_layers(
            obj,
            num_layers=num_layers,
            layer_spacing_m=layer_spacing_m,
            unwrap_phase_rad=unwrap_phase_rad,
            amplitude_floor=amplitude_floor,
        )

    if preserve_total_transmission:
        target_depths_m = target_depths_m * (source_span_m / target_span_m)

    source_boundaries_m = _voronoi_boundaries_m(source_depths_m)
    target_boundaries_m = _voronoi_boundaries_m(target_depths_m)

    log_transmission = _layer_log_transmission(
        obj.get_array(), unwrap_phase_rad=unwrap_phase_rad, amplitude_floor=amplitude_floor
    )
    cumulative = numpy.concatenate(
        (
            numpy.zeros((1, *log_transmission.shape[1:]), dtype=log_transmission.dtype),
            numpy.cumsum(log_transmission, axis=0),
        )
    )

    # PchipInterpolator takes real data only, and the real and imaginary parts of the
    # cumulative path are both smooth, so interpolating them separately is exact.
    real = PchipInterpolator(source_boundaries_m, cumulative.real, axis=0)(target_boundaries_m)
    imaginary = PchipInterpolator(source_boundaries_m, cumulative.imag, axis=0)(target_boundaries_m)

    resampled = numpy.exp(numpy.diff(real + 1j * imaginary, axis=0))

    return _with_layers(obj, resampled, layer_spacing_m)


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
