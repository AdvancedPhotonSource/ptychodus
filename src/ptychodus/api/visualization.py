"""Visualization utilities: colormaps and image rendering for ptychography results."""

from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
from typing import Final, cast
import logging

from matplotlib.colors import Colormap, hsv_to_rgb
from scipy.stats import gaussian_kde
from skimage.restoration import unwrap_phase
import colorcet
import matplotlib
import numpy

from .common import ComplexArrayType, NumberArrayType, RealArrayType
from .geometry import Box2D, Interval, Line2D, PixelGeometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DisplayValues:
    """Real-valued array actually shown by a renderer, with its axis label.

    A renderer that maps a complex array to a single scalar component contributes one entry;
    the cylindrical color model, which encodes amplitude and phase simultaneously, contributes
    two.
    """

    label: str
    values: RealArrayType


@dataclass(frozen=True)
class LineCutSeries:
    """1D profile of a single displayed component, with its axis label."""

    label: str
    value: Sequence[float]


@dataclass(frozen=True)
class LineCut:
    """1D profile sampled along a line: distances in meters and one series per component."""

    distance_m: Sequence[float]
    series: Sequence[LineCutSeries]


@dataclass(frozen=True)
class KernelDensityEstimate:
    """KDE of pixel values within a region, including the value range and fitted kde object."""

    value_lower: float
    value_upper: float
    kde: gaussian_kde


class VisualizationProduct:
    """Colorized image with associated scalar values and pixel geometry for display and analysis."""

    EPS: Final[float] = 1.0e-6

    def __init__(
        self,
        value_label: str,
        values: NumberArrayType,
        rgba: RealArrayType,
        pixel_geometry: PixelGeometry,
        color_value_range: Interval[float],
        display_values: Sequence[DisplayValues] | None = None,
    ) -> None:
        if values.ndim != 2:
            raise ValueError(f'Values must be a 2-dimensional ndarray (actual={values.ndim}).')

        if rgba.ndim != 3:
            raise ValueError(f'RGBA must be a 3-dimensional ndarray (actual={rgba.ndim}).')

        if rgba.shape[2] != 4:
            raise ValueError(f'RGBA final dimension must have length=4 (actual={rgba.shape[2]}).')

        if values.shape[0] != rgba.shape[0] or values.shape[1] != rgba.shape[1]:
            raise ValueError(f'Shape mismatch (values={values.shape} and rgba={rgba.shape}).')

        if display_values is None:
            # No renderer told us which component is on screen; fall back to amplitude for
            # complex input and to the values themselves otherwise.
            fallback = numpy.absolute(values) if numpy.iscomplexobj(values) else values
            display_values = [DisplayValues(value_label, cast(RealArrayType, fallback))]

        for dv in display_values:
            if dv.values.ndim != 2:
                raise ValueError(
                    f'Display values "{dv.label}" must be a 2-dimensional ndarray '
                    f'(actual={dv.values.ndim}).'
                )

            if dv.values.shape[0] != rgba.shape[0] or dv.values.shape[1] != rgba.shape[1]:
                raise ValueError(
                    f'Shape mismatch (display values "{dv.label}"={dv.values.shape} '
                    f'and rgba={rgba.shape}).'
                )

        self._value_label = value_label
        self._values = values
        self._display_values = display_values
        self._rgba = rgba
        self._pixel_width_m = pixel_geometry.width_m
        self._pixel_height_m = pixel_geometry.height_m
        self._color_value_min = color_value_range.lower
        self._color_value_max = color_value_range.upper

    def get_value_label(self) -> str:
        return self._value_label

    def get_values(self) -> NumberArrayType:
        return self._values

    def get_display_values(self) -> Sequence[DisplayValues]:
        """Return the real-valued arrays the renderer put on screen, in display order."""
        return self._display_values

    def get_image_rgba(self) -> RealArrayType:
        return self._rgba

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self._pixel_width_m,
            height_m=self._pixel_height_m,
        )

    def get_color_value_range(self) -> Interval[float]:
        return Interval[float](self._color_value_min, self._color_value_max)

    @staticmethod
    def _intersect_bounding_box(begin: float, end: float, n: int) -> Interval[float]:
        length = end - begin

        if abs(length) < VisualizationProduct.EPS:
            return Interval[float](-numpy.inf, numpy.inf)
        else:
            return Interval[float].create_proper(
                (0 - begin) / length,
                (n - begin) / length,
            )

    @staticmethod
    def _intersect_grid_lines(
        begin: float, end: float, alpha_limits: Interval[float]
    ) -> Iterator[float]:
        ibegin = int(begin)
        iend = int(end)

        if iend < ibegin:
            ibegin, iend = iend, ibegin

        length = end - begin

        if abs(length) > VisualizationProduct.EPS:
            for idx in range(ibegin, iend + 1):
                alpha = (idx - begin) / length

                if alpha in alpha_limits:
                    yield alpha

    def _clip_to_bounding_box(self, line: Line2D) -> Interval[float]:
        alpha_x = self._intersect_bounding_box(line.begin.x, line.end.x, self._values.shape[-1])
        alpha_y = self._intersect_bounding_box(line.begin.y, line.end.y, self._values.shape[-2])

        return Interval[float].create_proper(
            max(0.0, max(alpha_x.lower, alpha_y.lower)),
            min(1.0, min(alpha_x.upper, alpha_y.upper)),
        )

    def _intersect_grid(self, line: Line2D) -> Sequence[float]:
        alpha_limits = self._clip_to_bounding_box(line)
        x_intersections = [
            x for x in self._intersect_grid_lines(line.begin.x, line.end.x, alpha_limits)
        ]
        y_intersections = [
            x for x in self._intersect_grid_lines(line.begin.y, line.end.y, alpha_limits)
        ]

        alpha = {alpha_limits.lower, alpha_limits.upper}
        alpha = alpha.union(x_intersections)
        alpha = alpha.union(y_intersections)
        return sorted(alpha)

    def get_info_text(self, x: float, y: float) -> str:
        ix = 0 if x < 0.0 else int(x)
        ix = min(ix, self._values.shape[-1])
        iy = 0 if y < 0.0 else int(y)
        iy = min(iy, self._values.shape[-2])
        value = self._values[iy, ix]

        if numpy.iscomplexobj(value):
            amplitude = numpy.absolute(value)
            phase = numpy.angle(value)
            return f'{x=:.1f} {y=:.1f} {amplitude=:6g} {phase=:6g}'

        return f'{x=:.1f} {y=:.1f} {value=:6g}'

    def get_line_cut(self, line: Line2D) -> LineCut:
        intersections = self._intersect_grid(line)

        dx = (line.end.x - line.begin.x) * self._pixel_width_m
        dy = (line.end.y - line.begin.y) * self._pixel_height_m
        line_length = numpy.hypot(dx, dy)

        distances: list[float] = list()
        values: list[list[float]] = [list() for dv in self._display_values]

        for alpha_l, alpha_r in zip(intersections[:-1], intersections[1:]):
            alpha = (alpha_l + alpha_r) / 2.0
            point = line.lerp(alpha)
            iy = int(point.y)
            ix = int(point.x)

            distances.append(alpha * line_length)

            for value_list, dv in zip(values, self._display_values):
                value_list.append(dv.values[iy, ix].item())

        series = [
            LineCutSeries(dv.label, value_list)
            for dv, value_list in zip(self._display_values, values)
        ]
        return LineCut(distances, series)

    def estimate_kernel_density(self, box: Box2D) -> KernelDensityEstimate:
        # A histogram has a single value axis, so only the primary displayed component is
        # estimated; the cylindrical color model contributes amplitude first for this reason.
        display_values = self._display_values[0].values

        x_range = Interval[int](0, display_values.shape[-1])
        x_begin = x_range.clamp(int(box.x_begin))
        x_end = x_range.clamp(int(box.x_end) + 1)

        y_range = Interval[int](0, display_values.shape[-2])
        y_begin = y_range.clamp(int(box.y_begin))
        y_end = y_range.clamp(int(box.y_end) + 1)

        values = display_values[..., y_begin:y_end, x_begin:x_end]
        values = values.reshape(values.shape[-3], -1) if values.ndim > 2 else values.reshape(-1)

        return KernelDensityEstimate(values.min(), values.max(), gaussian_kde(values))


class ComplexComponent(Enum):
    """Component of a complex-valued array to use for visualization."""

    REAL = auto()
    IMAGINARY = auto()
    AMPLITUDE = auto()
    PHASE_RAD = auto()
    UNWRAPPED_PHASE_RAD = auto()

    @property
    def is_cyclic(self) -> bool:
        return self is ComplexComponent.PHASE_RAD

    def extract_component(
        self,
        array: NumberArrayType,
        dtype: numpy.dtype[numpy.floating[Any]] = numpy.dtype(numpy.single),
    ) -> RealArrayType:
        match self:
            case ComplexComponent.REAL:
                return numpy.real(array).astype(dtype)
            case ComplexComponent.IMAGINARY:
                return numpy.imag(array).astype(dtype)
            case ComplexComponent.AMPLITUDE:
                return numpy.absolute(array).astype(dtype)
            case ComplexComponent.PHASE_RAD:
                return numpy.angle(array).astype(dtype)
            case ComplexComponent.UNWRAPPED_PHASE_RAD:
                phase_rad = numpy.angle(array).astype(dtype)
                return unwrap_phase(phase_rad)


class ScalarTransformation(Enum):
    """Monotonic scalar transformations applied to real-valued arrays before colormap mapping."""

    IDENTITY = auto()
    SQRT = auto()
    LOG2 = auto()
    LOG = auto()
    LOG10 = auto()

    def decorate_text(self, text: str) -> str:
        match self:
            case ScalarTransformation.SQRT:
                return f'$\\sqrt{{\\mathrm{{{text}}}}}$'
            case ScalarTransformation.LOG2:
                return f'$\\log_2{{\\left(\\mathrm{{{text}}}\\right)}}$'
            case ScalarTransformation.LOG:
                return f'$\\ln{{\\left(\\mathrm{{{text}}}\\right)}}$'
            case ScalarTransformation.LOG10:
                return f'$\\log_{{10}}{{\\left(\\mathrm{{{text}}}\\right)}}$'

        return text

    def transform(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)

        match self:
            case ScalarTransformation.SQRT:
                return numpy.sqrt(array, out=nil, where=(array > 0))
            case ScalarTransformation.LOG2:
                return numpy.log2(array, out=nil, where=(array > 0))
            case ScalarTransformation.LOG:
                return numpy.log(array, out=nil, where=(array > 0))
            case ScalarTransformation.LOG10:
                return numpy.log10(array, out=nil, where=(array > 0))

        return array


def linear_colormap_names() -> Iterator[str]:
    """Yield the preferred alias for every linear (non-diverging) colorcet colormap."""
    for original_name in colorcet.all_original_names(group='linear', not_group='diverging'):
        try:
            cmap_aliases = colorcet.aliases[original_name]
        except KeyError:
            yield original_name
        else:
            yield cmap_aliases[0]


def cyclic_colormap_names() -> Iterator[str]:
    """Yield the preferred alias for every cyclic colorcet colormap."""
    for group in ('cyclic', 'circle'):
        for original_name in colorcet.all_original_names(group=group):
            try:
                cmap_aliases = colorcet.aliases[original_name]
            except KeyError:
                yield original_name
            else:
                yield cmap_aliases[0]


def get_colormap_by_name(name: str) -> Colormap:
    """Return the colorcet Colormap for the given short name (prefixed with ``cet_``)."""
    return matplotlib.colormaps[f'cet_{name}']


def _normalize(
    values: RealArrayType,
    *,
    value_min: float | None = None,
    value_max: float | None = None,
    clip: bool = False,
) -> tuple[RealArrayType, Interval[float]]:
    mask = numpy.isfinite(values)

    if not mask.all():
        num_nonfinite = numpy.count_nonzero(numpy.logical_not(mask))
        logger.warning(f'Encountered {num_nonfinite} non-finite value(s) during normalization!')

    if value_min is None:
        value_min = numpy.min(values[mask]).item()
    elif clip:
        values = numpy.maximum(value_min, values)

    if value_max is None:
        value_max = numpy.max(values[mask]).item()
    elif clip:
        values = numpy.minimum(value_max, values)

    value_range = Interval[float](value_min, value_max)

    if value_max < value_min:
        raise ValueError('value_max < value_min')
    elif value_max > value_min:
        return (values - value_min) / (value_max - value_min), value_range
    else:
        return numpy.zeros_like(values), value_range


def visualize_real_values(
    value_label: str,
    values: RealArrayType,
    pixel_geometry: PixelGeometry,
    colormap: Colormap | str = 'gray',
    *,
    transform: ScalarTransformation = ScalarTransformation.IDENTITY,
    value_min: float | None = None,
    value_max: float | None = None,
    clip: bool = False,
) -> VisualizationProduct:
    """Render a real-valued 2D array as a colorized VisualizationProduct."""
    values_transformed = transform.transform(values)
    values_normalized, color_value_range = _normalize(
        values_transformed, value_min=value_min, value_max=value_max, clip=clip
    )
    cmap = colormap if isinstance(colormap, Colormap) else get_colormap_by_name(colormap)
    decorated_label = transform.decorate_text(value_label)
    return VisualizationProduct(
        value_label=decorated_label,
        values=values,
        rgba=cmap(values_normalized),
        pixel_geometry=pixel_geometry,
        color_value_range=color_value_range,
        display_values=[DisplayValues(decorated_label, values_transformed)],
    )


def visualize_complex_component(
    values: NumberArrayType,
    pixel_geometry: PixelGeometry,
    component: ComplexComponent,
    colormap: Colormap | str = 'gray',
    *,
    transform: ScalarTransformation = ScalarTransformation.IDENTITY,
    value_min: float | None = None,
    value_max: float | None = None,
    clip: bool = False,
) -> VisualizationProduct:
    """Render a single scalar component of a numeric array as a colorized VisualizationProduct."""
    product = visualize_real_values(
        value_label=component.name.title(),
        values=component.extract_component(values),
        pixel_geometry=pixel_geometry,
        colormap=colormap,
        transform=transform,
        value_min=value_min,
        value_max=value_max,
        clip=clip,
    )
    return VisualizationProduct(
        value_label=product.get_value_label(),
        values=values,
        rgba=product.get_image_rgba(),
        pixel_geometry=product.get_pixel_geometry(),
        color_value_range=product.get_color_value_range(),
        display_values=product.get_display_values(),
    )


def hsva_to_rgba(
    hue: RealArrayType, saturation: RealArrayType, value: RealArrayType, alpha: RealArrayType
) -> RealArrayType:
    """Convert per-pixel HSV + alpha arrays to an RGBA array."""
    hsv = numpy.stack((hue, saturation, value), axis=-1)
    rgb = hsv_to_rgb(hsv)

    if alpha.ndim == 1:
        return numpy.column_stack((rgb, alpha))

    return numpy.dstack((rgb, alpha))


def _v(m1: RealArrayType, m2: RealArrayType, hue: RealArrayType) -> RealArrayType:
    """Adapted from colorsys._v in the Python standard library."""
    if m1.shape != hue.shape or m2.shape != hue.shape:
        raise ValueError('Shape mismatch: m1, m2, and hue must have the same shape.')

    hue = hue % 1.0

    return numpy.select(
        [hue < 1.0 / 6.0, hue < 0.5, hue < 2.0 / 3.0],
        [m1 + (m2 - m1) * hue * 6.0, m2, m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0],
        default=m1,
    )


def hlsa_to_rgba(
    hue: RealArrayType, lightness: RealArrayType, saturation: RealArrayType, alpha: RealArrayType
) -> RealArrayType:
    """Adapted from colorsys.hls_to_rgb in the Python standard library."""
    one_third: Final[float] = 1.0 / 3.0

    m2 = numpy.where(
        lightness <= 0.5,
        lightness * (1.0 + saturation),
        lightness + saturation - (lightness * saturation),
    )
    m1 = 2.0 * lightness - m2

    red = numpy.where(saturation > 0.0, _v(m1, m2, hue + one_third), lightness)
    green = numpy.where(saturation > 0.0, _v(m1, m2, hue), lightness)
    blue = numpy.where(saturation > 0.0, _v(m1, m2, hue - one_third), lightness)

    return numpy.stack((red, green, blue, alpha), axis=-1)


class CylindricalColorModel(Enum):
    """Cylindrical color model variant used to encode complex amplitude as hue and a second channel."""

    HSV_SATURATION = auto()
    HSV_VALUE = auto()
    HSV_ALPHA = auto()
    HLS_LIGHTNESS = auto()
    HLS_SATURATION = auto()
    HLS_ALPHA = auto()

    @property
    def simple_name(self) -> str:
        hxx, variant = self.name.split('_')
        return f'{hxx}-{variant[0]}'

    @property
    def display_name(self) -> str:
        hxx, variant = self.name.split('_')
        return f'{hxx} {variant.title()}'

    def render_rgba(self, hue: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(hue)

        match self:
            case CylindricalColorModel.HSV_SATURATION:
                return hsva_to_rgba(hue, x, ones, ones)
            case CylindricalColorModel.HSV_VALUE:
                return hsva_to_rgba(hue, ones, x, ones)
            case CylindricalColorModel.HSV_ALPHA:
                return hsva_to_rgba(hue, ones, ones, x)
            case CylindricalColorModel.HLS_LIGHTNESS:
                return hlsa_to_rgba(hue, x, ones, ones)
            case CylindricalColorModel.HLS_SATURATION:
                return hlsa_to_rgba(hue, ones / 2.0, x, ones)
            case CylindricalColorModel.HLS_ALPHA:
                return hlsa_to_rgba(hue, ones / 2.0, ones, x)


def visualize_complex_values(
    values: ComplexArrayType,
    pixel_geometry: PixelGeometry,
    model: CylindricalColorModel,
    *,
    amplitude_transform: ScalarTransformation,
    value_min: float | None = None,
    value_max: float | None = None,
    clip: bool = False,
) -> VisualizationProduct:
    """Render a complex array using a cylindrical color model (hue = phase)."""
    amplitude_component = ComplexComponent.AMPLITUDE
    amplitude = amplitude_component.extract_component(values)
    amplitude_transformed = amplitude_transform.transform(amplitude)
    amplitude_normalized, color_value_range = _normalize(
        amplitude_transformed, value_min=value_min, value_max=value_max, clip=clip
    )
    phase_rad = ComplexComponent.PHASE_RAD.extract_component(values)
    hue = (phase_rad + numpy.pi) / (2 * numpy.pi)
    amplitude_label = amplitude_transform.decorate_text(amplitude_component.name.title())
    return VisualizationProduct(
        value_label=amplitude_label,
        values=values,
        rgba=model.render_rgba(hue, amplitude_normalized),
        pixel_geometry=pixel_geometry,
        color_value_range=color_value_range,
        display_values=[
            DisplayValues(amplitude_label, amplitude_transformed),
            DisplayValues('Phase [rad]', phase_rad),
        ],
    )
