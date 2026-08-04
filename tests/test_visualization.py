"""Unit tests for ptychodus.api.visualization."""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import Box2D, Interval, Line2D, PixelGeometry, Point2D
from ptychodus.api.visualization import (
    ComplexComponent,
    CylindricalColorModel,
    DisplayValues,
    KernelDensityEstimate,
    LineCut,
    ScalarTransformation,
    VisualizationProduct,
    cyclic_colormap_names,
    get_colormap_by_name,
    hlsa_to_rgba,
    hsva_to_rgba,
    linear_colormap_names,
    visualize_complex_component,
    visualize_complex_values,
    visualize_real_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pixel_geo(width_m: float = 1e-6, height_m: float = 1e-6) -> PixelGeometry:
    return PixelGeometry(width_m=width_m, height_m=height_m)


def _make_product(
    rows: int = 4,
    cols: int = 4,
    *,
    value_label: str = 'test',
    pixel_geo: PixelGeometry | None = None,
) -> VisualizationProduct:
    rng = numpy.random.default_rng(0)
    values = rng.random((rows, cols)).astype(numpy.float32)
    rgba = numpy.ones((rows, cols, 4), dtype=numpy.float32)
    geo = pixel_geo if pixel_geo is not None else _pixel_geo()
    color_range = Interval[float](0.0, 1.0)
    return VisualizationProduct(value_label, values, rgba, geo, color_range)


# ---------------------------------------------------------------------------
# VisualizationProduct construction
# ---------------------------------------------------------------------------


class TestVisualizationProductConstruction:
    def test_valid_construction(self):
        vp = _make_product()
        assert vp.get_value_label() == 'test'

    def test_rejects_1d_values(self):
        values = numpy.ones(4)
        rgba = numpy.ones((4, 4, 4))
        with pytest.raises(ValueError, match='2-dimensional'):
            VisualizationProduct('x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))

    def test_rejects_2d_rgba(self):
        values = numpy.ones((4, 4))
        rgba = numpy.ones((4, 4))
        with pytest.raises(ValueError, match='3-dimensional'):
            VisualizationProduct('x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))

    def test_rejects_rgba_wrong_last_dim(self):
        values = numpy.ones((4, 4))
        rgba = numpy.ones((4, 4, 3))
        with pytest.raises(ValueError, match='length=4'):
            VisualizationProduct('x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))

    def test_rejects_shape_mismatch(self):
        values = numpy.ones((4, 4))
        rgba = numpy.ones((3, 4, 4))
        with pytest.raises(ValueError, match='Shape mismatch'):
            VisualizationProduct('x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))

    def test_rejects_1d_display_values(self):
        values = numpy.ones((4, 4))
        rgba = numpy.ones((4, 4, 4))
        display_values = [DisplayValues('bad', numpy.ones(4))]
        with pytest.raises(ValueError, match='2-dimensional'):
            VisualizationProduct(
                'x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0), display_values
            )

    def test_rejects_display_values_shape_mismatch(self):
        values = numpy.ones((4, 4))
        rgba = numpy.ones((4, 4, 4))
        display_values = [DisplayValues('bad', numpy.ones((3, 4)))]
        with pytest.raises(ValueError, match='Shape mismatch'):
            VisualizationProduct(
                'x', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0), display_values
            )

    def test_default_display_values_for_real_input(self):
        vp = _make_product()
        (dv,) = vp.get_display_values()
        assert dv.label == 'test'
        numpy.testing.assert_array_equal(dv.values, vp.get_values())

    def test_default_display_values_for_complex_input(self):
        values = numpy.ones((4, 4), dtype=complex) * (3 + 4j)
        rgba = numpy.ones((4, 4, 4), dtype=numpy.float32)
        vp = VisualizationProduct('c', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))
        (dv,) = vp.get_display_values()
        numpy.testing.assert_allclose(dv.values, 5.0)


# ---------------------------------------------------------------------------
# VisualizationProduct accessors
# ---------------------------------------------------------------------------


class TestVisualizationProductAccessors:
    def setup_method(self):
        self.vp = _make_product(pixel_geo=PixelGeometry(width_m=2e-6, height_m=3e-6))

    def test_get_values_shape(self):
        assert self.vp.get_values().shape == (4, 4)

    def test_get_image_rgba_shape(self):
        assert self.vp.get_image_rgba().shape == (4, 4, 4)

    def test_get_pixel_geometry(self):
        geo = self.vp.get_pixel_geometry()
        assert geo.width_m == pytest.approx(2e-6)
        assert geo.height_m == pytest.approx(3e-6)

    def test_get_color_value_range(self):
        r = self.vp.get_color_value_range()
        assert r.lower == pytest.approx(0.0)
        assert r.upper == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VisualizationProduct.get_info_text
# ---------------------------------------------------------------------------


class TestGetInfoText:
    def test_real_value(self):
        vp = _make_product()
        text = vp.get_info_text(0.0, 0.0)
        assert 'value=' in text
        assert 'x=' in text and 'y=' in text

    def test_complex_value(self):
        values = numpy.ones((4, 4), dtype=complex) * (1 + 1j)
        rgba = numpy.ones((4, 4, 4), dtype=numpy.float32)
        vp = VisualizationProduct('c', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))
        text = vp.get_info_text(1.0, 1.0)
        assert 'amplitude=' in text
        assert 'phase=' in text

    def test_complex_value_with_zero_imaginary_part(self):
        # A complex-dtype pixel whose imaginary part is exactly zero must still take the
        # amplitude/phase branch; formatting a complex with '6g' raises TypeError.
        values = numpy.zeros((4, 4), dtype=complex)
        rgba = numpy.ones((4, 4, 4), dtype=numpy.float32)
        vp = VisualizationProduct('c', values, rgba, _pixel_geo(), Interval[float](0.0, 1.0))
        text = vp.get_info_text(1.0, 1.0)
        assert 'amplitude=' in text
        assert 'phase=' in text

    def test_clamps_negative_coords(self):
        vp = _make_product()
        text = vp.get_info_text(-5.0, -5.0)
        assert 'value=' in text


# ---------------------------------------------------------------------------
# VisualizationProduct.get_line_cut
# ---------------------------------------------------------------------------


class TestGetLineCut:
    def _make_gradient(self) -> VisualizationProduct:
        """4×4 array where values[i,j] = i*4 + j (0..15)."""
        values = numpy.arange(16, dtype=numpy.float32).reshape(4, 4)
        rgba = numpy.ones((4, 4, 4), dtype=numpy.float32)
        geo = PixelGeometry(width_m=1.0, height_m=1.0)
        return VisualizationProduct('grad', values, rgba, geo, Interval[float](0.0, 15.0))

    def test_returns_linecut(self):
        vp = self._make_gradient()
        line = Line2D(Point2D(0.0, 0.0), Point2D(3.0, 0.0))
        lc = vp.get_line_cut(line)
        assert isinstance(lc, LineCut)

    def test_single_series_for_real_values(self):
        vp = self._make_gradient()
        line = Line2D(Point2D(0.0, 0.0), Point2D(3.0, 0.0))
        lc = vp.get_line_cut(line)
        assert len(lc.series) == 1
        assert lc.series[0].label == 'grad'
        assert len(lc.series[0].value) == len(lc.distance_m)

    def test_distances_are_nonneg(self):
        vp = self._make_gradient()
        line = Line2D(Point2D(0.0, 0.0), Point2D(3.0, 0.0))
        lc = vp.get_line_cut(line)
        assert all(d >= 0 for d in lc.distance_m)

    def test_distances_monotone(self):
        vp = self._make_gradient()
        line = Line2D(Point2D(0.0, 0.0), Point2D(3.0, 0.0))
        lc = vp.get_line_cut(line)
        dists = list(lc.distance_m)
        assert dists == sorted(dists)

    def test_zero_length_line(self):
        # A degenerate (zero-length) line clips to alpha ∈ [0,1] and yields
        # no midpoints between intersections, so both sequences are empty.
        vp = self._make_gradient()
        line = Line2D(Point2D(1.0, 1.0), Point2D(1.0, 1.0))
        lc = vp.get_line_cut(line)
        # Degenerate lines expand bounding-box intersection to (-inf, inf),
        # yielding no grid crossings; the result has no well-defined midpoints.
        assert isinstance(lc, LineCut)


# ---------------------------------------------------------------------------
# VisualizationProduct.get_line_cut over complex arrays
# ---------------------------------------------------------------------------


def _complex_gradient() -> numpy.ndarray:
    """4x4 complex array with varying amplitude and phase, and no wrapped-phase ambiguity."""
    amplitude = numpy.arange(1.0, 17.0, dtype=numpy.float32).reshape(4, 4)
    phase_rad = numpy.linspace(-1.5, 1.5, 16, dtype=numpy.float32).reshape(4, 4)
    return amplitude * numpy.exp(1j * phase_rad)


# A horizontal line across the top row samples pixels (0, 0..3) exactly once each.
_TOP_ROW_LINE = Line2D(Point2D(0.0, 0.5), Point2D(4.0, 0.5))


class TestGetLineCutComplex:
    @pytest.mark.parametrize('component', list(ComplexComponent))
    def test_matches_selected_component(self, component: ComplexComponent):
        # Regression: the line cut used to sample the original complex array, so every
        # component silently plotted the real part.
        arr = _complex_gradient()
        vp = visualize_complex_component(arr, _pixel_geo(), component)
        lc = vp.get_line_cut(_TOP_ROW_LINE)
        expected = component.extract_component(arr)[0, :]
        assert len(lc.series) == 1
        numpy.testing.assert_allclose(lc.series[0].value, expected, rtol=1e-6)

    @pytest.mark.parametrize('component', list(ComplexComponent))
    def test_series_label_matches_value_label(self, component: ComplexComponent):
        arr = _complex_gradient()
        vp = visualize_complex_component(arr, _pixel_geo(), component)
        lc = vp.get_line_cut(_TOP_ROW_LINE)
        assert lc.series[0].label == vp.get_value_label()

    def test_components_differ_from_each_other(self):
        arr = _complex_gradient()
        cuts = {
            component: tuple(
                visualize_complex_component(arr, _pixel_geo(), component)
                .get_line_cut(_TOP_ROW_LINE)
                .series[0]
                .value
            )
            for component in ComplexComponent
        }
        # REAL and IMAGINARY must not coincide; if they did, the sampling would be ignoring
        # the component selection entirely.
        assert cuts[ComplexComponent.REAL] != cuts[ComplexComponent.IMAGINARY]
        assert cuts[ComplexComponent.AMPLITUDE] != cuts[ComplexComponent.REAL]

    def test_applies_scalar_transform(self):
        arr = _complex_gradient()
        vp = visualize_complex_component(
            arr, _pixel_geo(), ComplexComponent.AMPLITUDE, transform=ScalarTransformation.SQRT
        )
        lc = vp.get_line_cut(_TOP_ROW_LINE)
        expected = numpy.sqrt(ComplexComponent.AMPLITUDE.extract_component(arr)[0, :])
        numpy.testing.assert_allclose(lc.series[0].value, expected, rtol=1e-6)

    def test_cylindrical_model_yields_amplitude_and_phase(self):
        arr = _complex_gradient()
        vp = visualize_complex_values(
            arr,
            _pixel_geo(),
            CylindricalColorModel.HSV_VALUE,
            amplitude_transform=ScalarTransformation.IDENTITY,
        )
        lc = vp.get_line_cut(_TOP_ROW_LINE)
        assert len(lc.series) == 2
        amplitude_series, phase_series = lc.series
        assert amplitude_series.label == 'Amplitude'
        assert phase_series.label == 'Phase [rad]'
        numpy.testing.assert_allclose(amplitude_series.value, numpy.absolute(arr)[0, :], rtol=1e-6)
        numpy.testing.assert_allclose(phase_series.value, numpy.angle(arr)[0, :], rtol=1e-5)


# ---------------------------------------------------------------------------
# VisualizationProduct.estimate_kernel_density
# ---------------------------------------------------------------------------


class TestEstimateKernelDensity:
    def test_returns_kde(self):
        vp = _make_product(rows=8, cols=8)
        box = Box2D(x=1, y=1, width=4, height=4)
        kde = vp.estimate_kernel_density(box)
        assert isinstance(kde, KernelDensityEstimate)

    def test_value_range_ordered(self):
        vp = _make_product(rows=8, cols=8)
        box = Box2D(x=0, y=0, width=8, height=8)
        kde = vp.estimate_kernel_density(box)
        assert kde.value_lower <= kde.value_upper

    def test_complex_values_use_amplitude(self):
        # Use varying complex values so amplitudes differ (KDE requires variance).
        rng = numpy.random.default_rng(7)
        values = (rng.random((8, 8)) + 1.0 + 1j * (rng.random((8, 8)) + 1.0)).astype(complex)
        rgba = numpy.ones((8, 8, 4), dtype=numpy.float32)
        amplitudes = numpy.absolute(values)
        vp = VisualizationProduct('c', values, rgba, _pixel_geo(), Interval[float](0.0, 3.0))
        box = Box2D(x=0, y=0, width=8, height=8)
        kde = vp.estimate_kernel_density(box)
        assert kde.value_lower == pytest.approx(amplitudes.min(), rel=1e-4)
        assert kde.value_upper == pytest.approx(amplitudes.max(), rel=1e-4)

    def test_box_clamped_to_image(self):
        vp = _make_product(rows=4, cols=4)
        box = Box2D(x=-10, y=-10, width=100, height=100)
        kde = vp.estimate_kernel_density(box)
        assert kde.value_lower <= kde.value_upper

    def test_phase_component_spans_phase_range(self):
        # Regression: the histogram used to hardcode amplitude for any complex array, so a
        # phase histogram silently showed amplitudes.
        rng = numpy.random.default_rng(11)
        amplitude = rng.random((8, 8)) + 10.0
        phase_rad = rng.uniform(-numpy.pi, numpy.pi, (8, 8))
        arr = amplitude * numpy.exp(1j * phase_rad)
        vp = visualize_complex_component(arr, _pixel_geo(), ComplexComponent.PHASE_RAD)
        box = Box2D(x=0, y=0, width=8, height=8)
        kde = vp.estimate_kernel_density(box)
        assert kde.value_lower == pytest.approx(phase_rad.min(), rel=1e-4)
        assert kde.value_upper == pytest.approx(phase_rad.max(), rel=1e-4)

    def test_cylindrical_model_uses_amplitude(self):
        # A histogram has one value axis, so the Complex renderer keeps showing amplitude.
        rng = numpy.random.default_rng(13)
        amplitude = rng.random((8, 8)) + 1.0
        phase_rad = rng.uniform(-numpy.pi, numpy.pi, (8, 8))
        arr = amplitude * numpy.exp(1j * phase_rad)
        vp = visualize_complex_values(
            arr,
            _pixel_geo(),
            CylindricalColorModel.HSV_VALUE,
            amplitude_transform=ScalarTransformation.IDENTITY,
        )
        box = Box2D(x=0, y=0, width=8, height=8)
        kde = vp.estimate_kernel_density(box)
        assert kde.value_lower == pytest.approx(amplitude.min(), rel=1e-4)
        assert kde.value_upper == pytest.approx(amplitude.max(), rel=1e-4)


# ---------------------------------------------------------------------------
# ComplexComponent
# ---------------------------------------------------------------------------


class TestComplexComponent:
    def _make_complex(self) -> numpy.ndarray:
        return numpy.array([[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 1j]], dtype=complex)

    def test_real(self):
        arr = self._make_complex()
        result = ComplexComponent.REAL.extract_component(arr)
        numpy.testing.assert_allclose(result, numpy.real(arr), atol=1e-6)

    def test_imaginary(self):
        arr = self._make_complex()
        result = ComplexComponent.IMAGINARY.extract_component(arr)
        numpy.testing.assert_allclose(result, numpy.imag(arr), atol=1e-6)

    def test_amplitude(self):
        arr = self._make_complex()
        result = ComplexComponent.AMPLITUDE.extract_component(arr)
        numpy.testing.assert_allclose(result, numpy.ones((2, 2)), atol=1e-6)

    def test_phase_rad(self):
        arr = numpy.array([[1 + 0j]], dtype=complex)
        result = ComplexComponent.PHASE_RAD.extract_component(arr)
        numpy.testing.assert_allclose(result, [[0.0]], atol=1e-6)

    def test_unwrapped_phase_rad_shape(self):
        arr = self._make_complex()
        result = ComplexComponent.UNWRAPPED_PHASE_RAD.extract_component(arr)
        assert result.shape == arr.shape

    def test_is_cyclic_only_for_phase(self):
        assert ComplexComponent.PHASE_RAD.is_cyclic is True
        for comp in ComplexComponent:
            if comp is not ComplexComponent.PHASE_RAD:
                assert comp.is_cyclic is False


# ---------------------------------------------------------------------------
# ScalarTransformation
# ---------------------------------------------------------------------------


class TestScalarTransformation:
    def _pos_array(self) -> numpy.ndarray:
        return numpy.array([[1.0, 4.0], [9.0, 16.0]], dtype=numpy.float32)

    def test_identity(self):
        arr = self._pos_array()
        result = ScalarTransformation.IDENTITY.transform(arr)
        numpy.testing.assert_allclose(result, arr)

    def test_sqrt(self):
        arr = self._pos_array()
        result = ScalarTransformation.SQRT.transform(arr)
        numpy.testing.assert_allclose(result, numpy.sqrt(arr), rtol=1e-5)

    def test_log2(self):
        arr = self._pos_array()
        result = ScalarTransformation.LOG2.transform(arr)
        numpy.testing.assert_allclose(result, numpy.log2(arr), rtol=1e-5)

    def test_log(self):
        arr = self._pos_array()
        result = ScalarTransformation.LOG.transform(arr)
        numpy.testing.assert_allclose(result, numpy.log(arr), rtol=1e-5)

    def test_log10(self):
        arr = self._pos_array()
        result = ScalarTransformation.LOG10.transform(arr)
        numpy.testing.assert_allclose(result, numpy.log10(arr), rtol=1e-5)

    def test_zero_input_yields_zero_for_log(self):
        arr = numpy.array([[0.0, 1.0]], dtype=numpy.float32)
        result = ScalarTransformation.LOG.transform(arr)
        assert result[0, 0] == pytest.approx(0.0)

    def test_decorate_text_identity(self):
        assert ScalarTransformation.IDENTITY.decorate_text('I') == 'I'

    def test_decorate_text_sqrt(self):
        decorated = ScalarTransformation.SQRT.decorate_text('I')
        assert 'sqrt' in decorated.lower() or '\\sqrt' in decorated

    def test_decorate_text_log2(self):
        assert 'log' in ScalarTransformation.LOG2.decorate_text('I').lower()

    def test_decorate_text_log(self):
        assert 'ln' in ScalarTransformation.LOG.decorate_text('I')

    def test_decorate_text_log10(self):
        assert 'log' in ScalarTransformation.LOG10.decorate_text('I').lower()


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------


class TestColormapHelpers:
    def test_linear_colormap_names_nonempty(self):
        names = list(linear_colormap_names())
        assert len(names) > 0

    def test_linear_colormap_names_are_strings(self):
        for name in linear_colormap_names():
            assert isinstance(name, str)

    def test_cyclic_colormap_names_nonempty(self):
        names = list(cyclic_colormap_names())
        assert len(names) > 0

    def test_get_colormap_by_name_returns_colormap(self):
        name = next(linear_colormap_names())
        cmap = get_colormap_by_name(name)
        # should be callable and return an array
        result = cmap(numpy.array([0.0, 0.5, 1.0]))
        assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# hsva_to_rgba / hlsa_to_rgba
# ---------------------------------------------------------------------------


class TestColorConversions:
    def _ones(self, shape=(3,)) -> numpy.ndarray:
        return numpy.ones(shape, dtype=numpy.float32)

    def test_hsva_to_rgba_shape_1d(self):
        h = numpy.array([0.0, 0.5, 1.0])
        s = self._ones()
        v = self._ones()
        a = self._ones()
        rgba = hsva_to_rgba(h, s, v, a)
        assert rgba.shape == (3, 4)

    def test_hsva_to_rgba_shape_2d(self):
        h = numpy.zeros((3, 3))
        s = numpy.ones((3, 3))
        v = numpy.ones((3, 3))
        a = numpy.ones((3, 3))
        rgba = hsva_to_rgba(h, s, v, a)
        assert rgba.shape == (3, 3, 4)

    def test_hsva_red_hue_zero(self):
        """Hue=0, S=1, V=1 → red."""
        h = numpy.array([0.0])
        s = numpy.array([1.0])
        v = numpy.array([1.0])
        a = numpy.array([1.0])
        rgba = hsva_to_rgba(h, s, v, a)
        numpy.testing.assert_allclose(rgba[0, :3], [1.0, 0.0, 0.0], atol=1e-5)

    def test_hlsa_to_rgba_shape(self):
        h = numpy.zeros((3, 3))
        l = numpy.full((3, 3), 0.5)
        s = numpy.ones((3, 3))
        a = numpy.ones((3, 3))
        rgba = hlsa_to_rgba(h, l, s, a)
        assert rgba.shape == (3, 3, 4)

    def test_hlsa_achromatic(self):
        """Saturation=0 → grey (r==g==b==lightness)."""
        h = numpy.array([0.0, 0.5])
        l = numpy.array([0.4, 0.4])
        s = numpy.zeros(2)
        a = numpy.ones(2)
        rgba = hlsa_to_rgba(h, l, s, a)
        numpy.testing.assert_allclose(rgba[:, 0], rgba[:, 1], atol=1e-6)
        numpy.testing.assert_allclose(rgba[:, 1], rgba[:, 2], atol=1e-6)
        numpy.testing.assert_allclose(rgba[:, 0], 0.4, atol=1e-6)

    def test_alpha_channel_preserved(self):
        h = numpy.array([0.0])
        l = numpy.array([0.5])
        s = numpy.array([1.0])
        a = numpy.array([0.7])
        rgba = hlsa_to_rgba(h, l, s, a)
        assert rgba[0, 3] == pytest.approx(0.7, rel=1e-5)


# ---------------------------------------------------------------------------
# CylindricalColorModel
# ---------------------------------------------------------------------------


class TestCylindricalColorModel:
    def _hue_x(self, shape=(4, 4)):
        hue = numpy.linspace(0, 1, numpy.prod(shape)).reshape(shape).astype(numpy.float32)
        x = numpy.linspace(0, 1, numpy.prod(shape)).reshape(shape).astype(numpy.float32)
        return hue, x

    @pytest.mark.parametrize('model', list(CylindricalColorModel))
    def test_render_rgba_shape(self, model: CylindricalColorModel):
        hue, x = self._hue_x()
        rgba = model.render_rgba(hue, x)
        assert rgba.shape == (4, 4, 4)

    @pytest.mark.parametrize('model', list(CylindricalColorModel))
    def test_render_rgba_range(self, model: CylindricalColorModel):
        hue, x = self._hue_x()
        rgba = model.render_rgba(hue, x)
        assert numpy.all(rgba >= 0.0)
        assert numpy.all(rgba <= 1.0 + 1e-6)

    def test_simple_name_format(self):
        for model in CylindricalColorModel:
            name = model.simple_name
            assert '-' in name

    def test_display_name_format(self):
        for model in CylindricalColorModel:
            name = model.display_name
            assert ' ' in name


# ---------------------------------------------------------------------------
# visualize_real_values
# ---------------------------------------------------------------------------


class TestVisualizeRealValues:
    def test_returns_visualization_product(self):
        values = numpy.ones((8, 8), dtype=numpy.float32)
        vp = visualize_real_values('I', values, _pixel_geo())
        assert isinstance(vp, VisualizationProduct)

    def test_rgba_shape(self):
        values = numpy.ones((8, 8), dtype=numpy.float32)
        vp = visualize_real_values('I', values, _pixel_geo())
        assert vp.get_image_rgba().shape == (8, 8, 4)

    def test_values_preserved(self):
        values = numpy.arange(16, dtype=numpy.float32).reshape(4, 4)
        vp = visualize_real_values('I', values, _pixel_geo())
        numpy.testing.assert_array_equal(vp.get_values(), values)

    def test_display_values_are_transformed(self):
        # The value label is decorated with the transform, so the displayed values must be
        # transformed too or the axis label lies.
        values = numpy.arange(1, 17, dtype=numpy.float32).reshape(4, 4)
        vp = visualize_real_values('I', values, _pixel_geo(), transform=ScalarTransformation.SQRT)
        (dv,) = vp.get_display_values()
        assert dv.label == vp.get_value_label()
        numpy.testing.assert_allclose(dv.values, numpy.sqrt(values), rtol=1e-6)

    def test_with_transform(self):
        values = numpy.array([[1.0, 4.0], [9.0, 16.0]], dtype=numpy.float32)
        vp = visualize_real_values('I', values, _pixel_geo(), transform=ScalarTransformation.SQRT)
        assert vp.get_value_label() != 'I'  # decorated

    def test_with_value_range(self):
        values = numpy.array([[0.0, 5.0], [10.0, 15.0]], dtype=numpy.float32)
        vp = visualize_real_values('I', values, _pixel_geo(), value_min=2.0, value_max=12.0)
        r = vp.get_color_value_range()
        assert r.lower == pytest.approx(2.0)
        assert r.upper == pytest.approx(12.0)

    def test_with_colormap_string(self):
        values = numpy.ones((4, 4), dtype=numpy.float32)
        name = next(linear_colormap_names())
        vp = visualize_real_values('I', values, _pixel_geo(), colormap=name)
        assert vp.get_image_rgba().shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# visualize_complex_component
# ---------------------------------------------------------------------------


class TestVisualizeComplexComponent:
    def _complex_array(self) -> numpy.ndarray:
        return numpy.ones((4, 4), dtype=complex) * (1 + 1j)

    @pytest.mark.parametrize('component', list(ComplexComponent))
    def test_returns_visualization_product(self, component: ComplexComponent):
        arr = self._complex_array()
        vp = visualize_complex_component(arr, _pixel_geo(), component)
        assert isinstance(vp, VisualizationProduct)

    def test_original_complex_values_stored(self):
        arr = self._complex_array()
        vp = visualize_complex_component(arr, _pixel_geo(), ComplexComponent.AMPLITUDE)
        numpy.testing.assert_array_equal(vp.get_values(), arr)


# ---------------------------------------------------------------------------
# visualize_complex_values
# ---------------------------------------------------------------------------


class TestVisualizeComplexValues:
    def _complex_array(self) -> numpy.ndarray:
        rng = numpy.random.default_rng(42)
        return (rng.random((8, 8)) + 1j * rng.random((8, 8))).astype(complex)

    @pytest.mark.parametrize('model', list(CylindricalColorModel))
    def test_returns_visualization_product(self, model: CylindricalColorModel):
        arr = self._complex_array()
        vp = visualize_complex_values(
            arr, _pixel_geo(), model, amplitude_transform=ScalarTransformation.IDENTITY
        )
        assert isinstance(vp, VisualizationProduct)

    @pytest.mark.parametrize('model', list(CylindricalColorModel))
    def test_rgba_shape(self, model: CylindricalColorModel):
        arr = self._complex_array()
        vp = visualize_complex_values(
            arr, _pixel_geo(), model, amplitude_transform=ScalarTransformation.IDENTITY
        )
        assert vp.get_image_rgba().shape == (8, 8, 4)

    def test_original_complex_values_stored(self):
        arr = self._complex_array()
        vp = visualize_complex_values(
            arr,
            _pixel_geo(),
            CylindricalColorModel.HSV_VALUE,
            amplitude_transform=ScalarTransformation.IDENTITY,
        )
        numpy.testing.assert_array_equal(vp.get_values(), arr)
