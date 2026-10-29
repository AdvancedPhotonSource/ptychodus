"""The fluorescence enhancers as plugins: what they declare, where it persists, how they run."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock
import logging

import numpy
import pytest

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
    SubprocessFluorescenceEnhancer,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.spawn import ChildError
from ptychodus.model.plugins import PluginChooser
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.fluorescence import ENHANCER_LIBRARY, FluorescenceCore
from ptychodus.model.fluorescence.core import _BRANCHES_BY_ENHANCER
from ptychodus.model.processing import ProcessingParameterStore
from ptychodus.model.devices import DeviceSelection
from ptychodus.plugins.fluorescence import TwoStepFluorescenceEnhancer, VSPIFluorescenceEnhancer
from ptychodus.plugins.fluorescence.deconvolution import DeconvolutionMethod
from ptychodus.plugins.fluorescence.two_step import TwoStepOptions
from ptychodus.plugins.fluorescence.upscaling import UpscalingMethod
from ptychodus.plugins.fluorescence.vspi import VSPIOptions
from ptychodus.plugins.ptychozoon import PtychozoonFluorescenceEnhancer
from ptychodus.plugins.ptychozoon.options import PtychozoonOptions

_PIXEL_SIZE_M = 1.0e-7
_CANVAS_PX = 24
_PROBE_PX = 6
_SCAN_SIDE = 6


def _make_product() -> Product:
    """A product scanned on a square raster centered on the object canvas."""
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    offsets_px = numpy.linspace(-6.0, 6.0, _SCAN_SIDE)
    positions = [
        ProbePosition(index=index, x_m=x_px * _PIXEL_SIZE_M, y_m=y_px * _PIXEL_SIZE_M)
        for index, (y_px, x_px) in enumerate(
            (y_px, x_px) for y_px in offsets_px for x_px in offsets_px
        )
    ]

    yy, xx = numpy.mgrid[:_PROBE_PX, :_PROBE_PX] - (_PROBE_PX - 1) / 2
    probe_array = numpy.exp(-(yy**2 + xx**2) / 4.0).astype(complex)

    return Product(
        metadata=ProductMetadata(
            name='fluorescence',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=5.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(
            array=probe_array[numpy.newaxis, :, :],
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.ones((_CANVAS_PX, _CANVAS_PX), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def _make_input() -> FluorescenceEnhancerInput:
    rng = numpy.random.default_rng(7)
    dataset = FluorescenceDataset(
        _element_maps=[
            ElementMap(name, rng.uniform(1.0, 2.0, (_SCAN_SIDE, _SCAN_SIDE)))
            for name in ('Fe', 'Zn')
        ],
        counts_per_second_path='/cps',
        channel_names_path='/names',
    )
    return FluorescenceEnhancerInput(dataset=dataset, product=_make_product())


def _last(outputs: Iterator[FluorescenceEnhancerOutput]) -> FluorescenceEnhancerOutput:
    collected = list(outputs)
    assert collected
    return collected[-1]


def _make_core(registry: SettingsRegistry, plugins: PluginRegistry) -> FluorescenceCore:
    devices = registry.create_group('Processing').create_string_parameter('Devices', '')
    return FluorescenceCore(
        MagicMock(),
        registry,
        MagicMock(),
        MagicMock(),
        PluginChooser(plugins.fluorescence_enhancers),
        DeviceSelection(devices),
        PluginChooser(plugins.fluorescence_file_readers),
        PluginChooser(plugins.fluorescence_file_writers),
        MagicMock(),
    )


def _register(store: ProcessingParameterStore, name: str, enhancer: FluorescenceEnhancer) -> None:
    store.register_declared(
        ENHANCER_LIBRARY, name, enhancer.get_default_options(), _BRANCHES_BY_ENHANCER[name]
    )


def test_each_enhancer_persists_in_a_group_of_its_own() -> None:
    registry = SettingsRegistry()
    store = ProcessingParameterStore(registry)
    _register(store, 'two_step', TwoStepFluorescenceEnhancer())
    _register(store, 'vspi', VSPIFluorescenceEnhancer())
    _register(store, 'ptychozoon', PtychozoonFluorescenceEnhancer())

    keys = {name: set(registry[name].parameters()) for name in registry}

    assert keys == {
        'FluorescenceTwoStep': {'UpscalingMethod', 'DeconvolutionMethod', 'WienerBalance'},
        'FluorescenceVSPI': {'DampingFactor', 'MaxIterations'},
        'FluorescencePtychozoon': {
            'DampingFactor',
            'MaxIter',
            'Atol',
            'Btol',
            'CheckpointInterval',
        },
    }


def test_ptychozoon_shares_no_value_with_vspi() -> None:
    """ptychozoon is maintained separately; a field of the same name is not the same value."""
    registry = SettingsRegistry()
    store = ProcessingParameterStore(registry)
    _register(store, 'vspi', VSPIFluorescenceEnhancer())
    _register(store, 'ptychozoon', PtychozoonFluorescenceEnhancer())

    registry['FluorescenceVSPI'].parameters()['DampingFactor'].set_value_from_string('0.5')

    vspi = store.get_options(ENHANCER_LIBRARY, 'vspi')
    ptychozoon = store.get_options(ENHANCER_LIBRARY, 'ptychozoon')
    assert isinstance(vspi, VSPIOptions)
    assert isinstance(ptychozoon, PtychozoonOptions)
    assert vspi.damping_factor == 0.5
    assert ptychozoon.damping_factor == 0.0


@pytest.mark.parametrize('method', list(UpscalingMethod))
def test_two_step_runs_every_upscaling_method(method: UpscalingMethod) -> None:
    input_ = _make_input()
    options = TwoStepOptions(upscaling_method=method)

    output = _last(TwoStepFluorescenceEnhancer().enhance(input_, options=options))

    assert output.progress == output.progress_goal == len(input_.dataset)
    assert [emap.name for emap in output.dataset] == ['Fe', 'Zn']


@pytest.mark.parametrize('method', list(DeconvolutionMethod))
def test_two_step_runs_every_deconvolution_method(method: DeconvolutionMethod) -> None:
    input_ = _make_input()
    options = TwoStepOptions(deconvolution_method=method)

    output = _last(TwoStepFluorescenceEnhancer().enhance(input_, options=options))

    assert output.progress == output.progress_goal == len(input_.dataset)
    for emap in output.dataset:
        assert emap.counts_per_second.shape == (_CANVAS_PX, _CANVAS_PX)
        assert numpy.isfinite(emap.counts_per_second).all()


def test_vspi_reports_a_goal_its_progress_reaches() -> None:
    input_ = _make_input()

    outputs = list(
        VSPIFluorescenceEnhancer().enhance(input_, options=VSPIOptions(max_iterations=5))
    )

    assert [output.progress for output in outputs] == [1, 2]
    assert all(output.progress_goal == len(input_.dataset) for output in outputs)
    assert outputs[-1].dataset[0].counts_per_second.shape == (_CANVAS_PX, _CANVAS_PX)


def test_an_enhancer_refuses_options_declared_for_another() -> None:
    with pytest.raises(TypeError):
        list(TwoStepFluorescenceEnhancer().enhance(_make_input(), options=VSPIOptions()))


def test_overrides_apply_to_one_run_and_leave_settings_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = SettingsRegistry()
    core = _make_core(registry, PluginRegistry.load_plugins())

    with caplog.at_level(logging.WARNING):
        options = core.fluorescence_api._get_options(
            'vspi', {'max_iterations': 7, 'not_a_field': 1}
        )

    assert isinstance(options, VSPIOptions)
    assert options.max_iterations == 7
    assert registry['FluorescenceVSPI'].parameters()['MaxIterations'].get_value() == 100
    assert 'not_a_field' in caplog.text


def test_an_override_the_declaration_rejects_fails_before_the_run() -> None:
    core = _make_core(SettingsRegistry(), PluginRegistry.load_plugins())

    with pytest.raises(ValueError):
        core.fluorescence_api._get_options('vspi', {'max_iterations': 0})


def test_placement_is_attached_only_to_an_enhancer_run_in_a_child() -> None:
    pytest.importorskip('ptychozoon')
    plugins = PluginRegistry.load_plugins()
    _make_core(SettingsRegistry(), plugins)

    for plugin in plugins.fluorescence_enhancers:
        if isinstance(plugin.strategy, SubprocessFluorescenceEnhancer):
            assert plugin.simple_name == 'ptychozoon'
            assert plugin.strategy.device_environment() == {}
        else:
            assert not hasattr(plugin.strategy, 'device_environment')


def test_ptychozoon_runs_in_its_child() -> None:
    pytest.importorskip('ptychozoon')
    pytest.importorskip('cupy')
    input_ = _make_input()
    options = PtychozoonOptions(max_iter=4, checkpoint_interval=2)

    try:
        outputs = list(PtychozoonFluorescenceEnhancer().enhance(input_, options=options))
    except ChildError as exc:
        if exc.child_exception_type == 'RuntimeError' and 'CUDA device' in str(exc):
            pytest.skip('No CUDA device is visible to ptychozoon.')

        raise

    assert [output.progress for output in outputs] == [2, 4]
    assert all(output.progress_goal == 4 for output in outputs)
    assert [emap.name for emap in outputs[-1].dataset] == ['Fe', 'Zn']
    assert outputs[-1].dataset[0].counts_per_second.shape == (_CANVAS_PX, _CANVAS_PX)


def test_ptychozoon_says_so_when_the_selection_leaves_no_gpu() -> None:
    """ptychozoon cannot even be imported without a CUDA device, so the child says why."""
    pytest.importorskip('ptychozoon')
    pytest.importorskip('cupy')
    enhancer = PtychozoonFluorescenceEnhancer()
    enhancer.device_environment = lambda: {'CUDA_VISIBLE_DEVICES': ''}

    with pytest.raises(ChildError, match='needs a visible CUDA device'):
        list(enhancer.enhance(_make_input(), options=PtychozoonOptions()))
