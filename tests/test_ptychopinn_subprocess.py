from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.model.processing._subprocess_protocol import ChildError
from ptychodus.model.processing.subprocess_reconstructor import SubprocessReconstructor
from ptychodus.model.ptychopinn._subprocess import _create_raw_data


_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

FIXTURES_MODULE = 'subprocess_child_fixtures'


def test_reconstruct_passes_configured_gridsize_to_spawned_child() -> None:
    payload = SimpleNamespace(
        inference_config=SimpleNamespace(model=SimpleNamespace(N=64, gridsize=2)),
        model_bundle_dir=Path('/unused'),
        n_nearest_neighbors=7,
        n_samples=3,
        reconstruct_input=object(),
    )
    adapter = SubprocessReconstructor(
        name='PtychoPINN grouping probe',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:capture_ptychopinn_grouping_call',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=lambda _parameters, _loaded: payload,
    )

    with pytest.raises(ChildError) as excinfo:
        list(adapter.reconstruct(SimpleNamespace()))  # type: ignore[arg-type]

    assert excinfo.value.child_exception_type == 'GroupingCallCapturedError'
    assert excinfo.value.child_exception is not None
    assert excinfo.value.child_exception.args == ({'N': 64, 'K': 7, 'nsamples': 3, 'gridsize': 2},)


# --------------------------- _create_raw_data ---------------------------
#
# Inference must preprocess its input the way the exported training data was
# prepared, or the model sees detector artifacts it never trained on.


NUM_POSITIONS = 4
DETECTOR_PX = 16
NUM_MODES = 3


def _make_reconstruct_input() -> ReconstructInput:
    rng = numpy.random.default_rng(5)
    pixel_geometry = PixelGeometry(width_m=1.0e-8, height_m=1.0e-8)
    probe_shape = (2, NUM_MODES, DETECTOR_PX, DETECTOR_PX)
    object_shape = (1, 32, 32)

    product = Product(
        metadata=ProductMetadata(
            name='fixture',
            comments='',
            detector_distance_m=2.0,
            photon_energy_eV=10000.0,
            probe_photon_count=0.0,
            exposure_time_s=0.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(
            [ProbePosition(i, i * 1.0e-8, i * 2.0e-8) for i in range(NUM_POSITIONS)]
        ),
        probes=ProbeSequence(
            array=rng.standard_normal(probe_shape) + 1j * rng.standard_normal(probe_shape),
            opr_weights=rng.standard_normal((NUM_POSITIONS, 2)),
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=rng.standard_normal(object_shape) + 1j * rng.standard_normal(object_shape),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
            layer_spacing_m=[],
        ),
        losses=[],
    )

    patterns = rng.integers(0, 1000, size=(NUM_POSITIONS, DETECTOR_PX, DETECTOR_PX)).astype(
        numpy.uint16
    )
    bad_pixels = numpy.zeros((DETECTOR_PX, DETECTOR_PX), dtype=bool)
    bad_pixels[4:7, 4:7] = True

    return ReconstructInput(patterns, bad_pixels, product)


@pytest.fixture
def raw_data_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture what _create_raw_data hands the external RawData constructor."""
    captured: dict[str, Any] = {}

    class FakeRawData:
        @staticmethod
        def from_coords_without_pc(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

    package = ModuleType('ptycho')
    package.__path__ = []
    raw_data_module = ModuleType('ptycho.raw_data')
    raw_data_module.RawData = FakeRawData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'ptycho', package)
    monkeypatch.setitem(sys.modules, 'ptycho.raw_data', raw_data_module)

    return captured


def test_create_raw_data_passes_every_probe_mode(raw_data_kwargs: dict[str, Any]) -> None:
    """Inference gets the full mode stack, matching what training data carries."""
    parameters = _make_reconstruct_input()

    _create_raw_data(parameters)

    probe_guess = raw_data_kwargs['probeGuess']
    assert probe_guess.shape == (NUM_MODES, DETECTOR_PX, DETECTOR_PX)
    numpy.testing.assert_array_equal(probe_guess, parameters.product.probes.get_array()[0])


def test_create_raw_data_repairs_bad_pixels(raw_data_kwargs: dict[str, Any]) -> None:
    """Masked positions are inpainted rather than handed over as raw artifacts."""
    parameters = _make_reconstruct_input()
    bad_pixels = parameters.bad_pixels
    assert bad_pixels is not None

    _create_raw_data(parameters)

    diff3d = raw_data_kwargs['diff3d']
    expected = inpaint_bad_pixels(parameters.diffraction_patterns, bad_pixels)

    numpy.testing.assert_array_equal(diff3d, expected)
    assert diff3d.dtype == parameters.diffraction_patterns.dtype
    # The repair has to actually change something, or this proves nothing.
    assert not numpy.array_equal(
        diff3d[:, bad_pixels], parameters.diffraction_patterns[:, bad_pixels]
    )
