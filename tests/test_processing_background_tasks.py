"""The reconstruct and train tasks run their reconstructor in a spawned child.

What a reconstructor yields comes back; what it does to itself does not. So the train
task, not the reconstructor, loads the model a finished run names, and a stopped run
loads nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.model.processing.monitor import ReconstructBackgroundTask, TrainBackgroundTask

# The child resolves the fixtures module by name against the sys.path it inherits.
_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from subprocess_reconstructor_fixtures import EmptyOptions, FakeTrainableReconstructor  # noqa: E402


def _reconstruct_input() -> ReconstructInput:
    pixel_geometry = PixelGeometry(1.0e-9, 1.0e-9)
    product = Product(
        metadata=ProductMetadata(
            name='fake',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10_000.0,
            probe_photon_count=1.0,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence([]),
        probes=ProbeEnsemble(
            array=numpy.zeros((1, 1, 4, 4), dtype=numpy.complex64),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.zeros((1, 4, 4), dtype=numpy.complex64),
            layer_spacing_m=[],
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(0.0, 0.0),
        ),
        losses=[],
    )
    return ReconstructInput(
        diffraction_patterns=numpy.zeros((1, 4, 4), dtype=numpy.float32),
        bad_pixels=numpy.zeros((4, 4), dtype=numpy.bool_),
        product=product,
    )


def _task_monitor(*, is_stopping: bool = False) -> MagicMock:
    monitor = MagicMock()
    monitor.__enter__.return_value = monitor
    monitor.is_stopping = is_stopping
    return monitor


def test_reconstruct_streams_what_the_child_yields() -> None:
    monitor = _task_monitor()
    product_item = MagicMock()
    task = ReconstructBackgroundTask(
        monitor,
        FakeTrainableReconstructor(num_outputs=3),
        _reconstruct_input(),
        product_item,
        None,
        EmptyOptions(),
        dict,
    )

    task()

    assert monitor.update_product.call_count == 3
    monitor.update_progress.assert_called_with(3, 3)


def test_train_loads_the_model_the_child_names(tmp_path: Path) -> None:
    reconstructor = FakeTrainableReconstructor()
    task = TrainBackgroundTask(
        _task_monitor(),
        reconstructor,
        _reconstruct_input(),
        tmp_path,
        tmp_path,
        EmptyOptions(),
        dict,
    )

    task()

    assert not reconstructor.trained_in_this_process
    assert reconstructor.loaded_model == tmp_path / 'model.fake'


def test_a_stopped_training_run_loads_nothing(tmp_path: Path) -> None:
    reconstructor = FakeTrainableReconstructor()
    task = TrainBackgroundTask(
        _task_monitor(is_stopping=True),
        reconstructor,
        _reconstruct_input(),
        tmp_path,
        tmp_path,
        EmptyOptions(),
        dict,
    )

    task()

    assert not reconstructor.is_model_loaded()
