"""Kickstart's parent-side options translation and its child, end to end.

Kickstart needs nothing beyond ptychodus, so the child runs for real: once called
directly, and once spawned through its library, which is what checks that the payload
pickles and that the child module resolves.
"""

from __future__ import annotations

from dataclasses import replace
import pickle
from typing import Any, cast

import numpy
import pytest

from parallax_scan_fixtures import (
    METADATA,
    OBJECT_GEOMETRY,
    PIXEL_GEOMETRY,
    PROBE_GEOMETRY,
    make_positions,
    make_probe,
    simulate_scan,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    ReconstructInput,
    ReconstructOutput,
    ReconstructorMessage,
    SubprocessOperation,
    SubprocessPayload,
    SubprocessReconstructor,
    prepare_reconstruct_input,
)
from ptychodus.plugins.ptychodus import PtychodusReconstructorLibrary
from ptychodus.plugins.ptychodus._subprocess import run_reconstruct
from ptychodus.plugins.ptychodus.options import KickstartOptions
from ptychodus.plugins.ptychodus.reconstructor import build_parameters, build_reconstructor


class _ListQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _input(*, layer_spacing_m: list[float], far_field: bool = True) -> ReconstructInput:
    num_layers = len(layer_spacing_m) + 1
    product = Product(
        metadata=replace(METADATA, far_field=far_field),
        probe_positions=ProbePositionSequence(make_positions()),
        probes=ProbeEnsemble.from_probe(make_probe(0.0)),
        object_=Object(
            array=numpy.ones(
                (num_layers, OBJECT_GEOMETRY.height_px, OBJECT_GEOMETRY.width_px), dtype=complex
            ),
            pixel_geometry=PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
            layer_spacing_m=layer_spacing_m,
        ),
        losses=[],
    )
    return prepare_reconstruct_input(simulate_scan(2.0e-3), product)


def _options() -> KickstartOptions:
    """Bins of two pixels: the fixture's bright field is only eight pixels in radius."""
    options = KickstartOptions()
    options.parallax_bin_size_px = 2
    return options


def _overlap(a: numpy.ndarray, b: numpy.ndarray) -> float:
    return float(abs(numpy.vdot(a, b)) / (numpy.linalg.norm(a) * numpy.linalg.norm(b)))


def test_the_declared_options_become_the_apis_parameters() -> None:
    options = KickstartOptions()
    options.defocus_distance_m = 1.5e-3
    options.parallax_min_defocus_dof = 2.0
    options.incident_photon_count_quantile = 0.9
    options.probe_bright_field_mad_threshold = 6.0
    options.phase_regularization_fov_fraction = 0.5

    parameters = build_parameters(options)

    assert parameters.defocus_distance_m == 1.5e-3
    assert parameters.parallax_min_defocus_dof == 2.0
    assert parameters.incident_photon_count_quantile == 0.9
    assert parameters.bright_field_mad_threshold == 6.0
    assert parameters.regularization_fov_fraction == 0.5


def test_near_field_is_refused_before_a_child_is_spawned() -> None:
    reconstructor = build_reconstructor()

    with pytest.raises(ValueError, match='far-field'):
        next(
            iter(
                reconstructor.reconstruct(
                    _input(layer_spacing_m=[], far_field=False), options=KickstartOptions()
                )
            )
        )


def test_the_child_estimates_the_product_on_the_inputs_grids() -> None:
    input_ = _input(layer_spacing_m=[1.0e-6])
    options = _options()
    reconstructor = build_reconstructor()
    payload = SubprocessPayload(
        operation=SubprocessOperation.RECONSTRUCT,
        reconstructor=reconstructor.name,
        options=reconstructor._dump_reconstruct_options(input_, options),
        input_=input_,
    )
    queue = _ListQueue()

    run_reconstruct(payload, cast(Any, queue))

    assert len(queue.items) == 1
    tag, body = queue.items[0]
    assert tag == ReconstructorMessage.OUTPUT
    output: ReconstructOutput = pickle.loads(body)
    product = output.product
    assert (output.progress, output.progress_goal) == (1, 1)
    assert product.probes.get_array().shape == (
        1,
        1,
        PROBE_GEOMETRY.height_px,
        PROBE_GEOMETRY.width_px,
    )
    assert _overlap(product.probes.get_array()[0, 0], make_probe(2.0e-3).get_array()[0]) > 0.95
    assert product.object_.get_geometry() == input_.product.object_.get_geometry()
    assert list(product.object_.layer_spacing_m) == [1.0e-6]
    numpy.testing.assert_array_equal(
        product.probe_positions.get_indexes(), input_.product.probe_positions.get_indexes()
    )


def test_the_library_spawns_a_child_that_returns_the_estimate() -> None:
    library = PtychodusReconstructorLibrary()
    assert library.is_available()
    (reconstructor,) = list(library)
    assert isinstance(reconstructor, SubprocessReconstructor)

    outputs = list(reconstructor.reconstruct(_input(layer_spacing_m=[]), options=_options()))

    assert len(outputs) == 1
    probe = outputs[0].product.probes.get_array()[0, 0]
    assert _overlap(probe, make_probe(2.0e-3).get_array()[0]) > 0.95
