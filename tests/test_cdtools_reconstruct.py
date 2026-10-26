"""cdtools' FancyPtycho, driven from a ptychodus product, end to end.

The conventions are checked against cdtools itself: a model built from the fixture
product must predict the fixture's patterns, which ptychodus simulated with its own
probe placement. That pins the corner frame, the sub-pixel shift's sign, the probe
scale and the forward model together. The child then runs for real, once called
directly and once spawned through its library.
"""

from __future__ import annotations

from dataclasses import replace
import pickle
from typing import Any

import numpy
import pytest

pytest.importorskip('cdtools')

import torch  # noqa: E402

from cdtools_fixtures import (  # noqa: E402
    PROBE_HEIGHT_PX,
    PROBE_WIDTH_PX,
    far_field_intensity,
    make_input,
    make_product,
)
from ptychodus.api.reconstruct import (  # noqa: E402
    ReconstructInput,
    ReconstructOutput,
    ReconstructorMessage,
    SubprocessOperation,
    SubprocessPayload,
)
from ptychodus.plugins.cdtools import CDToolsReconstructorLibrary  # noqa: E402
from ptychodus.plugins.cdtools._subprocess import (  # noqa: E402
    _build_model,
    _far_field_intensity,
    _object_guess,
    _object_transmission,
    run_reconstruct,
)
from ptychodus.plugins.cdtools.gauge import ProbeGauge  # noqa: E402
from ptychodus.plugins.cdtools.geometry import build_frame  # noqa: E402
from ptychodus.plugins.cdtools.options import CDToolsOptimizer, CDToolsOptions  # noqa: E402
from ptychodus.plugins.cdtools.reconstructor import build_reconstructor  # noqa: E402

_UNIT_GAUGE = ProbeGauge(to_cdtools=1.0, expected_factor=1.0, sampled_positions=0)


class _ListQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _predict(parameters: ReconstructInput, options: CDToolsOptions) -> numpy.ndarray:
    frame = build_frame(parameters.product)
    model = _build_model(parameters, frame, _UNIT_GAUGE, options)
    num_positions = len(parameters.product.probe_positions)

    with torch.no_grad():
        predicted = model.forward(
            torch.arange(num_positions),
            torch.as_tensor(frame.translations, dtype=torch.float32),
        )

    return predicted.numpy()


def _short_options(**refinement: bool) -> CDToolsOptions:
    options = CDToolsOptions()
    options.schedule.stage1.epochs = 3
    options.schedule.stage2.epochs = 2
    options.schedule.stage3.epochs = 0
    options.optimizer.num_sync_epochs = 2

    for name, value in refinement.items():
        setattr(options.refinement, name, value)

    return options


def _run(parameters: ReconstructInput, options: CDToolsOptions) -> list[ReconstructOutput]:
    queue = _ListQueue()
    payload = SubprocessPayload(
        operation=SubprocessOperation.RECONSTRUCT,
        reconstructor='FancyPtycho',
        options=build_reconstructor()._dump_reconstruct_options(parameters, options),
        input_=parameters,
    )
    run_reconstruct(payload, queue)  # type: ignore[arg-type]
    assert all(tag is ReconstructorMessage.OUTPUT for tag, _ in queue.items)
    return [pickle.loads(blob) for _, blob in queue.items]


def _probe_power(output: ReconstructOutput) -> float:
    return float(numpy.sum(numpy.abs(output.product.probes.get_array()) ** 2))


def test_cdtools_far_field_is_the_fixtures() -> None:
    rng = numpy.random.default_rng(0)
    shape = (3, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)
    wavefield = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    numpy.testing.assert_allclose(
        _far_field_intensity(wavefield), far_field_intensity(wavefield), rtol=1e-5, atol=1e-5
    )


def test_the_model_predicts_the_patterns_ptychodus_simulated() -> None:
    """Positions fall between pixels, so a flipped shift or a half-pixel slip shows."""
    parameters = make_input()
    predicted = _predict(parameters, CDToolsOptions())
    expected = parameters.diffraction_patterns
    numpy.testing.assert_allclose(predicted, expected, atol=1e-5 * expected.max())


def test_the_model_adds_the_background_and_applies_the_flat_field() -> None:
    clean = make_input()
    background = numpy.full((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), 4.0)
    flat_field = numpy.linspace(0.5, 1.5, PROBE_HEIGHT_PX * PROBE_WIDTH_PX).reshape(
        PROBE_HEIGHT_PX, PROBE_WIDTH_PX
    )
    parameters = replace(clean, background=background, flat_field=flat_field)
    expected = clean.diffraction_patterns * flat_field + background
    predicted = _predict(parameters, CDToolsOptions())
    numpy.testing.assert_allclose(predicted, expected, rtol=1e-4)


def test_the_maps_are_left_out_when_not_wanted() -> None:
    clean = make_input()
    background = numpy.full((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), 4.0)
    parameters = replace(clean, background=background, flat_field=numpy.full_like(background, 2.0))
    options = CDToolsOptions()
    options.refinement.use_background = False
    options.refinement.use_flat_field = False
    expected = clean.diffraction_patterns
    numpy.testing.assert_allclose(
        _predict(parameters, options), expected, atol=1e-5 * expected.max()
    )


def test_an_exponentiated_object_predicts_the_same_patterns() -> None:
    parameters = make_input()
    options = CDToolsOptions()
    options.model.exponentiate_obj = True
    expected = parameters.diffraction_patterns
    numpy.testing.assert_allclose(
        _predict(parameters, options), expected, atol=1e-5 * expected.max()
    )


def test_the_object_transform_round_trips() -> None:
    options = CDToolsOptions()
    options.model.exponentiate_obj = True
    transmission = make_product().object_.get_layers_flattened()
    numpy.testing.assert_allclose(
        _object_transmission(_object_guess(transmission, options), options), transmission
    )


def test_a_phase_only_object_comes_back_at_unit_transmission() -> None:
    """FancyPtycho honours phase-only only for an exponentiated object, so asking for it
    alone must still hold the modulus."""
    options = _short_options(refine_probe=False)
    options.model.phase_only = True
    object_out = _run(make_input(), options)[-1].product.object_.get_array()
    numpy.testing.assert_allclose(numpy.abs(object_out), 1.0, rtol=1e-6)


@pytest.mark.parametrize('optimizer', list(CDToolsOptimizer))
def test_a_run_streams_at_every_sync_and_every_stage_end(optimizer: CDToolsOptimizer) -> None:
    options = _short_options()
    options.optimizer.optimizer = optimizer

    if optimizer is CDToolsOptimizer.SGD:
        for stage in options.schedule.stages():
            stage.learning_rate = 1.0e-9

    outputs = _run(make_input(), options)

    assert [output.progress for output in outputs] == [2, 3, 4, 5]
    assert {output.progress_goal for output in outputs} == {5}
    assert [len(output.product.losses) for output in outputs] == [2, 3, 4, 5]


def test_the_probe_comes_back_in_ptychodus_gauge() -> None:
    parameters = make_input()
    outputs = _run(parameters, _short_options())
    power_in = float(numpy.sum(numpy.abs(parameters.product.probes.get_array()) ** 2))
    assert _probe_power(outputs[-1]) == pytest.approx(power_in, rel=0.05)


def test_what_is_not_refined_comes_back_unchanged() -> None:
    parameters = make_input()
    product_in = parameters.product
    outputs = _run(parameters, _short_options(refine_probe=False, refine_object=False))
    product_out = outputs[-1].product

    numpy.testing.assert_allclose(
        product_out.probes.get_array(), product_in.probes.get_array(), rtol=1e-5, atol=1e-4
    )
    numpy.testing.assert_allclose(
        product_out.object_.get_array(), product_in.object_.get_array(), rtol=1e-5, atol=1e-6
    )
    assert list(product_out.probe_positions) == list(product_in.probe_positions)


def test_refined_positions_move_and_keep_their_indexes() -> None:
    parameters = make_input()
    positions_in = parameters.product.probe_positions
    options = _short_options(refine_positions=True)
    positions_out = _run(parameters, options)[-1].product.probe_positions

    assert [p.index for p in positions_out] == [p.index for p in positions_in]
    assert list(positions_out) != list(positions_in)


def test_the_parent_refuses_an_empty_schedule() -> None:
    options = CDToolsOptions()
    options.schedule.stage1.epochs = 0
    options.schedule.stage2.epochs = 0
    options.schedule.stage3.epochs = 0

    with pytest.raises(ValueError, match='zero epochs'):
        build_reconstructor()._dump_reconstruct_options(make_input(), options)


def test_a_spawned_run_streams_through_the_library() -> None:
    (reconstructor,) = list(CDToolsReconstructorLibrary())
    options = _short_options()
    outputs = list(reconstructor.reconstruct(make_input(), options=options))
    assert [output.progress for output in outputs] == [2, 3, 4, 5]
