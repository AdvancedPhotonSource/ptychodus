"""Which per-position counts the illumination map and the OPR analysis use.

Behaviors verified:
  - the illumination map weights positions by their I0, and equally without it, never
    by the detector totals
  - the OPR analysis compares the weights with the I0, and with the detector totals of
    the bound data's moments when no I0 was recorded
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy
import pytest

from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.model.analysis.illumination import IlluminationMapper
from ptychodus.model.analysis.opr import OPRModeAnalyzer


def _positions(i0: list[float] | None) -> ProbePositionSequence:
    return ProbePositionSequence(
        [ProbePosition(index, 0.0, 0.0, None if i0 is None else i0[index]) for index in range(3)]
    )


def _moments(totals: list[float]) -> DiffractionMoments:
    frame = numpy.zeros((2, 2))
    rows = numpy.zeros(len(totals))
    return DiffractionMoments(
        num_patterns=len(totals),
        minimum_frame=frame,
        mean_frame=frame,
        maximum_frame=frame,
        bad_pixels=numpy.zeros((2, 2), dtype=bool),
        indexes=numpy.arange(len(totals)),
        total_counts=numpy.asarray(totals, dtype=numpy.float64),
        center_of_mass_x_px=rows,
        center_of_mass_y_px=rows.copy(),
    )


def _repository(positions: ProbePositionSequence, moments: DiffractionMoments | None) -> MagicMock:
    item = MagicMock()
    item.get_product.return_value.probe_positions = positions
    item.get_probe_positions_item.return_value.get_probe_positions.return_value = positions
    item.get_dataset.return_value.get_diffraction_moments.return_value = moments
    repository = MagicMock()
    repository.__getitem__.return_value = item
    return repository


@pytest.mark.parametrize(
    ('i0', 'expected'), [([1.0, 2.0, 3.0], {0: 1.0, 1: 2.0, 2: 3.0}), (None, {})]
)
def test_the_illumination_map_is_weighted_by_i0_only(
    i0: list[float] | None, expected: dict[int, float]
) -> None:
    repository = _repository(_positions(i0), _moments([10.0, 20.0, 30.0]))

    with patch('ptychodus.model.analysis.illumination.compute_illumination_map') as compute:
        IlluminationMapper(repository).map(0)

    assert compute.call_args.kwargs['probe_photon_counts_by_index'] == expected


def test_the_opr_analysis_compares_with_i0_when_recorded() -> None:
    repository = _repository(_positions([1.0, 2.0, 3.0]), _moments([10.0, 20.0, 30.0]))

    counts = OPRModeAnalyzer(repository).get_measured_photon_counts(0)

    assert counts is not None
    numpy.testing.assert_array_equal(counts, [1.0, 2.0, 3.0])


def test_the_opr_analysis_falls_back_to_the_detector_totals() -> None:
    repository = _repository(_positions(None), _moments([10.0, 20.0]))

    counts = OPRModeAnalyzer(repository).get_measured_photon_counts(0)

    assert counts is not None
    numpy.testing.assert_array_equal(counts, [10.0, 20.0, numpy.nan])


def test_the_opr_analysis_has_nothing_without_i0_or_moments() -> None:
    repository = _repository(_positions(None), None)

    assert OPRModeAnalyzer(repository).get_measured_photon_counts(0) is None
