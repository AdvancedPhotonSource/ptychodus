"""Two-dataset integration test for the diffraction model.

Exercises the DiffractionDatasetRepository together with real
AssembledDiffractionDataset instances (built via the repository's factory)
to verify per-dataset bad-pixels ownership and stable index-based routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.repository import DiffractionDatasetRepository
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


def _make_repository() -> DiffractionDatasetRepository:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)
    sizer = PatternSizer(detector_settings, diffraction_settings)
    task_manager = MagicMock()
    task_monitor = MagicMock()

    def _factory(name: str) -> AssembledDiffractionDataset:
        return AssembledDiffractionDataset(
            diffraction_settings,
            sizer,
            detector_settings,
            task_manager,
            task_monitor,
            name=name,
        )

    return DiffractionDatasetRepository(factory=_factory)


def test_two_datasets_end_up_at_stable_indexes() -> None:
    repo = _make_repository()

    a = repo.create_dataset('scan_a')
    index_a = repo.insert_dataset(a)

    b = repo.create_dataset('scan_b')
    index_b = repo.insert_dataset(b)

    assert index_a == 0
    assert index_b == 1
    assert repo[0].get_name() == 'scan_a'
    assert repo[1].get_name() == 'scan_b'


def test_bad_pixels_are_per_dataset() -> None:
    repo = _make_repository()

    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)
    b = repo.create_dataset('scan_b')
    repo.insert_dataset(b)

    # Both start with all-good masks of matching shape.
    mask_a_initial = repo[0].get_bad_pixels()
    mask_b_initial = repo[1].get_bad_pixels()
    assert mask_a_initial.shape == mask_b_initial.shape
    assert not mask_a_initial.any()
    assert not mask_b_initial.any()

    # Set a distinctive mask on dataset A only.
    custom_mask_a = numpy.zeros_like(mask_a_initial)
    custom_mask_a[0, 0] = True
    repo[0].set_bad_pixels(custom_mask_a)

    # A picks it up; B is untouched.
    assert repo[0].get_bad_pixels()[0, 0]
    assert not repo[1].get_bad_pixels()[0, 0]

    # And they are not aliased.
    assert repo[0].get_bad_pixels() is not repo[1].get_bad_pixels()


def test_removing_first_dataset_shifts_indexes() -> None:
    repo = _make_repository()

    repo.insert_dataset(repo.create_dataset('scan_a'))
    b = repo.create_dataset('scan_b')
    repo.insert_dataset(b)

    repo.remove_dataset(0)

    assert len(repo) == 1
    assert repo[0] is b
    assert repo[0].get_name() == 'scan_b'


def test_reset_bad_pixels_restores_default() -> None:
    repo = _make_repository()
    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)

    mask = numpy.zeros_like(a.get_bad_pixels())
    mask[1, 1] = True
    a.set_bad_pixels(mask)
    assert a.get_bad_pixels()[1, 1]

    a.reset_bad_pixels()
    assert not a.get_bad_pixels().any()


def test_create_unique_name_prevents_collision_after_insert() -> None:
    repo = _make_repository()
    repo.insert_dataset(repo.create_dataset('scan'))
    repo.insert_dataset(repo.create_dataset('scan'))
    assert [ds.get_name() for ds in repo] == ['scan', 'scan-1']
