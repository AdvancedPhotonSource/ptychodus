"""Tests for ptychodus-generate-training-data, run through ``main`` on a tiny prepared directory."""

from __future__ import annotations

from pathlib import Path
import logging
import sys

import numpy
import pytest

from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    StandardFileLayout,
    load_diffraction_data,
    load_product,
    save_diffraction_data,
    save_product,
)
from ptychodus.api.object import Object, ObjectCenter, compute_object_geometry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.cli import generate_training_data
from ptychodus.cli.generate_training_data import _compute_canvas_geometry, _TrainingDirectory
from ptychodus.plugins.ptychopinn.training_file import find_ptychopinn_training_archives

_PIXEL_SIZE_M = 75e-9
_STEP_M = 4 * _PIXEL_SIZE_M


def _make_product() -> Product:
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    y, x = numpy.mgrid[-8:8, -8:8]
    probe = 30.0 * numpy.exp(-(x**2 + y**2) / 18.0).astype(complex)
    positions = [
        ProbePosition(index=3 * row + column, x_m=(column - 1) * _STEP_M, y_m=(row - 1) * _STEP_M)
        for row in range(3)
        for column in range(3)
    ]
    return Product(
        metadata=ProductMetadata(
            name='tiny',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10_000.0,
            probe_photon_count=1.0,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(array=probe[None], opr_weights=None, pixel_geometry=pixel_geometry),
        object_=Object(
            array=numpy.full((40, 40), 0.9 * numpy.exp(0.3j)),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    directory = tmp_path / 'scan'
    directory.mkdir()
    product = _make_product()
    data = generate_diffraction_data(product, numpy.random.default_rng(0))
    save_diffraction_data(StandardFileLayout.DIFFRACTION.path(directory), data)
    save_product(StandardFileLayout.PRODUCT.path(directory), product)
    return directory


def _run(monkeypatch: pytest.MonkeyPatch, *arguments: str | Path) -> ExitCode:
    argv = ['ptychodus-generate-training-data', *(str(argument) for argument in arguments)]
    monkeypatch.setattr(sys, 'argv', argv)
    return generate_training_data.main()


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


class TestMeasuredExport:
    def test_writes_the_paired_scan(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / 'training'

        assert _run(monkeypatch, '-i', scan, '-o', output, '--file-type', 'PtychoPINN') == 0
        assert _names(output) == ['train_data_0000.npz']

        expected = prepare_reconstruct_input(
            load_diffraction_data(StandardFileLayout.DIFFRACTION.path(scan)),
            load_product(StandardFileLayout.PRODUCT.path(scan)),
        )

        with numpy.load(output / 'train_data_0000.npz') as archive:
            numpy.testing.assert_array_equal(archive['diff3d'], expected.diffraction_patterns)
            assert len(archive['xcoords']) == 9

    def test_applies_the_index_filter(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / 'training'
        arguments = ('-i', scan, '-o', output, '--file-type', 'PtychoPINN')

        assert _run(monkeypatch, *arguments, '--index-filter', 'even') == 0

        with numpy.load(output / 'train_data_0000.npz') as archive:
            assert len(archive['xcoords']) == 5


class TestSyntheticGeneration:
    def test_writes_count_files_without_diffraction_data(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        StandardFileLayout.DIFFRACTION.path(scan).unlink()
        output = tmp_path / 'training'

        status = _run(
            monkeypatch,
            *('-i', scan, '-o', output, '--file-type', 'PtychoPINN'),
            *('--synthetic', 'dead_leaves', '--count', '3', '--seed', '1'),
        )

        assert status == 0
        assert _names(output) == [f'train_data_{index:04d}.npz' for index in range(3)]
        assert find_ptychopinn_training_archives(output) == sorted(output.iterdir())

        with numpy.load(output / 'train_data_0000.npz') as first:
            with numpy.load(output / 'train_data_0001.npz') as second:
                assert first['diff3d'].shape == (9, 16, 16)
                assert not numpy.array_equal(first['objectGuess'], second['objectGuess'])

    def test_same_seed_and_index_reproduce(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        arguments = ('--file-type', 'PtychoPINN', '--synthetic', 'dead_leaves', '--seed', '5')

        for name in ('a', 'b'):
            assert _run(monkeypatch, '-i', scan, '-o', tmp_path / name, *arguments) == 0

        # a continued run with the same seed adds new data under the next index
        assert _run(monkeypatch, '-i', scan, '-o', tmp_path / 'a', *arguments) == 0

        with (
            numpy.load(tmp_path / 'a' / 'train_data_0000.npz') as a0,
            numpy.load(tmp_path / 'b' / 'train_data_0000.npz') as b0,
            numpy.load(tmp_path / 'a' / 'train_data_0001.npz') as a1,
        ):
            numpy.testing.assert_array_equal(a0['diff3d'], b0['diff3d'])
            numpy.testing.assert_array_equal(a0['objectGuess'], b0['objectGuess'])
            assert not numpy.array_equal(a0['objectGuess'], a1['objectGuess'])

    def test_jitter_moves_positions(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / 'training'
        arguments = ('-i', scan, '-o', output, '--file-type', 'PtychoPINN')

        assert _run(monkeypatch, *arguments, '--synthetic', 'dead_leaves', '--seed', '2') == 0
        assert (
            _run(
                monkeypatch,
                *arguments,
                *('--synthetic', 'dead_leaves', '--seed', '2', '--position-jitter-m', '1e-7'),
            )
            == 0
        )

        with (
            numpy.load(output / 'train_data_0000.npz') as nominal,
            numpy.load(output / 'train_data_0001.npz') as jittered,
        ):
            offsets_px = jittered['xcoords'] - nominal['xcoords']
            assert numpy.all(offsets_px != 0.0)
            assert numpy.abs(offsets_px).max() < 10 * 1e-7 / _PIXEL_SIZE_M

    def test_canvas_grows_to_cover_jittered_positions(self) -> None:
        product = _make_product()
        far = ProbePosition(index=99, x_m=50 * _PIXEL_SIZE_M, y_m=-30 * _PIXEL_SIZE_M)
        positions = ProbePositionSequence([*product.probe_positions, far])

        canvas = _compute_canvas_geometry(product, positions)
        required = compute_object_geometry(positions, product.probes.get_geometry())

        assert canvas.center_x_m == product.object_.get_geometry().center_x_m
        assert abs(required.center_x_m - canvas.center_x_m) + 0.5 * required.width_m <= (
            0.5 * canvas.width_m + 1e-15
        )
        assert abs(required.center_y_m - canvas.center_y_m) + 0.5 * required.height_m <= (
            0.5 * canvas.height_m + 1e-15
        )
        assert _compute_canvas_geometry(product, product.probe_positions).width_px == 40

    def test_print_options_round_trips(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        scan: Path,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, '--print-options', 'dead_leaves')

        assert exit_info.value.code == 0
        document = tmp_path / 'leaves.yaml'
        document.write_text(
            capsys.readouterr().out.replace(
                'num_subsamples_per_axis: 2', 'num_subsamples_per_axis: 1'
            )
        )
        output = tmp_path / 'training'

        status = _run(
            monkeypatch,
            *('-i', scan, '-o', output, '--file-type', 'PtychoPINN'),
            *('--synthetic', 'dead_leaves', '--options', document),
        )

        assert status == 0
        assert _names(output) == ['train_data_0000.npz']

    def test_refuses_options_for_another_strategy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        scan: Path,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit):
            _run(monkeypatch, '--print-options', 'random')

        document = tmp_path / 'random.yaml'
        document.write_text(capsys.readouterr().out)

        with pytest.raises(SystemExit) as exit_info:
            _run(
                monkeypatch,
                *('-i', scan, '-o', tmp_path / 'training', '--file-type', 'PtychoPINN'),
                *('--synthetic', 'dead_leaves', '--options', document),
            )

        assert exit_info.value.code == 2


class TestArguments:
    @pytest.mark.parametrize(
        'extra',
        [
            ('--synthetic', 'stxm_dpc'),
            ('--synthetic', 'no_such_strategy'),
            ('--count', '2'),
            ('--seed', '1'),
            ('--position-jitter-m', '1e-7'),
            ('--synthetic', 'dead_leaves', '--index-filter', 'odd'),
            ('--synthetic', 'dead_leaves', '--position-jitter-m', '-1'),
        ],
    )
    def test_refuses_flags_out_of_place(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path, extra: tuple[str, ...]
    ) -> None:
        output = tmp_path / 'training'

        with pytest.raises(SystemExit) as exit_info:
            _run(monkeypatch, '-i', scan, '-o', output, '--file-type', 'PtychoPINN', *extra)

        assert exit_info.value.code == 2
        assert not output.exists()

    def test_refuses_an_unknown_file_type(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            _run(monkeypatch, '-i', scan, '-o', tmp_path / 'out', '--file-type', 'NoSuchFormat')

    def test_refuses_the_input_directory_as_output(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path
    ) -> None:
        with pytest.raises(SystemExit):
            _run(monkeypatch, '-i', scan, '-o', scan, '--file-type', 'PtychoPINN')


class TestNoOverwrite:
    def test_measured_and_synthetic_share_numbering(
        self, monkeypatch: pytest.MonkeyPatch, scan: Path, tmp_path: Path
    ) -> None:
        output = tmp_path / 'training'
        arguments = ('-i', scan, '-o', output, '--file-type', 'PtychoPINN')
        sentinel = output / 'train_data_0000.npz'
        output.mkdir()
        sentinel.write_bytes(b'keep me')

        assert _run(monkeypatch, *arguments) == 0
        assert _run(monkeypatch, *arguments, '--synthetic', 'dead_leaves', '--count', '2') == 0

        assert _names(output) == [f'train_data_{index:04d}.npz' for index in range(4)]
        assert sentinel.read_bytes() == b'keep me'

    def test_skips_a_held_reservation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        scan: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / 'training'
        held = output / '.staging-train_data_0000'
        held.mkdir(parents=True)

        with caplog.at_level(logging.WARNING):
            status = _run(
                monkeypatch,
                *('-i', scan, '-o', output, '--file-type', 'PtychoPINN'),
                *('--synthetic', 'dead_leaves'),
            )

        assert status == 0
        assert _names(output) == ['.staging-train_data_0000', 'train_data_0001.npz']
        assert '.staging-train_data_0000' in caplog.text

    def test_a_collision_links_nothing(self, tmp_path: Path) -> None:
        directory = _TrainingDirectory(tmp_path, 'pair')
        index, staging = directory.reserve()
        (staging / f'{directory.name(index)}_dp.hdf5').write_bytes(b'new pattern')
        (staging / f'{directory.name(index)}_para.hdf5').write_bytes(b'new parameter')
        planted = tmp_path / f'{directory.name(index)}_para.hdf5'
        planted.write_bytes(b'existing')

        with pytest.raises(FileExistsError):
            directory.commit(staging)

        assert planted.read_bytes() == b'existing'
        assert not (tmp_path / f'{directory.name(index)}_dp.hdf5').exists()

    def test_numbering_follows_every_file_of_a_pair(self, tmp_path: Path) -> None:
        (tmp_path / 'pair_0007_para.hdf5').write_bytes(b'')
        (tmp_path / 'pair_extra.npz').write_bytes(b'')
        (tmp_path / 'other_0042.npz').write_bytes(b'')

        index, staging = _TrainingDirectory(tmp_path, 'pair').reserve()

        assert index == 8
        assert staging == tmp_path / '.staging-pair_0008'
