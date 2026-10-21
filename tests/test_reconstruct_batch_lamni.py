"""The settings file and scan selection behind ``ptychodus-reconstruct-batch-lamni``.

The settings file is the whole command-line surface of a published command, and it is
also the record archived beside every reconstruction, so a key that silently resolves to
the wrong thing is both a wrong run and a wrong account of it. These tests pin the
defaults, the cross-field rules, and the two places where a value is reinterpreted rather
than copied: the affine matrix and the multislice depth grid.

Device enumeration and the fan-out itself are not covered here -- they need GPUs and real
beamline data. What is covered is everything that decides *what* a run would do.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip('ptychi')

from ptychodus.api.devices import Device  # noqa: E402
from ptychodus.api.exit_codes import ExitCode  # noqa: E402
from ptychodus.api.object import compute_uniform_layer_spacing_m  # noqa: E402
from ptychodus.cli import reconstruct_batch_lamni  # noqa: E402
from ptychodus.cli._batch_lamni_settings import (  # noqa: E402
    Batch,
    BatchSettings,
    Paths,
    dump_batch_settings,
    load_batch_settings,
    load_batch_settings_from_text,
    record_payload,
    select_label,
    select_records,
)
from ptychodus.cli.reconstruct_batch_lamni import BatchResult, plan_batch_lamni  # noqa: E402
from ptychodus.plugins.aps31id_lamni.scan_table import APS31IDEScanRecord  # noqa: E402

_SCAN_TABLE = Path(__file__).parent / 'data' / 'lamni' / 'tomography_scannumbers.txt'

_MINIMAL = """
paths:
  experiment_folder: /experiment
  output_products_folder: /output
"""


def _settings(body: str = '') -> BatchSettings:
    return load_batch_settings_from_text(_MINIMAL + body)


def _record(scan_no: int, *, label: str = 'tomo', measurement_id: int = 0) -> APS31IDEScanRecord:
    return APS31IDEScanRecord(
        scan_no=scan_no,
        golden_angle_deg=0.0,
        encoder_angle_deg=float(scan_no) / 10.0,
        measurement_id=measurement_id,
        subtomo_no=0,
        detector_position=0,
        label=label,
    )


# --- the settings file -----------------------------------------------------------------


def test_a_minimal_file_is_valid_and_every_other_section_defaults() -> None:
    """Only the two paths are required; a first run needs nothing else decided."""
    settings = _settings()

    assert settings.paths.experiment_folder == Path('/experiment')
    assert settings.initial_probe.probe_source == 'simulated'
    assert settings.initial_object.object_source == 'simulated'
    assert settings.initial_object.num_slices == 1
    assert settings.positions.affine_matrix is None
    assert settings.misc.save_diffraction


def test_skipping_finished_scans_is_the_default() -> None:
    """A 1400-projection run that dies partway must resume by re-running the command."""
    assert not _settings().batch.overwrite_existing_products


def test_an_unknown_key_is_rejected() -> None:
    """Nothing but a person writes this file, so an unresolvable key is a typo."""
    with pytest.raises(ValueError):
        _settings('batch:\n  scan_strt: 1000\n')


def test_an_unknown_key_in_a_nested_section_is_rejected() -> None:
    with pytest.raises(ValueError):
        _settings('initial_object:\n  object_padding: 8\n')


def test_an_empty_file_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_batch_settings_from_text('')


def test_a_non_mapping_file_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_batch_settings_from_text('- not a mapping\n')


def test_paths_are_required() -> None:
    with pytest.raises(ValueError):
        load_batch_settings_from_text('batch:\n  scan_start: 1\n')


def test_load_batch_settings_reads_a_file(tmp_path: Path) -> None:
    settings_file = tmp_path / 'batch.yaml'
    settings_file.write_text(_MINIMAL)

    assert load_batch_settings(settings_file).paths.output_products_folder == Path('/output')


def test_a_probe_source_of_file_requires_a_path() -> None:
    with pytest.raises(ValueError):
        _settings('initial_probe:\n  probe_source: file\n')


def test_an_object_source_of_file_requires_a_path() -> None:
    with pytest.raises(ValueError):
        _settings('initial_object:\n  object_source: file\n')


def test_an_inverted_scan_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        _settings('batch:\n  scan_start: 50\n  scan_end: 10\n')


def test_the_warm_start_sources_are_independent() -> None:
    """Either, both or neither may come from a file."""
    settings = _settings(
        'initial_probe:\n  probe_source: file\n  probe_from_file:\n    file_path: /probe.h5\n'
    )

    assert settings.initial_probe.probe_source == 'file'
    assert settings.initial_object.object_source == 'simulated'


# --- multislice ------------------------------------------------------------------------


def test_multislice_requires_a_positive_thickness() -> None:
    """Zero thickness is a legal object whose slices all sit at one depth.

    ``compute_uniform_layer_spacing_m`` returns the right number of zero-width gaps, so
    nothing downstream can tell that apart from a real depth grid: the run would be
    multislice in name with no propagation between slices. Refuse it instead.
    """
    with pytest.raises(ValueError):
        _settings('initial_object:\n  num_slices: 3\n')

    with pytest.raises(ValueError):
        _settings('initial_object:\n  num_slices: 3\n  object_thickness_m: 0.0\n')


def test_a_single_slice_needs_no_thickness() -> None:
    assert _settings('initial_object:\n  num_slices: 1\n').initial_object.num_slices == 1


def test_the_depth_grid_spans_the_requested_thickness() -> None:
    settings = _settings('initial_object:\n  num_slices: 4\n  object_thickness_m: 1.2e-06\n')
    spacing_m = compute_uniform_layer_spacing_m(
        settings.initial_object.object_thickness_m or 0.0,
        settings.initial_object.num_slices,
    )

    assert len(spacing_m) == settings.initial_object.num_slices - 1
    assert sum(spacing_m) == pytest.approx(1.2e-06)


# --- the affine matrix -----------------------------------------------------------------


def test_a_two_by_two_matrix_widens_with_zero_translation() -> None:
    settings = _settings('positions:\n  affine_matrix: [[1.5, 0.25], [-0.5, 2.0]]\n')
    transform = settings.positions.get_transform()

    assert transform is not None
    assert (transform.a00, transform.a01, transform.a02) == (1.5, 0.25, 0.0)
    assert (transform.a10, transform.a11, transform.a12) == (-0.5, 2.0, 0.0)


def test_a_two_by_three_matrix_keeps_its_translation() -> None:
    settings = _settings('positions:\n  affine_matrix: [[1.0, 0.0, 3.0], [0.0, 1.0, -4.0]]\n')
    transform = settings.positions.get_transform()

    assert transform is not None
    assert transform.a02 == 3.0
    assert transform.a12 == -4.0


def test_no_matrix_means_no_transform() -> None:
    assert _settings().positions.get_transform() is None


@pytest.mark.parametrize(
    'matrix',
    [
        '[[1.0, 0.0]]',  # one row
        '[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]',  # three rows
        '[[1.0], [0.0]]',  # too narrow
        '[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]',  # too wide
        '[[1.0, 0.0], [0.0, 1.0, 0.0]]',  # ragged
    ],
)
def test_a_misshapen_affine_matrix_is_rejected(matrix: str) -> None:
    with pytest.raises(ValueError):
        _settings(f'positions:\n  affine_matrix: {matrix}\n')


# --- the output tree -------------------------------------------------------------------


def test_the_output_directory_is_flat_and_matches_the_raw_scan_name() -> None:
    """Five digits, as the raw per-scan directory is named, and no label level."""
    settings = _settings()

    assert settings.output_directory(1000) == Path('/output/S01000')
    assert settings.output_directory(7) == Path('/output/S00007')


def test_the_output_directory_keeps_sorting_past_scan_9999() -> None:
    settings = _settings()
    names = [settings.output_directory(n).name for n in (9999, 10000, 12345)]

    assert names == sorted(names)


# --- scan selection --------------------------------------------------------------------


def test_the_scan_range_is_inclusive_at_both_ends() -> None:
    records = [_record(n) for n in (999, 1000, 1500, 2400, 2401)]
    selected = select_records(records, Batch(scan_start=1000, scan_end=2400), 'tomo')

    assert [r.scan_no for r in selected] == [1000, 1500, 2400]


def test_excluded_scans_are_dropped() -> None:
    records = [_record(n) for n in (1000, 1010, 1020)]
    selected = select_records(records, Batch(exclude_scans=[1010]), 'tomo')

    assert [r.scan_no for r in selected] == [1000, 1020]


def test_an_open_ended_range_selects_everything_with_the_label() -> None:
    records = [_record(1000), _record(2000, label='other')]
    selected = select_records(records, Batch(), 'tomo')

    assert [r.scan_no for r in selected] == [1000]


def test_selection_is_sorted_by_scan_number() -> None:
    """The table is not guaranteed sorted, and the logs read in acquisition order."""
    records = [_record(n) for n in (1200, 1000, 1100)]
    selected = select_records(records, Batch(), 'tomo')

    assert [r.scan_no for r in selected] == [1000, 1100, 1200]


def test_a_scan_absent_from_the_table_is_not_selected() -> None:
    """The table is where a scan's rotation angle comes from, so a gap stays a gap."""
    records = [_record(1000), _record(1002)]
    selected = select_records(records, Batch(scan_start=1000, scan_end=1002), 'tomo')

    assert [r.scan_no for r in selected] == [1000, 1002]


def test_select_label_takes_the_largest_tomogram_when_none_is_named() -> None:
    records = [_record(1, label='small'), _record(2), _record(3)]

    assert select_label(records, None) == 'tomo'


def test_select_label_rejects_a_label_the_table_does_not_hold() -> None:
    with pytest.raises(ValueError):
        select_label([_record(1)], 'absent')


def test_select_label_rejects_an_empty_table() -> None:
    with pytest.raises(ValueError):
        select_label([], None)


# --- the worker payload ----------------------------------------------------------------


def test_the_record_payload_is_plain_data() -> None:
    """A payload holding a live object would import its module -- and torch with it -- in
    the child, during argument unpickling, before ``CUDA_VISIBLE_DEVICES`` is set.
    """
    payload = record_payload(_record(1234))

    assert set(payload) == {'scan_no', 'label', 'encoder_angle_deg', 'text'}
    assert all(isinstance(value, (int, float, str)) for value in payload.values())


# --- device mapping --------------------------------------------------------------------
#
# The probe and the worker count devices differently, and conflating them sends a worker
# to a GPU the scheduler never allocated. The probe inherits CUDA_VISIBLE_DEVICES, so its
# ordinals are positions *within* that mask; the value a worker sets is read against the
# physical devices. These pin the composition between the two.


def test_the_fanout_runs_one_worker_per_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU is a device to the shared probe, but not something to fan out across."""
    probed = [
        Device('cpu', 'CPU'),
        Device('cuda:0', 'A'),
        Device('cuda:1', 'B'),
    ]
    monkeypatch.setattr(reconstruct_batch_lamni, 'probe_devices', lambda: probed)

    assert [d.identifier for d in reconstruct_batch_lamni.visible_accelerators()] == [
        'cuda:0',
        'cuda:1',
    ]


def test_a_cpu_only_host_fans_out_across_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reconstruct_batch_lamni, 'probe_devices', lambda: [Device('cpu', 'CPU')])
    assert reconstruct_batch_lamni.visible_accelerators() == []


# --- the Python API --------------------------------------------------------------------
#
# A caller that already holds a BatchSettings drives the same batch the command does. What
# is pinned here is everything that decides *what* such a run would do: the selection it
# resolves, the settings text it archives when there is no file to copy, and the account it
# returns. The fan-out itself still needs GPUs and real data.


def _experiment(
    tmp_path: Path, *, scan_start: int | None = None, scan_end: int | None = None
) -> BatchSettings:
    """Settings over a tmp experiment tree holding the real scan table."""
    dat_files = tmp_path / 'experiment' / 'data' / 'dat-files'
    dat_files.mkdir(parents=True)
    shutil.copy(_SCAN_TABLE, dat_files / 'tomography_scannumbers.txt')

    return BatchSettings(
        paths=Paths(
            experiment_folder=tmp_path / 'experiment',
            output_products_folder=tmp_path / 'out',
        ),
        batch=Batch(label='BOREAS_3d_50um', scan_start=scan_start, scan_end=scan_end),
    )


def test_the_plan_selects_one_tomogram_in_acquisition_order(tmp_path: Path) -> None:
    selected = plan_batch_lamni(_experiment(tmp_path))

    assert [record.scan_no for record in selected] == [188, 189, 190, 368, 369]


def test_the_plan_honors_the_scan_range(tmp_path: Path) -> None:
    selected = plan_batch_lamni(_experiment(tmp_path, scan_start=189, scan_end=368))

    assert [record.scan_no for record in selected] == [189, 190, 368]


def test_the_plan_carries_the_rotation_angle_each_product_is_stamped_with(
    tmp_path: Path,
) -> None:
    selected = plan_batch_lamni(_experiment(tmp_path, scan_start=188, scan_end=188))

    assert selected[0].encoder_angle_deg == pytest.approx(-90.0461)


def test_an_empty_selection_is_an_error_rather_than_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_batch_lamni(_experiment(tmp_path, scan_start=9000))


def test_a_folder_that_is_not_an_experiment_directory_is_an_error(tmp_path: Path) -> None:
    settings = BatchSettings(
        paths=Paths(experiment_folder=tmp_path, output_products_folder=tmp_path / 'out')
    )

    with pytest.raises(ValueError):
        plan_batch_lamni(settings)


# --- the archived settings -------------------------------------------------------------


def test_dumped_settings_reload_unchanged() -> None:
    """A run built from an object archives a rendering of it, and workers rebuild from it.

    Both copies would otherwise describe a different run than the one that happened.
    """
    settings = _settings()

    assert load_batch_settings_from_text(dump_batch_settings(settings)) == settings


def test_dumped_settings_reload_unchanged_with_every_section_set() -> None:
    settings = _settings(
        """
batch:
  label: sample_A
  scan_start: 1000
  scan_end: 2400
  exclude_scans: [1140]
  overwrite_existing_products: true
initial_probe:
  probe_source: file
  probe_from_file:
    file_path: /probe.h5
  num_probe_modes: 4
initial_object:
  num_slices: 3
  object_thickness_m: 1.5e-06
positions:
  affine_matrix: [[1.0, 0.0, 3.0], [0.0, 1.0, -4.0]]
metadata_overrides:
  detector_distance_m: 1.92
  probe_energy_eV: 8800.0
diffraction:
  crop_extent_px: 128
  bad_pixels_file: /mask.npy
misc:
  num_sync_epochs: 10
  continue_on_error: true
"""
    )

    assert load_batch_settings_from_text(dump_batch_settings(settings)) == settings


# --- the result ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('failed', 'reason', 'expected'),
    [
        ((), None, ExitCode.SUCCESS),
        ((1042,), None, ExitCode.FAILURE),
        ((), 'SIGINT', ExitCode.CANCELLED),
        # A cancellation the first failure requested still reports as cancelled, because
        # the scans after it were never attempted.
        ((1042,), 'a scan failed', ExitCode.CANCELLED),
    ],
)
def test_the_exit_code_reports_cancellation_ahead_of_failure(
    failed: tuple[int, ...], reason: str | None, expected: ExitCode
) -> None:
    result = BatchResult(
        selected=(1042,),
        succeeded=(),
        skipped=(),
        failed=failed,
        cancelled=(),
        cancellation_reason=reason,
    )

    assert result.exit_code == expected


def test_every_exported_name_resolves() -> None:
    """``__all__`` is the surface a caller imports; a rename has to be deliberate."""
    for name in reconstruct_batch_lamni.__all__:
        assert hasattr(reconstruct_batch_lamni, name), name
