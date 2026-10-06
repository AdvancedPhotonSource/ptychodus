"""Rank-guarding and cancellation of the shared reconstruction body.

``save_assembled_diffraction`` and ``run_reconstruction`` write four artifacts between them,
each with a truncating open. Under a multi-process launch every rank runs the same
reconstruction and ends up holding the same product, so without a guard every rank would
write the same paths concurrently. These tests pin down that the writes happen exactly once,
on the main process, while the reconstruction itself and the return value stay the same on
every rank -- and that a rank is recognized whichever launcher reported it.

The cancellation tests pin down the other half: a run told to stop still leaves the product
it had reached, a run told to stop before it reached anything says so instead of pretending,
and ranks never disagree about which epoch they stopped on. None of them raise a real signal
or build a real process group; both are stubbed at the one function that reports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import logging
import signal

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import BeamCenter, CropRegion
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import FilterValuesStep
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput
from ptychi.api import LSQMLOptions

# Deliberately imported at module scope: this file already does
# ``pytest.importorskip('ptychi')`` above.
from ptychodus.cli import _reconstruct_common  # noqa: E402
from ptychodus.cli._reconstruct_common import resolve_crop_region  # noqa: E402
from ptychodus.cli._reconstruct_common import (  # noqa: E402
    CancellationToken,
    install_signal_handlers,
    is_main_process,
    load_ptychi_options,
    run_reconstruction,
    save_assembled_diffraction,
)


@pytest.fixture(autouse=True)
def _neutral_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every rank source, so these tests do not inherit the launcher they run under.

    Without this the suite fails when pytest is itself started by `srun` or `torchrun`,
    which is an ordinary thing to do on the machines this code targets.
    """
    for name in ('RANK', 'SLURM_PROCID', 'LOCAL_WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)


PIXEL_M = 1.0e-9
OBJ_HEIGHT_PX = 32
OBJ_WIDTH_PX = 40
PROBE_HEIGHT_PX = 8
PROBE_WIDTH_PX = 8
NUM_PATTERNS = 3
NUM_EPOCHS = 4
NUM_SYNC_EPOCHS = 2


def _make_reconstruct_input() -> ReconstructInput:
    rng = numpy.random.default_rng(0)
    obj = Object(
        array=(
            rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
            + 1j * rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
        ).astype(numpy.complex128),
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[],
    )
    probes = ProbeSequence(
        array=(
            rng.standard_normal((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
            + 1j * rng.standard_normal((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
        ).astype(numpy.complex128),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
    )
    positions = ProbePositionSequence(
        [ProbePosition(index=i, x_m=i * PIXEL_M, y_m=-i * PIXEL_M) for i in range(NUM_PATTERNS)]
    )
    product = Product(
        metadata=ProductMetadata(
            name='test',
            comments='',
            detector_distance_m=2.5,
            probe_energy_eV=10_000.0,
            probe_photon_count=1.25e6,
            exposure_time_s=1.0,
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=positions,
        probes=probes,
        object_=obj,
        losses=[],
    )
    patterns = rng.random((NUM_PATTERNS, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)).astype(numpy.float32)
    bad_pixels = numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_)
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=bad_pixels,
        product=product,
    )


def _set_rank(
    monkeypatch: pytest.MonkeyPatch, rank: int | str, *, rank_variable: str | None
) -> None:
    """Make this process look like `rank`, reported the way `rank_variable` implies.

    With `rank_variable` the rank comes from the environment and the process group is
    absent, which is both the state during the first write of any run and the only state
    a plain `srun` job is ever in. Without it the group answers instead.
    """
    if rank_variable is None:
        # Resolved by a function-local import at call time, so patching the source works.
        monkeypatch.setattr('ptychi.parallel.get_rank', lambda: rank)
    else:
        # No process group, so pty-chi reports 0 for every process.
        monkeypatch.setattr('ptychi.parallel.get_rank', lambda: 0)
        monkeypatch.setenv(rank_variable, str(rank))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
    *,
    rank_variable: str | None = None,
) -> Product:
    """Run the tail with pty-chi stubbed out, as if on `rank`."""
    reconstruct_input = _make_reconstruct_input()
    sync_points: list[int] = []

    def fake_reconstruct(
        parameters: ReconstructInput,
        task_options: object,
        num_sync_epochs: int,
    ) -> Iterator[ReconstructOutput]:
        # Mirror the real generator's cadence so the checkpoint count is realistic.
        for epoch in range(num_sync_epochs, NUM_EPOCHS + 1, num_sync_epochs):
            sync_points.append(epoch)
            yield ReconstructOutput(product=parameters.product, progress=epoch)

    monkeypatch.setattr(_reconstruct_common, 'reconstruct_with_ptychi', fake_reconstruct)
    _set_rank(monkeypatch, rank, rank_variable=rank_variable)

    options = LSQMLOptions()
    options.reconstructor_options.num_epochs = NUM_EPOCHS
    options.object_options.pixel_size_m = PIXEL_M

    product = run_reconstruction(
        logging.getLogger('test'),
        reconstruct_input,
        options,
        tmp_path,
        num_sync_epochs=NUM_SYNC_EPOCHS,
    )

    assert sync_points, 'the reconstruction generator was never iterated'
    assert product is not None
    return product


def test_main_process_writes_every_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(monkeypatch, tmp_path, rank=0)

    assert (tmp_path / 'ptychi_options.json').is_file()
    assert (tmp_path / 'product.h5').is_file()
    # One checkpoint per sync point: epochs 2 and 4 at a stride of 2.
    assert sorted(p.name for p in tmp_path.glob('product.0*.h5')) == [
        'product.000002.h5',
        'product.000004.h5',
    ]


def test_secondary_process_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, rank=1)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('rank', [0, 1])
def test_product_is_returned_on_every_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank: int
) -> None:
    # The return value feeds callers that read refined positions back out, so it must be
    # the reconstructed product regardless of which rank is asking.
    product = _run(monkeypatch, tmp_path, rank=rank)

    assert isinstance(product, Product)
    assert len(product.probe_positions) == NUM_PATTERNS


def test_secondary_process_writes_nothing_before_the_group_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The first artifact is written before the reconstruction task is built, so there is
    # no process group yet and pty-chi reports rank 0 everywhere. Without consulting the
    # launcher's environment too, every rank would write the same paths at once.
    _run(monkeypatch, tmp_path, rank=1, rank_variable='RANK')

    assert list(tmp_path.iterdir()) == []


def test_main_process_writes_when_rank_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(monkeypatch, tmp_path, rank=0, rank_variable='RANK')

    assert (tmp_path / 'ptychi_options.json').is_file()
    assert (tmp_path / 'product.h5').is_file()


def test_srun_secondary_task_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Plain srun sets none of the variables pty-chi's launcher detection looks for, so it
    # builds no process group and every task reports rank 0. SLURM_PROCID is the only
    # thing that tells them apart.
    _run(monkeypatch, tmp_path, rank=1, rank_variable='SLURM_PROCID')

    assert list(tmp_path.iterdir()) == []


def test_srun_main_task_writes_every_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(monkeypatch, tmp_path, rank=0, rank_variable='SLURM_PROCID')

    assert (tmp_path / 'ptychi_options.json').is_file()
    assert (tmp_path / 'product.h5').is_file()


@pytest.mark.parametrize('rank_variable', ['RANK', 'SLURM_PROCID'])
@pytest.mark.parametrize('value', ['', 'oops'])
def test_an_unreadable_rank_variable_still_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank_variable: str, value: str
) -> None:
    # Something in the environment claims a rank it cannot state. Refusing to write the
    # run's output is the worse of the two readings, so it is ignored.
    monkeypatch.setattr('ptychi.parallel.get_rank', lambda: 0)
    monkeypatch.setenv(rank_variable, value)

    assert is_main_process()


def test_rank_is_taken_from_the_first_variable_that_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A job launched through srun under torchrun sets both. RANK is the one that tracks
    # the process group pty-chi will build, so it wins.
    monkeypatch.setattr('ptychi.parallel.get_rank', lambda: 0)
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('SLURM_PROCID', '1')

    assert is_main_process()


def _assembled_data() -> AssembledDiffractionData:
    rng = numpy.random.default_rng(0)
    return AssembledDiffractionData(
        indexes=numpy.arange(NUM_PATTERNS),
        patterns=rng.random((NUM_PATTERNS, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)).astype(numpy.float32),
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        bad_pixels=numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_),
    )


def test_save_assembled_diffraction_writes_once_on_the_main_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_rank(monkeypatch, 0, rank_variable=None)
    output_directory = tmp_path / 'out'

    save_assembled_diffraction(
        logging.getLogger('test'), output_directory, _assembled_data(), skip=False
    )

    assert (output_directory / 'diffraction.h5').is_file()


@pytest.mark.parametrize('rank_variable', [None, 'RANK', 'SLURM_PROCID'])
def test_save_assembled_diffraction_writes_nothing_on_a_secondary_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank_variable: str | None
) -> None:
    # The largest artifact of the run, and a truncating write: concurrent ranks writing
    # the same path corrupt it.
    _set_rank(monkeypatch, 1, rank_variable=rank_variable)
    output_directory = tmp_path / 'out'

    save_assembled_diffraction(
        logging.getLogger('test'), output_directory, _assembled_data(), skip=False
    )

    assert list(output_directory.iterdir()) == []


def test_save_assembled_diffraction_creates_the_directory_for_every_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The reconstruction outputs land here too, so the directory has to exist whether or
    # not this rank is the one that writes the patterns.
    _set_rank(monkeypatch, 1, rank_variable=None)
    output_directory = tmp_path / 'out'

    save_assembled_diffraction(
        logging.getLogger('test'), output_directory, _assembled_data(), skip=False
    )

    assert output_directory.is_dir()


def test_save_assembled_diffraction_skips_the_write_on_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_rank(monkeypatch, 0, rank_variable=None)
    output_directory = tmp_path / 'out'

    save_assembled_diffraction(
        logging.getLogger('test'), output_directory, _assembled_data(), skip=True
    )

    assert output_directory.is_dir()
    assert not (output_directory / 'diffraction.h5').exists()


@pytest.fixture(autouse=True)
def _no_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cancel consensus off the collective path unless a test asks for it.

    The suite may itself be run under a launcher that has built a process group, and a
    collective with no peers to answer it would hang the test run rather than fail it.
    """
    monkeypatch.setattr('torch.distributed.is_initialized', lambda: False)


def _run_with_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cancel_at_epoch: int | None,
    cancel_before_start: bool = False,
) -> tuple[Product | None, CancellationToken, list[tuple[int, Path | None]]]:
    """Run the tail, cancelling at a chosen point, and report what the sync hook saw."""
    reconstruct_input = _make_reconstruct_input()
    cancellation = CancellationToken()
    syncs: list[tuple[int, Path | None]] = []

    if cancel_before_start:
        cancellation.cancel('TEST')

    def fake_reconstruct(
        parameters: ReconstructInput,
        task_options: object,
        num_sync_epochs: int,
    ) -> Iterator[ReconstructOutput]:
        for epoch in range(num_sync_epochs, NUM_EPOCHS + 1, num_sync_epochs):
            if epoch == cancel_at_epoch:
                # As a signal arriving mid-chunk would: the flag is already up by the
                # time this epoch's output reaches the loop.
                cancellation.cancel('SIGTERM')

            yield ReconstructOutput(product=parameters.product, progress=epoch)

    monkeypatch.setattr(_reconstruct_common, 'reconstruct_with_ptychi', fake_reconstruct)
    _set_rank(monkeypatch, 0, rank_variable=None)

    options = LSQMLOptions()
    options.reconstructor_options.num_epochs = NUM_EPOCHS
    options.object_options.pixel_size_m = PIXEL_M

    product = run_reconstruction(
        logging.getLogger('test'),
        reconstruct_input,
        options,
        tmp_path,
        num_sync_epochs=NUM_SYNC_EPOCHS,
        cancellation=cancellation,
        on_sync=lambda output, checkpoint: syncs.append((output.progress, checkpoint)),
    )
    return product, cancellation, syncs


def test_a_cancelled_run_still_writes_the_product_it_reached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Stopping politely exists to keep the partially converged result. Killing the process
    # would already have thrown it away, so a cancel that wrote nothing would be pointless.
    product, cancellation, syncs = _run_with_cancellation(
        monkeypatch, tmp_path, cancel_at_epoch=NUM_SYNC_EPOCHS
    )

    assert product is not None
    assert cancellation.is_cancelled
    assert (tmp_path / 'product.h5').is_file()
    # Stopped at the first sync point, so the later epochs' checkpoints never happened.
    assert sorted(p.name for p in tmp_path.glob('product.0*.h5')) == ['product.000002.h5']
    assert [epoch for epoch, _ in syncs] == [NUM_SYNC_EPOCHS]


def test_a_run_cancelled_before_the_first_epoch_returns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No epoch completed, so there is no product to save and none to return. Writing the
    # unreconstructed input product here would look like a result.
    product, cancellation, syncs = _run_with_cancellation(
        monkeypatch, tmp_path, cancel_at_epoch=None, cancel_before_start=True
    )

    assert product is None
    assert cancellation.is_cancelled
    assert syncs == []
    assert not (tmp_path / 'product.h5').exists()


def test_an_uncancelled_run_that_produces_nothing_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The cancelled empty run is expected; the uncancelled one is a reconstructor fault and
    # must stay loud.
    def fake_reconstruct(
        parameters: ReconstructInput,
        task_options: object,
        num_sync_epochs: int,
    ) -> Iterator[ReconstructOutput]:
        return iter(())

    monkeypatch.setattr(_reconstruct_common, 'reconstruct_with_ptychi', fake_reconstruct)
    _set_rank(monkeypatch, 0, rank_variable=None)

    options = LSQMLOptions()
    options.object_options.pixel_size_m = PIXEL_M

    with pytest.raises(RuntimeError):
        run_reconstruction(
            logging.getLogger('test'),
            _make_reconstruct_input(),
            options,
            tmp_path,
            num_sync_epochs=NUM_SYNC_EPOCHS,
        )


def test_the_sync_hook_reports_every_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, syncs = _run_with_cancellation(monkeypatch, tmp_path, cancel_at_epoch=None)

    assert [epoch for epoch, _ in syncs] == [2, 4]
    assert [checkpoint.name for _, checkpoint in syncs if checkpoint is not None] == [
        'product.000002.h5',
        'product.000004.h5',
    ]


def test_the_sync_hook_reports_no_checkpoint_on_a_secondary_rank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A rank that writes nothing must say so rather than name a path it did not create.
    reconstruct_input = _make_reconstruct_input()
    syncs: list[tuple[int, Path | None]] = []

    def fake_reconstruct(
        parameters: ReconstructInput,
        task_options: object,
        num_sync_epochs: int,
    ) -> Iterator[ReconstructOutput]:
        yield ReconstructOutput(product=parameters.product, progress=num_sync_epochs)

    monkeypatch.setattr(_reconstruct_common, 'reconstruct_with_ptychi', fake_reconstruct)
    _set_rank(monkeypatch, 1, rank_variable=None)

    options = LSQMLOptions()
    options.object_options.pixel_size_m = PIXEL_M

    run_reconstruction(
        logging.getLogger('test'),
        reconstruct_input,
        options,
        tmp_path,
        num_sync_epochs=NUM_SYNC_EPOCHS,
        on_sync=lambda output, checkpoint: syncs.append((output.progress, checkpoint)),
    )

    assert syncs == [(NUM_SYNC_EPOCHS, None)]


def test_a_run_with_no_token_never_touches_the_collective(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Ten of the eleven drivers pass a token, but the parameter is optional and a caller
    # that omits it must not pay for a reduction it cannot use.
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('all_reduce was called without a cancellation token')

    monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
    monkeypatch.setattr('torch.distributed.all_reduce', fail)

    product = _run(monkeypatch, tmp_path, rank=0)

    assert isinstance(product, Product)


def test_a_peers_cancellation_stops_this_rank_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The hazard the reduction exists for: this rank was never signalled, but a peer was.
    # Carrying on alone would leave it waiting at a collective its peers have left.
    import torch

    monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
    monkeypatch.setattr('torch.distributed.get_backend', lambda: 'gloo')

    # The peer is signalled after the run is under way, so the pre-loop reduction still
    # agrees on "carry on" and only the one at the first sync point reports the cancel.
    reductions = 0

    def peer_cancels_after_the_run_starts(tensor: torch.Tensor, op: object = None) -> None:
        nonlocal reductions
        reductions += 1

        if reductions > 1:
            tensor.fill_(1)

    monkeypatch.setattr('torch.distributed.all_reduce', peer_cancels_after_the_run_starts)

    product, cancellation, syncs = _run_with_cancellation(
        monkeypatch, tmp_path, cancel_at_epoch=None
    )

    # This rank's own flag stayed down, yet it stopped at the first sync point anyway.
    assert not cancellation.is_cancelled
    assert [epoch for epoch, _ in syncs] == [NUM_SYNC_EPOCHS]
    assert product is not None


def test_install_signal_handlers_claims_both_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: dict[int, object] = {}
    monkeypatch.setattr(signal, 'signal', lambda num, handler: installed.setdefault(num, handler))

    token = install_signal_handlers()

    assert set(installed) == {signal.SIGINT, signal.SIGTERM}
    assert not token.is_cancelled


def test_the_first_signal_requests_and_the_second_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run wedged somewhere pty-chi cannot interrupt must still answer a second Ctrl-C,
    # so the handler stands aside rather than swallowing it.
    installed: dict[int, object] = {}
    killed: list[int] = []
    monkeypatch.setattr(signal, 'signal', lambda num, handler: installed.setdefault(num, handler))
    monkeypatch.setattr('os.kill', lambda pid, num: killed.append(num))

    token = install_signal_handlers()
    handler = installed[signal.SIGTERM]
    assert callable(handler)

    handler(int(signal.SIGTERM), None)
    assert token.is_cancelled
    assert token.reason == 'SIGTERM'
    assert killed == []

    handler(int(signal.SIGTERM), None)
    assert killed == [int(signal.SIGTERM)]
    assert installed[signal.SIGTERM] is handler  # setdefault kept the first registration


def test_the_cancel_callback_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # A launcher emits its "cancelling" event from here, and two of them would be a lie
    # about how many requests arrived.
    reasons: list[str] = []
    token = CancellationToken(on_cancel=reasons.append)

    token.cancel('SIGTERM')
    token.cancel('SIGINT')

    assert reasons == ['SIGTERM']
    assert token.reason == 'SIGTERM'


def test_load_ptychi_options_builds_a_fresh_object_each_time() -> None:
    # The per-scan loop edits the options it is handed, and a shared object would carry
    # one projection's edits into the next.
    first = load_ptychi_options(None)
    first.reconstructor_options.num_epochs = 11

    assert load_ptychi_options(None).reconstructor_options.num_epochs != 11


def test_load_ptychi_options_round_trips_a_file(tmp_path: Path) -> None:
    options = LSQMLOptions()
    options.reconstructor_options.num_epochs = 7
    options_file = tmp_path / 'ptychi_options.json'
    options_file.write_text(_reconstruct_common.dump_task_options(options))

    assert load_ptychi_options(options_file).reconstructor_options.num_epochs == 7


def test_load_ptychi_options_reads_stdin_for_a_dash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import io

    options = LSQMLOptions()
    options.reconstructor_options.num_epochs = 5
    monkeypatch.setattr('sys.stdin', io.StringIO(_reconstruct_common.dump_task_options(options)))

    loaded = load_ptychi_options(_reconstruct_common._STDIN_ARGUMENT)

    assert loaded.reconstructor_options.num_epochs == 5


# --- resolve_crop_region ---------------------------------------------------------------
#
# Shared between the standard pipeline and the batch-LamNI loop, which previously carried
# byte-identical copies. The two differ only in what they are summarizing, which `subject`
# supplies, so that wording is pinned alongside the arithmetic.


class _FakeSummary:
    """Just the mean pattern, which is all the estimate branch reads."""

    def __init__(self) -> None:
        self.mean_pattern = numpy.array([[1, 9], [3, 4]], dtype=numpy.uint16)


class _FakeMetadata:
    def __init__(self, extent: ImageExtent, beam_center: BeamCenter | None) -> None:
        self.detector_extent = extent
        self.beam_center = beam_center


class _FakeDataset:
    """Only what resolve_crop_region touches: metadata, and the patterns it may summarize."""

    def __init__(self, extent: ImageExtent, beam_center: BeamCenter | None = None) -> None:
        self._metadata = _FakeMetadata(extent, beam_center)

    def get_metadata(self) -> _FakeMetadata:
        return self._metadata


def _resolve(dataset: _FakeDataset, **kwargs: object) -> CropRegion | None:
    defaults: dict[str, object] = {
        'crop_extent_px': None,
        'beam_center_x_px': None,
        'beam_center_y_px': None,
        'bad_pixels': None,
        'value_filter': None,
    }
    defaults.update(kwargs)
    return resolve_crop_region(logging.getLogger('test'), dataset, **defaults)  # type: ignore[arg-type]


def test_no_crop_requested_reads_the_whole_frame() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512))
    assert _resolve(dataset) is None


def test_a_crop_matching_the_detector_reads_the_whole_frame() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=256, height_px=256), BeamCenter(x_px=128, y_px=128))
    assert _resolve(dataset, crop_extent_px=256) is None


def test_the_command_line_beam_center_wins_over_the_file() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=100, y_px=100))
    region = _resolve(dataset, crop_extent_px=64, beam_center_x_px=256, beam_center_y_px=256)

    assert region is not None
    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=256, y_px=256), ImageExtent(width_px=64, height_px=64)
    )


def test_the_file_beam_center_is_used_when_no_flag_is_given() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=200, y_px=300))
    region = _resolve(dataset, crop_extent_px=32)

    assert region is not None
    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=200, y_px=300), ImageExtent(width_px=32, height_px=32)
    )


def test_an_overhanging_crop_is_an_error_rather_than_a_silent_clamp() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=4, y_px=4))

    with pytest.raises(ValueError, match='runs off the'):
        _resolve(dataset, crop_extent_px=256)


def test_subject_names_what_the_beam_center_was_estimated_over(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The only wording the two callers differ on, so it is the only thing parameterized.

    Reaching the estimate branch at all requires no center from either the flag or the
    file; the summarize/estimate pair is stubbed because what is under test is the
    reporting, not the estimator.
    """
    monkeypatch.setattr(
        _reconstruct_common, 'summarize_dataset', lambda dataset, bad_pixels=None: _FakeSummary()
    )
    monkeypatch.setattr(
        _reconstruct_common, 'estimate_beam_center', lambda pattern: BeamCenter(x_px=256, y_px=256)
    )
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), beam_center=None)

    with caplog.at_level(logging.INFO):
        _resolve(dataset, crop_extent_px=64, subject='first scan')

    assert 'Summarizing the first scan to estimate the beam center' in caplog.text
    assert 'an estimate over the whole first scan' in caplog.text


def test_subject_defaults_to_the_whole_dataset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        _reconstruct_common, 'summarize_dataset', lambda dataset, bad_pixels=None: _FakeSummary()
    )
    monkeypatch.setattr(
        _reconstruct_common, 'estimate_beam_center', lambda pattern: BeamCenter(x_px=256, y_px=256)
    )
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), beam_center=None)

    with caplog.at_level(logging.INFO):
        _resolve(dataset, crop_extent_px=64)

    assert 'Summarizing the dataset to estimate the beam center' in caplog.text
    assert 'an estimate over the whole dataset' in caplog.text


def test_the_value_filter_is_applied_to_the_mean_pattern_before_estimating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cut used for the estimate must match the cut used for the assembled patterns."""
    seen: list[numpy.ndarray] = []

    monkeypatch.setattr(
        _reconstruct_common, 'summarize_dataset', lambda dataset, bad_pixels=None: _FakeSummary()
    )

    def record_and_center(pattern: numpy.ndarray) -> BeamCenter:
        seen.append(pattern)
        return BeamCenter(x_px=256, y_px=256)

    monkeypatch.setattr(_reconstruct_common, 'estimate_beam_center', record_and_center)
    value_filter = FilterValuesStep(lower_bound=0, upper_bound=5)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), beam_center=None)

    _resolve(dataset, crop_extent_px=64, value_filter=value_filter)

    assert len(seen) == 1
    # The fake mean pattern holds a 9, which the filter's upper bound of 5 zeroes.
    assert seen[0].max() <= 5
