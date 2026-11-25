"""Rank-guarding and cancellation of the shared reconstruction body.

``save_assembled_diffraction`` and ``run_reconstruction`` write four files between them,
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

from dataclasses import replace
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, Iterator

import logging
import signal

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.accumulate import AssemblyMomentAccumulator
from ptychodus.api.assemble import (
    AssembledDiffractionData,
    CropSuggestion,
    DetectorCorrections,
    assemble_dataset,
)
from ptychodus.api.diffraction import (
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.diffraction import BeamCenter, CropRegion
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import FilterValuesStep, SignalRadius
from ptychodus.api.preprocess.noise import RobustStatistics
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.defocus import AutofocusEstimate, ParallaxEstimate
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.reconstruct import (
    PositionIndexFilter,
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
)
from pydantic import BaseModel

# Deliberately imported at module scope: this file already does
# ``pytest.importorskip('ptychi')`` above.
from ptychodus.cli import CancellationToken, install_signal_handlers  # noqa: E402
from ptychodus.cli import _reconstruct_common  # noqa: E402
from ptychodus.cli._reconstruct_common import resolve_crop_region  # noqa: E402
from ptychodus.cli._reconstruct_common import (  # noqa: E402
    default_reconstruction,
    estimate_focus_object_distance_m,
    estimate_incident_beam_from_assembled_data,
    find_options_document,
    invalid_count_threshold,
    is_main_process,
    pair_patterns_with_positions,
    read_reconstruction,
    require_near_field_focus_object_distance_m,
    resolve_flag,
    resolve_stated_focus_object_distance_m,
    ResolvedReconstruction,
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


class _FakeOptions(BaseModel):
    num_epochs: int = NUM_EPOCHS
    num_sync_epochs: int = NUM_SYNC_EPOCHS


class _FakeReconstructor(Reconstructor):
    """Yields every `num_sync_epochs` epochs, calling `on_epoch` first, as a reconstructor would.

    With `progress_values` it yields those progress values instead, in order.
    """

    def __init__(
        self,
        *,
        on_epoch: Callable[[int], None] | None = None,
        empty: bool = False,
        progress_values: Sequence[int] | None = None,
    ) -> None:
        self._on_epoch = on_epoch
        self._empty = empty
        self._progress_values = progress_values
        self.sync_points: list[int] = []

    @property
    def name(self) -> str:
        return 'Fake'

    def get_default_options(self) -> BaseModel:
        return _FakeOptions()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        assert isinstance(options, _FakeOptions)

        if self._empty:
            return

        progress_values = self._progress_values or range(
            options.num_sync_epochs, options.num_epochs + 1, options.num_sync_epochs
        )

        for epoch in progress_values:
            if self._on_epoch is not None:
                self._on_epoch(epoch)

            self.sync_points.append(epoch)
            yield ReconstructOutput(
                product=input_.product, progress=epoch, progress_goal=options.num_epochs
            )


def _resolved(reconstructor: _FakeReconstructor) -> ResolvedReconstruction:
    return ResolvedReconstruction('Fake', reconstructor, _FakeOptions())


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
    probes = ProbeEnsemble(
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
            photon_energy_eV=10_000.0,
            probe_photon_count=1.25e6,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
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
        monkeypatch.setattr(_reconstruct_common, '_group_rank', lambda: rank)
    else:
        # No process group, so the group reports 0 for every process.
        monkeypatch.setattr(_reconstruct_common, '_group_rank', lambda: 0)
        monkeypatch.setenv(rank_variable, str(rank))


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rank: int,
    *,
    rank_variable: str | None = None,
) -> Product:
    """Run the tail with a stub reconstructor, as if on `rank`."""
    reconstructor = _FakeReconstructor()
    _set_rank(monkeypatch, rank, rank_variable=rank_variable)

    product = run_reconstruction(
        logging.getLogger('test'), _resolved(reconstructor), _make_reconstruct_input(), tmp_path
    )

    assert reconstructor.sync_points, 'the reconstruction generator was never iterated'
    assert product is not None
    return product


def test_main_process_writes_every_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, rank=0)

    assert (tmp_path / 'reconstruct_options.yaml').is_file()
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
    # The first file is written before the reconstruction task is built, so there is
    # no process group yet and pty-chi reports rank 0 everywhere. Without consulting the
    # launcher's environment too, every rank would write the same paths at once.
    _run(monkeypatch, tmp_path, rank=1, rank_variable='RANK')

    assert list(tmp_path.iterdir()) == []


def test_main_process_writes_when_rank_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(monkeypatch, tmp_path, rank=0, rank_variable='RANK')

    assert (tmp_path / 'reconstruct_options.yaml').is_file()
    assert (tmp_path / 'product.h5').is_file()


def test_srun_secondary_task_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Plain srun sets none of the variables pty-chi's launcher detection looks for, so it
    # builds no process group and every task reports rank 0. SLURM_PROCID is the only
    # thing that tells them apart.
    _run(monkeypatch, tmp_path, rank=1, rank_variable='SLURM_PROCID')

    assert list(tmp_path.iterdir()) == []


def test_srun_main_task_writes_every_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _run(monkeypatch, tmp_path, rank=0, rank_variable='SLURM_PROCID')

    assert (tmp_path / 'reconstruct_options.yaml').is_file()
    assert (tmp_path / 'product.h5').is_file()


@pytest.mark.parametrize('rank_variable', ['RANK', 'SLURM_PROCID'])
@pytest.mark.parametrize('value', ['', 'oops'])
def test_an_unreadable_rank_variable_still_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rank_variable: str, value: str
) -> None:
    # Something in the environment claims a rank it cannot state. Refusing to write the
    # run's output is the worse of the two readings, so it is ignored.
    monkeypatch.setattr(_reconstruct_common, '_group_rank', lambda: 0)
    monkeypatch.setenv(rank_variable, value)

    assert is_main_process()


def test_rank_is_taken_from_the_first_variable_that_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A job launched through srun under torchrun sets both. RANK is the one that tracks
    # the process group pty-chi will build, so it wins.
    monkeypatch.setattr(_reconstruct_common, '_group_rank', lambda: 0)
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
    # The largest file of the run, and a truncating write: concurrent ranks writing
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

    def cancel_on(epoch: int) -> None:
        if epoch == cancel_at_epoch:
            # As a signal arriving mid-chunk would: the flag is already up by the time this
            # epoch's output reaches the loop.
            cancellation.cancel('SIGTERM')

    _set_rank(monkeypatch, 0, rank_variable=None)
    product = run_reconstruction(
        logging.getLogger('test'),
        _resolved(_FakeReconstructor(on_epoch=cancel_on)),
        reconstruct_input,
        tmp_path,
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
    _set_rank(monkeypatch, 0, rank_variable=None)

    with pytest.raises(RuntimeError):
        run_reconstruction(
            logging.getLogger('test'),
            _resolved(_FakeReconstructor(empty=True)),
            _make_reconstruct_input(),
            tmp_path,
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
    syncs: list[tuple[int, Path | None]] = []
    _set_rank(monkeypatch, 1, rank_variable=None)

    run_reconstruction(
        logging.getLogger('test'),
        _resolved(_FakeReconstructor()),
        _make_reconstruct_input(),
        tmp_path,
        on_sync=lambda output, checkpoint: syncs.append((output.progress, checkpoint)),
    )

    assert [checkpoint for _, checkpoint in syncs] == [None, None]


def _run_with_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    progress_values: Sequence[int],
    *,
    published_progress: int | None = None,
) -> list[tuple[int, Path | None]]:
    """Run the tail on the main process yielding `progress_values`; report the sync hook's view."""
    syncs: list[tuple[int, Path | None]] = []
    _set_rank(monkeypatch, 0, rank_variable=None)
    run_reconstruction(
        logging.getLogger('test'),
        _resolved(_FakeReconstructor(progress_values=progress_values)),
        _make_reconstruct_input(),
        tmp_path,
        on_sync=lambda output, checkpoint: syncs.append((output.progress, checkpoint)),
        published_progress=published_progress,
    )
    return syncs


def test_a_published_run_resource_is_never_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A run resource publishes its options document and a copy of the input product as
    # checkpoint 0 before the run starts, and readers may already hold either. Rewriting
    # one with a truncating open would pull it out from under them.
    published = {'product.000000.h5': b'input product', 'reconstruct_options.yaml': b'options'}

    for name, content in published.items():
        (tmp_path / name).write_bytes(content)

    syncs = _run_with_progress(monkeypatch, tmp_path, [0, 2, 2, 4], published_progress=0)

    for name, content in published.items():
        assert (tmp_path / name).read_bytes() == content

    assert sorted(p.name for p in tmp_path.glob('product.*.h5')) == [
        'product.000000.h5',
        'product.000002.h5',
        'product.000004.h5',
    ]
    assert (tmp_path / 'product.h5').is_file()
    assert [(progress, None if c is None else c.name) for progress, c in syncs] == [
        (0, None),
        (2, 'product.000002.h5'),
        (2, None),
        (4, 'product.000004.h5'),
    ]


def test_an_output_that_does_not_advance_keeps_its_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The fake yields the same product every time, so the bytes on disk cannot show a
    # rewrite; count the saves instead.
    saved: list[str] = []
    save_product = _reconstruct_common.save_product

    def spy(file_path: Path, product: Product) -> None:
        saved.append(file_path.name)
        save_product(file_path, product)

    monkeypatch.setattr(_reconstruct_common, 'save_product', spy)
    syncs = _run_with_progress(monkeypatch, tmp_path, [2, 2, 4])

    assert saved == ['product.000002.h5', 'product.000004.h5', 'product.h5']
    assert [checkpoint is None for _, checkpoint in syncs] == [False, True, False]
    assert (tmp_path / 'reconstruct_options.yaml').is_file()


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


# --- options documents --------------------------------------------------------------------
#
# A prepared directory is read by both ptychodus-reconstruct and the subprocess driver, so
# how its options document is chosen, read and written is pinned here once.


def test_an_explicit_options_file_wins(tmp_path: Path) -> None:
    (tmp_path / 'reconstruct_options.yaml').write_text('{}')
    chosen = tmp_path / 'elsewhere.yaml'

    assert find_options_document(chosen, tmp_path) == chosen


def test_the_stdin_sentinel_is_passed_through(tmp_path: Path) -> None:
    # ``-`` is not a path to probe for; resolving it against the input directory would
    # turn an in-memory handoff into a missing file.
    (tmp_path / 'reconstruct_options.yaml').write_text('{}')

    assert find_options_document(Path('-'), tmp_path) == Path('-')


def test_the_staged_options_are_used_when_present(tmp_path: Path) -> None:
    staged = tmp_path / 'reconstruct_options.yaml'
    staged.write_text('{}')

    assert find_options_document(None, tmp_path) == staged


def test_a_missing_staged_options_file_falls_back_rather_than_failing(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    # A directory staged by hand carries no options, and an error would make that ordinary
    # invocation the failing one.
    with caplog.at_level('WARNING'):
        assert find_options_document(None, tmp_path) is None

    assert any('stock options' in record.getMessage() for record in caplog.records)


def test_the_stock_fallback_follows_the_propagation_regime() -> None:
    libraries = PluginRegistry.load_plugins().reconstructor_libraries

    far = default_reconstruction(libraries, far_field=True)
    near = default_reconstruction(libraries, far_field=False)

    assert far.selectors() == {'library': 'PtyChi', 'algorithm': 'LSQML'}
    assert near.selectors() == {'library': 'PtyChi', 'algorithm': 'BH'}


def test_the_default_is_a_fresh_object_each_time() -> None:
    # The per-scan loop edits the options it is handed, and a shared object would carry
    # one projection's edits into the next.
    libraries = PluginRegistry.load_plugins().reconstructor_libraries
    first = default_reconstruction(libraries, far_field=True).options
    first.task.reconstructor_options.num_epochs = 11  # type: ignore[attr-defined]

    second = default_reconstruction(libraries, far_field=True).options
    assert second.task.reconstructor_options.num_epochs != 11  # type: ignore[attr-defined]


def test_a_written_document_reads_back_as_the_same_run(tmp_path: Path) -> None:
    libraries = PluginRegistry.load_plugins().reconstructor_libraries
    resolved = default_reconstruction(libraries, far_field=False)
    resolved.options.task.reconstructor_options.num_epochs = 9  # type: ignore[attr-defined]
    options_file = tmp_path / 'reconstruct_options.yaml'
    options_file.write_text(resolved.dump())

    loaded = read_reconstruction(libraries, options_file)

    assert loaded.selectors() == resolved.selectors()
    assert type(loaded.options) is type(resolved.options)
    assert loaded.options.task.reconstructor_options.num_epochs == 9  # type: ignore[attr-defined]


def test_a_document_naming_an_unknown_library_is_refused(tmp_path: Path) -> None:
    options_file = tmp_path / 'options.yaml'
    options_file.write_text('library: NoSuchLibrary\nalgorithm: X\n')

    with pytest.raises(ValueError, match='NoSuchLibrary'):
        read_reconstruction(PluginRegistry.load_plugins().reconstructor_libraries, options_file)


def test_the_options_are_written_once_on_the_main_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_rank(monkeypatch, 1, rank_variable=None)
    _reconstruct_common.save_reconstruct_options(
        logging.getLogger('test'), _resolved(_FakeReconstructor()), tmp_path
    )
    assert not (tmp_path / 'reconstruct_options.yaml').exists()

    _set_rank(monkeypatch, 0, rank_variable=None)
    _reconstruct_common.save_reconstruct_options(
        logging.getLogger('test'), _resolved(_FakeReconstructor()), tmp_path
    )
    assert (tmp_path / 'reconstruct_options.yaml').read_text().startswith('library: Fake\n')


def _product_with_positions(num_positions: int) -> Product:
    product = _make_reconstruct_input().product
    return Product(
        metadata=product.metadata,
        probe_positions=ProbePositionSequence(list(product.probe_positions)[:num_positions]),
        probes=product.probes,
        object_=product.object_,
        losses=[],
    )


def test_pairing_every_pattern_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level('WARNING', logger='test'):
        reconstruct_input = pair_patterns_with_positions(
            logging.getLogger('test'), _assembled_data(), _product_with_positions(NUM_PATTERNS)
        )

    assert reconstruct_input.diffraction_patterns.shape[0] == NUM_PATTERNS
    assert not caplog.records


def test_patterns_past_the_last_position_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    # The last pattern index lies outside the position range, so it is dropped rather
    # than extrapolated -- a stopped scan, which is worth a warning but not an error.
    with caplog.at_level('WARNING', logger='test'):
        reconstruct_input = pair_patterns_with_positions(
            logging.getLogger('test'), _assembled_data(), _product_with_positions(NUM_PATTERNS - 1)
        )

    assert reconstruct_input.diffraction_patterns.shape[0] == NUM_PATTERNS - 1
    assert any('Paired 2 of 3' in record.getMessage() for record in caplog.records)


def test_patterns_an_index_filter_excludes_are_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level('WARNING', logger='test'):
        reconstruct_input = pair_patterns_with_positions(
            logging.getLogger('test'),
            _assembled_data(),
            _product_with_positions(NUM_PATTERNS),
            index_filter=PositionIndexFilter.EVEN,
        )

    assert reconstruct_input.diffraction_patterns.shape[0] < NUM_PATTERNS
    assert not caplog.records


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

    def get_bad_pixels(self) -> numpy.ndarray:
        return numpy.zeros(self._metadata.detector_extent.get_shape(), dtype=bool)


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


def test_a_full_extent_reads_the_whole_frame_without_summarizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarize = _CountingSummarize()
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', summarize)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512))

    assert _resolve(dataset, crop_extent_px='full') is None
    assert summarize.num_calls == 0


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


def _signal_radius(radius_px: int) -> SignalRadius:
    return SignalRadius(
        radius_px=radius_px,
        radial_profile=numpy.zeros(1),
        noise_floor=RobustStatistics(median=0.0, median_absolute_deviation=1.0),
        threshold=4.5,
    )


class _FakeSuggest:
    """Stands in for suggest_crop_region with a fixed radius, recording what it was handed."""

    def __init__(self, radius_px: int) -> None:
        self._radius_px = radius_px
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        mean_pattern: numpy.ndarray,
        beam_center: BeamCenter,
        detector_extent: ImageExtent,
        **kwargs: Any,
    ) -> CropSuggestion:
        self.calls.append({'mean_pattern': mean_pattern, **kwargs})

        if self._radius_px < 1:
            raise ValueError('No signal rises above the noise floor about the beam center.')

        return CropSuggestion(
            signal=_signal_radius(self._radius_px),
            region=CropRegion.from_smallest_pow2(beam_center, self._radius_px, detector_extent),
        )


class _CountingSummarize:
    def __init__(self) -> None:
        self.num_calls = 0

    def __call__(self, dataset: object, bad_pixels: object = None) -> _FakeSummary:
        self.num_calls += 1
        return _FakeSummary()


def test_an_auto_extent_is_the_power_of_two_holding_the_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', _FakeSuggest(20))
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=200, y_px=300))

    region = _resolve(dataset, crop_extent_px='auto')

    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=200, y_px=300), ImageExtent(width_px=64, height_px=64)
    )


def test_an_auto_extent_shares_one_summary_with_the_beam_center_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarize = _CountingSummarize()
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', summarize)
    monkeypatch.setattr(
        _reconstruct_common, 'estimate_beam_center', lambda pattern: BeamCenter(x_px=256, y_px=256)
    )
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', _FakeSuggest(40))
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), beam_center=None)

    region = _resolve(dataset, crop_extent_px='auto')

    assert summarize.num_calls == 1
    assert region is not None
    assert region.width_px == 128


def test_an_auto_extent_hands_the_filter_and_corrections_over_unapplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """suggest_crop_region applies both, so the extent path must not cut the mean itself."""
    suggest = _FakeSuggest(20)
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', suggest)
    value_filter = FilterValuesStep(lower_bound=0, upper_bound=5)
    corrections = DetectorCorrections(subtract_background=True)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    _resolve(dataset, crop_extent_px='auto', value_filter=value_filter, corrections=corrections)

    (call,) = suggest.calls
    assert call['value_filter'] is value_filter
    assert call['corrections'] is corrections
    # The fake mean pattern holds a 9, which the filter would have zeroed.
    assert call['mean_pattern'].max() == 9


def test_an_auto_extent_filling_the_detector_reads_the_whole_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', _FakeSuggest(1000))
    dataset = _FakeDataset(ImageExtent(width_px=256, height_px=256), BeamCenter(x_px=128, y_px=128))

    assert _resolve(dataset, crop_extent_px='auto') is None


def test_an_auto_extent_without_signal_is_an_error_naming_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', _FakeSuggest(0))
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    with pytest.raises(ValueError, match='noise floor.*--crop-extent-px.*"full"'):
        _resolve(dataset, crop_extent_px='auto')


def test_an_omitted_extent_without_signal_reads_the_whole_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nobody asked for a crop, so failing to size one must not fail the run."""
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', _FakeSuggest(0))
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    with caplog.at_level(logging.WARNING):
        assert _resolve(dataset) is None

    assert 'Reading the whole detector frame: No signal' in caplog.text


def test_an_omitted_extent_is_estimated(monkeypatch: pytest.MonkeyPatch) -> None:
    suggest = _FakeSuggest(20)
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', suggest)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=200, y_px=300))

    region = _resolve(dataset)

    assert len(suggest.calls) == 1
    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=200, y_px=300), ImageExtent(width_px=64, height_px=64)
    )


def test_an_omitted_extent_takes_the_warm_start_extent_without_summarizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarize = _CountingSummarize()
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', summarize)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=200, y_px=300))

    region = _resolve(dataset, warm_start_extent=ImageExtent(width_px=96, height_px=96))

    assert summarize.num_calls == 0
    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=200, y_px=300), ImageExtent(width_px=96, height_px=96)
    )


def test_a_warm_start_extent_need_not_be_square() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    region = _resolve(dataset, warm_start_extent=ImageExtent(width_px=128, height_px=64))

    assert region is not None
    assert (region.width_px, region.height_px) == (128, 64)


def test_a_warm_start_without_a_recorded_center_summarizes_once_for_the_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summarize = _CountingSummarize()
    suggest = _FakeSuggest(20)
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', summarize)
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', suggest)
    monkeypatch.setattr(
        _reconstruct_common, 'estimate_beam_center', lambda pattern: BeamCenter(x_px=256, y_px=256)
    )
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), beam_center=None)

    region = _resolve(dataset, warm_start_extent=ImageExtent(width_px=96, height_px=96))

    assert summarize.num_calls == 1
    assert suggest.calls == []
    assert region == CropRegion.from_center_extent(
        BeamCenter(x_px=256, y_px=256), ImageExtent(width_px=96, height_px=96)
    )


def test_an_explicit_auto_extent_overrides_the_warm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    suggest = _FakeSuggest(20)
    monkeypatch.setattr(_reconstruct_common, 'summarize_dataset', _CountingSummarize())
    monkeypatch.setattr(_reconstruct_common, 'suggest_crop_region', suggest)
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    region = _resolve(
        dataset, crop_extent_px='auto', warm_start_extent=ImageExtent(width_px=96, height_px=96)
    )

    assert len(suggest.calls) == 1
    assert region is not None
    assert region.width_px == 64


def test_an_explicit_extent_overrides_the_warm_start() -> None:
    dataset = _FakeDataset(ImageExtent(width_px=512, height_px=512), BeamCenter(x_px=256, y_px=256))

    region = _resolve(
        dataset, crop_extent_px=32, warm_start_extent=ImageExtent(width_px=96, height_px=96)
    )

    assert region is not None
    assert region.width_px == 32


class TestResolveFlag:
    """Provenance matters as much as the value: a guessed regime must look guessed."""

    def test_the_command_line_wins(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            assert resolve_flag(logging.getLogger(), 'Far field', False, True, True, '--x') is False

        assert '--x' in caplog.text

    def test_the_file_wins_over_the_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            assert resolve_flag(logging.getLogger(), 'Far field', None, False, True, '--x') is False

        assert 'diffraction file' in caplog.text

    def test_the_fallback_says_it_guessed(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            assert resolve_flag(logging.getLogger(), 'Far field', None, None, True, '--x') is True

        assert 'built-in default' in caplog.text


class TestResolveStatedFocusObjectDistance:
    """The focus is stated explicitly or recorded by the file; nothing presets it."""

    def test_an_explicit_value_beats_the_file(self) -> None:
        resolved = resolve_stated_focus_object_distance_m(logging.getLogger(), -9.0e-4, -21.873e-3)

        assert resolved == pytest.approx(-9.0e-4)

    def test_the_file_states_it_otherwise(self) -> None:
        resolved = resolve_stated_focus_object_distance_m(logging.getLogger(), None, -21.873e-3)

        assert resolved == pytest.approx(-21.873e-3)

    def test_an_explicit_zero_is_a_statement(self) -> None:
        assert resolve_stated_focus_object_distance_m(logging.getLogger(), 0.0, -21.873e-3) == 0.0

    def test_with_neither_nothing_is_stated(self) -> None:
        assert resolve_stated_focus_object_distance_m(logging.getLogger(), None, None) is None

    def test_the_log_names_the_source(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO):
            resolve_stated_focus_object_distance_m(
                logging.getLogger(), -9.0e-4, None, source='metadata_overrides.focus'
            )

        assert 'metadata_overrides.focus' in caplog.text


class TestRequireNearFieldFocusObjectDistance:
    """Near field has no estimate, so its focus must be stated."""

    @pytest.mark.parametrize('stated', [-9.0e-4, 0.0])
    def test_a_stated_focus_passes(self, stated: float) -> None:
        assert require_near_field_focus_object_distance_m(stated, source='--x') == stated

    def test_an_unstated_focus_is_refused_naming_where_to_state_it(self) -> None:
        with pytest.raises(ValueError, match='--focus-object-distance-m'):
            require_near_field_focus_object_distance_m(None, source='--focus-object-distance-m')


# The estimate is patched below, so its inputs are never read.
_UNREAD: Any = None


_KEPT = AutofocusEstimate(
    defocus_distance_m=2.0e-3,
    candidates_m=(),
    residuals=(),
    num_patterns=0,
    kept_initial=True,
    on_edge=False,
)


def _estimate_with(
    monkeypatch: pytest.MonkeyPatch,
    estimate: ParallaxEstimate | None | Exception,
    refined: AutofocusEstimate | Exception = _KEPT,
    *,
    autofocus: bool = True,
) -> float:
    def fake(*args: object, **kwargs: object) -> ParallaxEstimate | None:
        if isinstance(estimate, Exception):
            raise estimate

        return estimate

    def fake_autofocus(*args: object, **kwargs: object) -> AutofocusEstimate:
        if isinstance(refined, Exception):
            raise refined

        return refined

    monkeypatch.setattr(_reconstruct_common, 'estimate_defocus_from_assembled_data', fake)
    monkeypatch.setattr(_reconstruct_common, 'estimate_defocus_by_autofocus', fake_autofocus)
    return estimate_focus_object_distance_m(
        logging.getLogger(),
        _UNREAD,
        _UNREAD,
        _UNREAD,
        _UNREAD,
        photon_wavelength_m=1.0e-10,
        detector_distance_m=2.0,
        source='--focus-object-distance-m',
        autofocus=autofocus,
    )


_RESOLVED = ParallaxEstimate(
    defocus_distance_m=2.0e-3, standard_error_m=1.0e-5, depth_of_focus_m=1.0e-4, num_bins=9
)


class TestEstimateFocusObjectDistance:
    """A resolved parallax sets the focus; anything less leaves it at zero, and says so."""

    def test_a_resolved_defocus_sets_the_negated_focus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Autofocus keeps it here, as it does near focus."""
        assert _estimate_with(monkeypatch, _RESOLVED) == pytest.approx(-2.0e-3)

    def test_autofocus_refines_a_resolved_defocus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        refined = replace(_KEPT, defocus_distance_m=2.6e-3, num_patterns=100, kept_initial=False)

        assert _estimate_with(monkeypatch, _RESOLVED, refined) == pytest.approx(-2.6e-3)

    def test_a_failed_autofocus_keeps_the_parallax_defocus(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            focus_m = _estimate_with(monkeypatch, _RESOLVED, ValueError('no position pairs'))

        assert focus_m == pytest.approx(-2.0e-3)
        assert 'Autofocus failed' in caplog.text

    def test_without_autofocus_the_parallax_defocus_is_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unused = AssertionError('autofocus is not asked for')

        assert _estimate_with(monkeypatch, _RESOLVED, unused, autofocus=False) == pytest.approx(
            -2.0e-3
        )

    def test_an_edge_minimum_is_applied_with_a_remedy(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        refined = replace(_KEPT, defocus_distance_m=6.0e-3, kept_initial=False, on_edge=True)

        with caplog.at_level(logging.WARNING):
            assert _estimate_with(monkeypatch, _RESOLVED, refined) == pytest.approx(-6.0e-3)

        assert 'outermost candidate' in caplog.text
        assert '--focus-object-distance-m' in caplog.text

    @pytest.mark.parametrize(
        'estimate',
        [
            # Half a depth of focus: indistinguishable from the focus.
            ParallaxEstimate(
                defocus_distance_m=5.0e-5,
                standard_error_m=1.0e-6,
                depth_of_focus_m=1.0e-4,
                num_bins=9,
            ),
            None,
            ValueError('too few bright-field bins'),
        ],
        ids=['unresolved', 'none', 'refused'],
    )
    def test_anything_less_is_zero_with_a_remedy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        estimate: ParallaxEstimate | None | Exception,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert _estimate_with(monkeypatch, estimate) == 0.0

        assert '--focus-object-distance-m' in caplog.text


def test_the_beam_estimate_takes_the_moments_accumulated_during_assembly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The estimate agrees with one pass, also after a filter drops a pattern."""
    rng = numpy.random.default_rng(3)
    patterns = rng.poisson(50.0, size=(6, 16, 16)).astype(numpy.uint32)
    arrays = [
        SimpleDiffractionArray('a', numpy.arange(3), patterns[:3]),
        SimpleDiffractionArray('b', numpy.arange(3, 6), patterns[3:]),
    ]
    metadata = DiffractionMetadata(
        num_patterns_per_array=[3, 3],
        pattern_dtype=numpy.dtype(numpy.uint32),
        detector_extent=ImageExtent(width_px=16, height_px=16),
        detector_pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
    )
    dataset = SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), arrays)
    positions = ProbePositionSequence(
        [ProbePosition(index=i, x_m=0.0, y_m=0.0, i0=100.0 - i) for i in range(6)]
    )
    assembly_moments = AssemblyMomentAccumulator()
    assembled = assemble_dataset(dataset, on_array_assembled=assembly_moments.add_assembled)
    keep = assembled.get_indexes() != 4
    filtered = AssembledDiffractionData(
        assembled.get_indexes()[keep],
        assembled.get_patterns()[keep],
        assembled.get_pixel_geometry(),
        assembled.get_bad_pixels(),
    )

    with caplog.at_level(logging.INFO):
        accumulated = estimate_incident_beam_from_assembled_data(
            filtered, positions, assembly_moments=assembly_moments, i0_photons_per_count=0.6
        )

    one_pass = estimate_incident_beam_from_assembled_data(
        filtered, positions, i0_photons_per_count=0.6
    )

    assert 'accumulated during assembly' in caplog.text
    assert accumulated.incident_photon_count == pytest.approx(one_pass.incident_photon_count)
    assert accumulated.mean_amplitude_transmission == pytest.approx(
        one_pass.mean_amplitude_transmission
    )
    numpy.testing.assert_array_equal(accumulated.moments.indexes, [0, 1, 2, 3, 5])


class TestInvalidCountThreshold:
    """The invalid-pixel marker: the instrument's for a named dtype, else the dtype maximum."""

    def test_an_instrument_marker_wins_for_its_dtype(self) -> None:
        assert invalid_count_threshold(numpy.dtype('uint16'), {'uint16': 4095}) == 4095

    def test_other_dtypes_keep_their_maximum(self) -> None:
        threshold = invalid_count_threshold(numpy.dtype('uint32'), {'uint16': 4095})

        assert threshold == numpy.iinfo(numpy.uint32).max

    def test_without_a_marker_it_is_the_dtype_maximum(self) -> None:
        assert invalid_count_threshold(numpy.dtype('uint16')) == 65535

    def test_float_data_has_no_marker(self) -> None:
        assert invalid_count_threshold(numpy.dtype('float32')) is None


def test_printing_another_librarys_options_takes_its_only_algorithm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # pty-chi's LSQML is the fallback only for pty-chi; naming another library alone must
    # not ask it for an algorithm it does not have.
    pytest.importorskip('cdtools')
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    _reconstruct_common.add_print_options_argument(parser)

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(['--print-options', 'cdtools'])

    assert caught.value.code == 0
    document = yaml.safe_load(capsys.readouterr().out)
    assert (document['library'], document['algorithm']) == ('cdtools', 'FancyPtycho')


def test_an_nccl_flag_lives_on_the_ranks_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    # A library puts each rank on its own GPU through torch's default device, while the
    # current device stays cuda:0 everywhere. A flag there would have every rank but the
    # first build a second communicator its peers never join, hanging the run.
    import torch

    devices: list[object] = []
    real_tensor = torch.tensor

    def recording_tensor(data: object, **kwargs: Any) -> torch.Tensor:
        devices.append(kwargs.pop('device', None))
        return real_tensor(data, **kwargs)

    monkeypatch.setattr('torch.distributed.is_initialized', lambda: True)
    monkeypatch.setattr('torch.distributed.get_backend', lambda: 'nccl')
    monkeypatch.setattr('torch.distributed.all_reduce', lambda tensor, op=None: None)
    monkeypatch.setattr(torch, 'get_default_device', lambda: torch.device('cuda', 1))
    monkeypatch.setattr(torch, 'tensor', recording_tensor)

    assert not _reconstruct_common._agree_on_cancellation(CancellationToken())
    assert devices == [torch.device('cuda', 1)]
