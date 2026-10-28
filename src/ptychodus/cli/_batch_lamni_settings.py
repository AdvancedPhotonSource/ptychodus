"""The settings file behind ``ptychodus-reconstruct-batch-lamni``, and the scans it selects.

Separate from the driver because three processes need it and only one of them may pull in
a GPU stack: the parent reads the file and plans the run, each worker re-reads the same
text to rebuild its own copy, and neither path should import torch to do so. Nothing here
imports a reconstruction backend.

The settings are a file rather than command-line flags so that a run is reproducible from
something archivable, and the file is the *only* source: a tuning value that is not in it
cannot be set. That is what lets the copy written beside each scan's output stand as the
record of how that scan was reconstructed.

A run driven from Python starts from a :class:`BatchSettings` object and has no file text
to pass on. :func:`dump_batch_settings` renders one, so such a run archives a canonical
spelling of the settings that were actually used -- every default written out, and no
comments -- rather than bytes nobody typed.

Unknown keys are rejected rather than ignored. This file has no writer but a person, so a
key that does not resolve is a typo, and silently running with a different value than the
one that was typed is the failure worth preventing.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptychodus.api.affine import AffineTransform
from ptychodus.plugins.aps31id_lamni.scan_table import APS31IDEScanRecord

__all__ = [
    'Batch',
    'BatchSettings',
    'Diffraction',
    'InitialObject',
    'InitialProbe',
    'MetadataOverrides',
    'Misc',
    'ObjectFromFile',
    'Paths',
    'Positions',
    'WorkerMessage',
    'ProbeFromFile',
    'SimulatedProbe',
    'dump_batch_settings',
    'load_batch_settings',
    'load_batch_settings_from_text',
    'select_label',
    'select_records',
]

logger = logging.getLogger(__name__)

# Scans live under eiger_4 in blocks of a thousand.
_SCAN_BLOCK_SIZE = 1000


class _Section(BaseModel):
    """Base for every block of the file: reject what we cannot explain."""

    model_config = ConfigDict(extra='forbid')


class Paths(_Section):
    experiment_folder: Path
    output_products_folder: Path


class Batch(_Section):
    label: str | None = None
    scan_start: int | None = None
    scan_end: int | None = None
    exclude_scans: list[int] = Field(default_factory=list)
    overwrite_existing_products: bool = False


class ProbeFromFile(_Section):
    file_path: Path | None = None
    file_type: str = 'HDF5'


class SimulatedProbe(_Section):
    fzp_preset: str = 'APS 31-ID-E LamNI'
    fzp_defocus_m: float = 800.0e-6


class InitialProbe(_Section):
    probe_source: Literal['simulated', 'file'] = 'simulated'
    probe_from_file: ProbeFromFile = Field(default_factory=ProbeFromFile)
    simulated_probe: SimulatedProbe = Field(default_factory=SimulatedProbe)
    num_probe_modes: int = Field(default=1, ge=1)
    num_opr_modes: int = Field(default=1, ge=1)

    @model_validator(mode='after')
    def _check_source(self) -> InitialProbe:
        if self.probe_source == 'file' and self.probe_from_file.file_path is None:
            raise ValueError('initial_probe.probe_from_file.file_path is required for "file".')

        return self


class ObjectFromFile(_Section):
    file_path: Path | None = None
    file_type: str = 'HDF5'


class InitialObject(_Section):
    object_source: Literal['simulated', 'file'] = 'simulated'
    object_from_file: ObjectFromFile = Field(default_factory=ObjectFromFile)
    object_padding_px: int = Field(default=512, ge=0)
    num_slices: int = Field(default=1, ge=1)
    object_thickness_m: float | None = None

    @model_validator(mode='after')
    def _check_source_and_thickness(self) -> InitialObject:
        if self.object_source == 'file' and self.object_from_file.file_path is None:
            raise ValueError('initial_object.object_from_file.file_path is required for "file".')

        if self.num_slices > 1 and not (self.object_thickness_m or 0.0) > 0.0:
            # Zero thickness yields the right number of zero-width gaps, so the object is
            # legal and every slice sits at the same depth: multislice in name, with no
            # propagation between slices. Nothing downstream can detect that, so refuse.
            raise ValueError(
                'initial_object.object_thickness_m must be positive when num_slices > 1.'
            )

        return self


class Positions(_Section):
    affine_matrix: list[list[float]] | None = None

    @model_validator(mode='after')
    def _check_matrix(self) -> Positions:
        if self.affine_matrix is None:
            return self

        shape = (len(self.affine_matrix), {len(row) for row in self.affine_matrix})

        if shape not in ((2, {2}), (2, {3})):
            raise ValueError(
                'positions.affine_matrix must be 2x2 or 2x3, row-major in the (x, y) frame.'
            )

        return self

    def get_transform(self) -> AffineTransform | None:
        """The supplied transform, widening a translation-free 2x2 with zero translation."""
        if self.affine_matrix is None:
            return None

        rows = [[*row, 0.0] if len(row) == 2 else list(row) for row in self.affine_matrix]
        (a00, a01, a02), (a10, a11, a12) = rows
        return AffineTransform(a00, a01, a02, a10, a11, a12)


class MetadataOverrides(_Section):
    detector_distance_m: float | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    probe_photon_count: float | None = None
    tomography_angle_deg: float | None = None
    # Near field is opt-in: None leaves every existing batch far field, as it was.
    far_field: bool | None = None
    # Signed, origin at the object, so a focus upstream is negative. Near field only;
    # defaults to the negated simulated_probe.fzp_defocus_m.
    focus_object_distance_m: float | None = None


class Diffraction(_Section):
    # 'auto' suggests the smallest power of two holding the signal above the noise floor.
    crop_extent_px: Annotated[int, Field(ge=1)] | Literal['auto'] | None = None
    beam_center_x_px: int | None = None
    beam_center_y_px: int | None = None
    min_total_counts: int | None = None
    max_valid_count: int | None = None
    bad_pixels_file: Path | None = None
    bad_pixels_file_type: str = 'NPY_Bad_Pixels'


class Misc(_Section):
    seed: int = 0
    num_sync_epochs: int = Field(default=100, ge=1)
    save_diffraction: bool = True
    save_input_product: bool = False
    continue_on_error: bool = False
    log_level: int = logging.INFO


class BatchSettings(_Section):
    """Everything a run needs that is not the pty-chi options."""

    paths: Paths
    batch: Batch = Field(default_factory=Batch)
    initial_probe: InitialProbe = Field(default_factory=InitialProbe)
    initial_object: InitialObject = Field(default_factory=InitialObject)
    positions: Positions = Field(default_factory=Positions)
    metadata_overrides: MetadataOverrides = Field(default_factory=MetadataOverrides)
    diffraction: Diffraction = Field(default_factory=Diffraction)
    misc: Misc = Field(default_factory=Misc)

    @model_validator(mode='after')
    def _check_scan_range(self) -> BatchSettings:
        start, end = self.batch.scan_start, self.batch.scan_end

        if start is not None and end is not None and start > end:
            raise ValueError(f'batch.scan_start ({start}) is above batch.scan_end ({end}).')

        return self

    @property
    def experiment_root(self) -> Path:
        """The experiment directory, accepting either it or the ``data`` directory inside."""
        folder = self.paths.experiment_folder

        if (folder / 'data' / 'dat-files').is_dir():
            return folder

        if (folder / 'dat-files').is_dir():
            return folder.parent

        raise ValueError(
            f'{folder} is not a 31-ID-E experiment directory: expected a "dat-files" '
            'subdirectory either directly inside it or under a "data" subdirectory.'
        )

    @property
    def data_dir(self) -> Path:
        return self.experiment_root / 'data'

    @property
    def scan_table_file(self) -> Path:
        return self.data_dir / 'dat-files' / 'tomography_scannumbers.txt'

    def position_file(self, scan_no: int) -> Path:
        return self.data_dir / 'scan_positions' / f'scan_{scan_no:05d}.dat'

    def diffraction_file(self, scan_no: int) -> Path:
        """Locate the one data file of a scan's detector series.

        The series is written beside a ``*_master_*`` index file that holds no patterns, and
        the stem differs between acquisition software versions, so the directory is globbed
        rather than assumed. An ambiguous match is an error: picking one arbitrarily would
        reconstruct an unpredictable subset.
        """
        block_start = (scan_no // _SCAN_BLOCK_SIZE) * _SCAN_BLOCK_SIZE
        block = f'S{block_start:05d}-{block_start + _SCAN_BLOCK_SIZE - 1:05d}'
        scan_dir = self.data_dir / 'eiger_4' / block / f'S{scan_no:05d}'

        candidates = sorted(
            path for path in scan_dir.glob(f'*_{scan_no:05d}_*.h5') if '_master_' not in path.name
        )

        if not candidates:
            raise FileNotFoundError(f'No diffraction file for scan {scan_no} in {scan_dir}.')

        if len(candidates) > 1:
            names = ', '.join(path.name for path in candidates)
            raise FileNotFoundError(
                f'Scan {scan_no} has {len(candidates)} candidate diffraction files in '
                f'{scan_dir}: {names}.'
            )

        return candidates[0]

    def output_directory(self, scan_no: int) -> Path:
        """Where one scan's standard-layout outputs go.

        Flat, and named the way the raw per-scan directory is, so the two sides of a run
        read alike. The settings file already names one tomogram, which is what an
        intervening label directory would otherwise separate.
        """
        return self.paths.output_products_folder / f'S{scan_no:05d}'


def load_batch_settings_from_text(text: str, source: str = '<text>') -> BatchSettings:
    """Validate settings from the file's text.

    The text, rather than the parsed object, is what travels to a worker and what is
    copied beside each scan's output, so every copy of the settings in a run is built
    from the same bytes the person wrote.
    """
    raw = yaml.safe_load(text)

    if raw is None:
        raise ValueError(f'{source} is empty.')

    if not isinstance(raw, dict):
        raise ValueError(f'{source} does not contain a YAML mapping.')

    return BatchSettings.model_validate(raw)


def load_batch_settings(file_path: Path) -> BatchSettings:
    """Read and validate the settings file."""
    return load_batch_settings_from_text(file_path.read_text(), str(file_path))


def dump_batch_settings(settings: BatchSettings) -> str:
    """Render settings as the YAML text a worker rebuilds from and a run archives.

    The inverse of :func:`load_batch_settings_from_text`, for a run whose settings were
    built rather than read: there is no file text to carry, so one is written from the
    object. ``mode='json'`` renders the ``Path`` fields as strings, which is what
    ``safe_dump`` can represent and what loading turns back into paths; ``sort_keys=False``
    keeps the sections in the order they are declared, for a file meant to be read.
    """
    return yaml.safe_dump(settings.model_dump(mode='json'), sort_keys=False)


def select_label(records: list[APS31IDEScanRecord], requested: str | None) -> str:
    """Choose which tomogram to reconstruct, reporting what else the table holds."""
    counts: dict[str, int] = dict()

    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1

    if not counts:
        raise ValueError('The scan table holds no parsable rows.')

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    summary = ', '.join(f'{label} ({count} scans)' for label, count in ordered)

    if requested is not None:
        if requested not in counts:
            raise ValueError(f'No scans labeled "{requested}". The table holds: {summary}.')

        logger.info(
            'Label: %s (%d scans). The table also holds: %s', requested, counts[requested], summary
        )
        return requested

    label, count = ordered[0]
    logger.warning(
        'No batch.label given; taking "%s", the largest tomogram in the table (%d scans). '
        'The table holds: %s',
        label,
        count,
        summary,
    )
    return label


def select_records(
    records: list[APS31IDEScanRecord], batch: Batch, label: str
) -> list[APS31IDEScanRecord]:
    """Filter the table to the requested scan numbers, in acquisition order."""
    excluded = frozenset(batch.exclude_scans)
    selected = [
        record
        for record in records
        if record.label == label
        and record.scan_no not in excluded
        and (batch.scan_start is None or record.scan_no >= batch.scan_start)
        and (batch.scan_end is None or record.scan_no <= batch.scan_end)
    ]

    # Scan number is acquisition order. Nothing in the run depends on it any longer, but a
    # run's logs and its summary read in the order the data was taken, and the table itself
    # is not guaranteed to be sorted.
    selected.sort(key=lambda record: record.scan_no)
    return selected


class WorkerMessage(StrEnum):
    """What a worker has to say about the scan it was given.

    Here rather than in either process because the worker is reached by name and
    the parent must not import it, and this module is already what the two agree
    on: it renders the record that goes out, so it names the tag that comes back.

    A string enum because the value is what travels. The value is the worker's
    own word and means a scan-completion record -- not a reconstruction result,
    which is a different child's vocabulary and travels under its own tag.
    """

    SCAN_RESULT = 'scan_result'
    """One scan has finished, with whether it ran to completion."""


def record_payload(record: APS31IDEScanRecord) -> dict[str, Any]:
    """The record as plain data, for a payload that must unpickle without heavy imports."""
    return {
        'scan_no': record.scan_no,
        'label': record.label,
        'encoder_angle_deg': record.encoder_angle_deg,
        'text': str(record),
    }
