"""Reader for the APS 31-ID-E ``tomography_scannumbers.txt`` index.

LamNI writes one row per acquired scan into a single space-delimited table under the
experiment's ``dat-files`` directory. The row is the only place that ties a scan number to
its rotation angle and to the tomogram it belongs to, so it is what turns a directory of
numbered scans into an ordered set of projections.

Columns, in order, with no header line::

    scan_no golden_angle encoder_angle measurement_id subtomo_no detector_position label

``encoder_angle`` is the rotation-stage readback and the value that belongs in a product's
``tomography_angle_deg``; it agrees with the ``lsamrot_encoder`` figure in the first line of
the matching ``scan_positions/scan_NNNNN.dat``. ``golden_angle`` is the commanded angle, on
the exact angular grid the scan was planned around.

A tomogram is split into interleaved sub-tomograms: ``subtomo_no`` counts those passes and
``measurement_id`` numbers the projection within one pass, so neither alone identifies a
projection. Neither does the pair, in practice -- a re-run of a projection appends a further
row carrying the same ``(subtomo_no, measurement_id)`` under a new scan number. Rows are
therefore returned exactly as written, and deduplicating is left to the caller, which is the
only party that knows whether a repeat supersedes its predecessor or stands beside it.

Rows that cannot be parsed are logged and skipped rather than raising: the table is appended
to live during an experiment, so a run reading it can catch a partially written final line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import logging

logger = logging.getLogger(__name__)

_NUM_COLUMNS = 7


@dataclass(frozen=True)
class APS31IDEScanRecord:
    """One row of ``tomography_scannumbers.txt``: a scan, its angle, and its place in a tomogram."""

    scan_no: int
    golden_angle_deg: float
    encoder_angle_deg: float
    measurement_id: int
    subtomo_no: int
    detector_position: int
    label: str

    def __str__(self) -> str:
        return f"""scan_no={self.scan_no}
        golden_angle_deg={self.golden_angle_deg}
        encoder_angle_deg={self.encoder_angle_deg}
        measurement_id={self.measurement_id}
        subtomo_no={self.subtomo_no}
        detector_position={self.detector_position}
        label={self.label}
        """


def read_aps31ide_scan_table(file_path: Path) -> list[APS31IDEScanRecord]:
    """Read every parsable row of a LamNI ``tomography_scannumbers.txt``, in file order."""
    records: list[APS31IDEScanRecord] = list()

    with file_path.open(newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for row in csv_reader:
            if not row or row[0].startswith('#'):
                continue

            if len(row) != _NUM_COLUMNS:
                logger.warning('Unexpected row in tomography_scannumbers.txt!')
                logger.debug(row)
                continue

            try:
                record = APS31IDEScanRecord(
                    scan_no=int(row[0]),
                    golden_angle_deg=float(row[1]),
                    encoder_angle_deg=float(row[2]),
                    measurement_id=int(row[3]),
                    subtomo_no=int(row[4]),
                    detector_position=int(row[5]),
                    label=str(row[6]),
                )
            except ValueError:
                logger.warning('Failed to parse row in tomography_scannumbers.txt!')
                logger.debug(row)
            else:
                records.append(record)

    return records
