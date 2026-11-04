"""Unit tests for the EPICS MDA probe-position readers.

The fixtures are synthesized by :mod:`mda_file_fixtures`, an XDR writer independent of
the reader: a flat rank-1 scan, a rank-2 scan whose outer readback holds y while each
lower scan holds x for :class:`MDAPositionFileReader`, and scans whose axes are detector
channels for :class:`MDADetectorChannelPositionFileReader`.

``xdrlib`` was removed from the standard library in Python 3.13, so the plugin
falls back to a vendored copy. On 3.11 and 3.12 that fallback never runs, which
would leave the code production depends on untested here; ``test_vendored_xdrlib_*``
forces it on every interpreter.
"""

from __future__ import annotations

from pathlib import Path
import json
import pkgutil
import subprocess
import sys

import h5py
import numpy
import pytest

import ptychodus.plugins
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionParseError, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.plugins.aps02id_diffraction_file import APS2IDDiffractionFileReader
from ptychodus.plugins.mda import _xdrlib, mda_position_file
from ptychodus.plugins.mda.mda_position_file import (
    RASTER_LINE_INDEX_STRIDE,
    EpicsType,
    MDADetectorChannelPositionFileReader,
    MDAFile,
    MDAPositionFileReader,
    read_detector_channel,
)

from mda_file_fixtures import (
    write_channel_mda,
    write_flat_mda,
    write_nested_channel_mda,
    write_nested_mda,
)

MICROMETER_M = 1e-6


# The description only supplies the name a registration would show in a file dialog.
# These tests exercise how positions are read and numbered, so it is a stub.
_TEST_DESCRIPTION = 'Test Files'


def _mda_reader(
    *, scale_to_meters: float, line_index_stride: int | None = None
) -> MDAPositionFileReader:
    return MDAPositionFileReader(
        scale_to_meters,
        description=_TEST_DESCRIPTION,
        line_index_stride=line_index_stride,
    )


def _aps2id_reader(*, line_index_stride: int | None = None) -> APS2IDDiffractionFileReader:
    return APS2IDDiffractionFileReader(
        description=_TEST_DESCRIPTION, line_index_stride=line_index_stride
    )


MILLIMETER_M = 1e-3

MDA_READER_NAMES = frozenset(
    {'MDA', 'APS_2IDD', 'APS_2IDE', 'APS_34IDE', 'APS_BNP', 'APS_ISN_MDA', 'CNM_APS_HXN'}
)


@pytest.fixture
def flat_mda(tmp_path: Path) -> Path:
    return write_flat_mda(
        tmp_path / 'flat.mda',
        xs=[1.0, 2.0, 3.0, 4.0],
        ys=[10.0, 20.0, 30.0, 40.0],
    )


@pytest.fixture
def nested_mda(tmp_path: Path) -> Path:
    return write_nested_mda(
        tmp_path / 'nested.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_flat_header_and_scan_metadata(flat_mda: Path) -> None:
    mda_file = MDAFile.read(flat_mda)

    assert mda_file.header.scan_number == 1
    assert mda_file.header.data_rank == 1
    assert mda_file.header.dimensions == [4]
    assert mda_file.header.is_regular
    assert mda_file.header.has_extra_pvs

    assert mda_file.scan.header.num_requested_points == 4
    assert mda_file.scan.info.scan_name == 'scan1'
    assert mda_file.scan.info.num_positioners == 2
    assert mda_file.scan.info.num_detectors == 1
    assert mda_file.scan.info.num_triggers == 1
    assert mda_file.scan.info.positioner[0].name == 'positioner0'
    assert mda_file.scan.info.positioner[0].unit == 'um'
    assert not mda_file.scan.lower_scans


def test_flat_extra_pvs_round_trip_both_epics_types(flat_mda: Path) -> None:
    """A counted string and a DBR_CTRL_DOUBLE take different branches of _read_pv."""
    extra_pvs = MDAFile.read(flat_mda).extra_pvs

    assert [pv.name for pv in extra_pvs] == ['S:SRcurrentAI', '2ide:sampleName']

    current, sample = extra_pvs
    assert current.epics_type == EpicsType.DBR_CTRL_DOUBLE
    assert current.unit == 'mA'
    assert current.value == pytest.approx([102.5])

    assert sample.epics_type == EpicsType.DBR_STRING
    assert sample.value == 'test-sample'


def test_nested_reader_walks_lower_scans(nested_mda: Path) -> None:
    """Rank-2 files pair each outer y with a whole lower scan of x values."""
    positions = _mda_reader(scale_to_meters=MICROMETER_M).read(nested_mda)

    assert len(positions) == 6
    assert [point.index for point in positions] == list(range(6))

    xs = [point.x_m for point in positions]
    assert xs == pytest.approx([v * MICROMETER_M for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)])

    assert positions[0].y_m == pytest.approx(100.0 * MICROMETER_M)
    assert positions[2].y_m == pytest.approx(100.0 * MICROMETER_M)
    assert positions[3].y_m == pytest.approx(200.0 * MICROMETER_M)
    assert positions[5].x_m == pytest.approx(6.0 * MICROMETER_M)


def test_nested_reader_rejects_a_deeper_rank(tmp_path: Path) -> None:
    """A rank-3 file nests a raster under each outer point, not a row.

    Walked as rank 2, its lower scans pair one point apiece with the outer readback, so a
    scan of thousands of frames returns a handful of positions -- along whichever axis the
    outermost scan drove, in whatever units that stage reports. The fixture declares the
    deeper rank in its header only, which is all the guard reads before refusing.
    """
    mda = write_nested_mda(
        tmp_path / 'rank3.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        header_dimensions=[2, 2, 3],
    )

    with pytest.raises(ProbePositionParseError, match='rank-3'):
        _mda_reader(scale_to_meters=MICROMETER_M).read(mda)


# --- aborted scans: current_point is the count of rows actually written


def test_channel_reader_drops_the_preallocated_tail(tmp_path: Path) -> None:
    """Past current_point the file holds zeros, not coordinates at the origin."""
    mda = write_channel_mda(
        tmp_path / 'aborted.mda',
        positioner=[1.0, 2.0, 3.0, 0.0, 0.0],
        channels=[
            ('Piezo Y (all)', [10.0, 20.0, 30.0, 0.0, 0.0]),
            ('X Axis', [1.0, 2.0, 3.0, 0.0, 0.0]),
        ],
        current_point=3,
    )

    positions = _isn_reader().read(mda)

    assert len(positions) == 3
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MILLIMETER_M for v in (1.0, 2.0, 3.0)]
    )
    assert all(point.x_m != 0.0 for point in positions)


def test_nested_reader_stops_at_the_last_written_row(tmp_path: Path) -> None:
    """Unwritten rows carry offset 0; seeking there would reparse the file header."""
    mda = write_nested_mda(
        tmp_path / 'aborted.mda',
        ys=[100.0, 200.0, 300.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        current_point=2,
    )

    positions = _mda_reader(scale_to_meters=MICROMETER_M).read(mda)

    assert len(positions) == 4
    assert [point.index for point in positions] == [0, 1, 2, 3]
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MICROMETER_M for v in (1.0, 2.0, 3.0, 4.0)]
    )
    assert positions[3].y_m == pytest.approx(200.0 * MICROMETER_M)


def test_nested_reader_halts_at_a_hole_rather_than_skipping_it(tmp_path: Path) -> None:
    """A zero offset inside the written range means the file disagrees with itself.

    Continuing past it would pair every later x row with the wrong y, which is worse
    than returning fewer points, so the reader stops at the hole.
    """
    mda = write_nested_mda(
        tmp_path / 'hole.mda',
        ys=[100.0, 200.0, 300.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        hole_at=1,
    )

    positions = _mda_reader(scale_to_meters=MICROMETER_M).read(mda)

    assert len(positions) == 2
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MICROMETER_M for v in (1.0, 2.0)]
    )
    assert all(point.y_m == pytest.approx(100.0 * MICROMETER_M) for point in positions)


def test_channel_reader_rejects_a_scan_with_no_points(tmp_path: Path) -> None:
    mda = write_channel_mda(
        tmp_path / 'empty.mda',
        positioner=[0.0, 0.0],
        channels=[('Piezo Y (all)', [0.0, 0.0]), ('X Axis', [0.0, 0.0])],
        current_point=0,
    )

    with pytest.raises(ProbePositionParseError) as info:
        _isn_reader().read(mda)

    assert '0 of 2' in str(info.value)


def test_nested_reader_rejects_a_scan_with_no_points(tmp_path: Path) -> None:
    mda = write_nested_mda(
        tmp_path / 'empty.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0]],
        current_point=0,
    )

    with pytest.raises(ProbePositionParseError):
        _mda_reader(scale_to_meters=MICROMETER_M).read(mda)


def test_complete_scan_is_not_reported_as_aborted(flat_mda: Path) -> None:
    assert not MDAFile.read(flat_mda).is_aborted


def test_aborted_scan_is_reported(tmp_path: Path) -> None:
    mda = write_flat_mda(tmp_path / 'aborted.mda', xs=[1.0, 0.0], ys=[10.0, 0.0], current_point=1)

    assert MDAFile.read(mda).is_aborted


def test_vendored_xdrlib_is_importable() -> None:
    """The fallback must ship; load_plugins() would otherwise skip MDA in silence."""
    assert hasattr(_xdrlib, 'Packer')
    assert hasattr(_xdrlib, 'Unpacker')


def test_vendored_xdrlib_parses_identically(
    isn_mda: Path, nested_mda: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the Python 3.13 code path on whatever interpreter is running.

    ``xdrlib`` is a module global in mda_position_file, so swapping the attribute
    redirects every unpacker in the file without touching sys.modules.
    """
    reader = _isn_reader()
    nested_reader = _mda_reader(scale_to_meters=MICROMETER_M)

    def coordinates() -> list[tuple[int, float, float]]:
        return [
            (point.index, point.x_m, point.y_m)
            for point in (*reader.read(isn_mda), *nested_reader.read(nested_mda))
        ]

    with_ambient = coordinates()
    monkeypatch.setattr(mda_position_file, 'xdrlib', _xdrlib)

    assert coordinates() == with_ambient
    assert mda_position_file.xdrlib is _xdrlib


_COLD_IMPORT_CHILD = """
import json
import sys


class BlockXdrlib:
    def find_spec(self, name, path=None, target=None):
        if name == 'xdrlib':
            raise ModuleNotFoundError("No module named 'xdrlib'", name=name)

        return None


sys.meta_path.insert(0, BlockXdrlib())

import ptychodus.plugins.mda as package
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.mda import mda_position_file

registry = PluginRegistry()
package.register_plugins(registry)

json.dump(
    {
        'xdrlib': mda_position_file.xdrlib.__name__,
        'readers': [plugin.simple_name for plugin in registry.probe_position_file_readers],
    },
    sys.stdout,
)
"""


def test_fallback_survives_the_partially_initialized_package() -> None:
    """Reproduce the real Python 3.13 import order in a cold interpreter.

    load_plugins() imports the package, whose ``__init__`` imports
    mda_position_file, which -- with no stdlib xdrlib -- imports ``_xdrlib`` back
    out of that same, still-executing package. It resolves because CPython falls
    back to importing the submodule when the attribute is not yet bound, but the
    cycle is real and is invisible on any interpreter that still ships xdrlib.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _COLD_IMPORT_CHILD],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    result = json.loads(completed.stdout)
    assert result['xdrlib'] == 'ptychodus.plugins.mda._xdrlib'
    assert MDA_READER_NAMES <= set(result['readers'])


def test_mda_package_is_visible_to_the_plugin_scanner() -> None:
    """The vendored module lives inside the package, so only the package is scanned.

    load_plugins() walks the top of ptychodus.plugins with pkgutil.iter_modules,
    which yields a directory only when it has an __init__.py. That file is what
    makes the MDA readers register, and its absence would drop them silently.
    """
    names = {
        module.name
        for module in pkgutil.iter_modules(
            ptychodus.plugins.__path__, ptychodus.plugins.__name__ + '.'
        )
    }

    assert 'ptychodus.plugins.mda' in names
    assert 'ptychodus.plugins.mda._xdrlib' not in names


def test_every_mda_reader_registers() -> None:
    """A missing vendored xdrlib is a caught ModuleNotFoundError, not a test failure.

    load_plugins() logs and skips a plugin it cannot import, by design, so nothing
    else in the suite notices when these readers disappear.
    """
    registry = PluginRegistry.load_plugins()
    names = {plugin.simple_name for plugin in registry.probe_position_file_readers}

    assert MDA_READER_NAMES <= names


# --- detector-channel positions (APS 19-ID-E In-situ Nanoprobe) ----------------------
#
# An ISN fly scan drives one trajectory positioner and records both axis encoders as
# detector channels. The descriptions below are the ones real 19idAERO files carry.


def _isn_reader(**overrides: object) -> MDADetectorChannelPositionFileReader:
    kwargs: dict[str, object] = {
        'description': _TEST_DESCRIPTION,
        'x_channel': 'X Axis',
        'y_channel': 'Piezo Y',
        'x_index_fallback': 1,
        'y_index_fallback': 0,
    }
    kwargs.update(overrides)
    return MDADetectorChannelPositionFileReader(MILLIMETER_M, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def isn_mda(tmp_path: Path) -> Path:
    return write_channel_mda(
        tmp_path / 'isn.mda',
        positioner=[1.0, 2.0, 3.0],
        channels=[
            ('Piezo Y (all)', [10.0, 20.0, 30.0]),
            ('X Axis', [1.5, 2.5, 3.5]),
            ('Z Axis', [-1.0, -1.0, -1.0]),
        ],
    )


def test_channel_reader_selects_axes_by_description(isn_mda: Path) -> None:
    positions = _isn_reader().read(isn_mda)

    assert [point.index for point in positions] == [0, 1, 2]
    assert [pytest.approx(point.x_m) for point in positions] == [
        1.5 * MILLIMETER_M,
        2.5 * MILLIMETER_M,
        3.5 * MILLIMETER_M,
    ]
    assert [pytest.approx(point.y_m) for point in positions] == [
        10.0 * MILLIMETER_M,
        20.0 * MILLIMETER_M,
        30.0 * MILLIMETER_M,
    ]


def test_channel_reader_prefers_the_description_over_the_index(tmp_path: Path) -> None:
    """Reordered channels must still be read correctly; that is the point of matching."""
    mda = write_channel_mda(
        tmp_path / 'reordered.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('Z Axis', [-1.0, -1.0]),
            ('Piezo Y (all)', [10.0, 20.0]),
            ('X Axis', [1.5, 2.5]),
        ],
    )
    positions = _isn_reader().read(mda)

    assert positions[0].x_m == pytest.approx(1.5 * MILLIMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)


def test_channel_reader_falls_back_to_indexes(tmp_path: Path) -> None:
    """A file whose descriptions were changed still reads through the pinned indexes."""
    mda = write_channel_mda(
        tmp_path / 'undescribed.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('unlabelled 0', [10.0, 20.0]),
            ('unlabelled 1', [1.5, 2.5]),
        ],
    )
    positions = _isn_reader().read(mda)

    assert positions[0].x_m == pytest.approx(1.5 * MILLIMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)


def test_channel_reader_rejects_an_ambiguous_description(tmp_path: Path) -> None:
    """Two plausible channels is an error: picking the first would be a silent guess."""
    mda = write_channel_mda(
        tmp_path / 'ambiguous.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('Piezo Y (all)', [10.0, 20.0]),
            ('X Axis', [1.5, 2.5]),
            ('X Axis coarse', [9.0, 9.0]),
        ],
    )
    with pytest.raises(ProbePositionParseError, match='Cannot choose between them'):
        _isn_reader().read(mda)


def test_channel_reader_names_the_channels_when_nothing_matches(tmp_path: Path) -> None:
    mda = write_channel_mda(
        tmp_path / 'nomatch.mda',
        positioner=[1.0, 2.0],
        channels=[('unlabelled 0', [10.0, 20.0])],
    )
    with pytest.raises(ProbePositionParseError, match='unlabelled 0'):
        _isn_reader(x_index_fallback=9, y_index_fallback=9).read(mda)


def test_channel_reader_prefers_an_exact_pv_name(tmp_path: Path) -> None:
    """A PV name is matched before any description, which may have been relabeled."""
    mda = write_channel_mda(
        tmp_path / 'named.mda',
        positioner=[1.0, 2.0],
        channels=[('ZP Piezo Y', [10.0, 20.0]), ('', [1.5, 2.5]), ('ZP Piezo X', [9.0, 9.0])],
        channel_names=['26idcnpi:m35.RBV', '26idcnpi:m34.RBV', 'other:m34.VAL'],
    )
    reader = MDADetectorChannelPositionFileReader(
        MICROMETER_M,
        description=_TEST_DESCRIPTION,
        x_channel='26idcnpi:m34.RBV',
        y_channel='26idcnpi:m35.RBV',
    )
    positions = reader.read(mda)

    assert positions[1].x_m == pytest.approx(2.5 * MICROMETER_M)
    assert positions[1].y_m == pytest.approx(20.0 * MICROMETER_M)


def test_channel_reader_without_a_fallback_names_the_channels(tmp_path: Path) -> None:
    mda = write_channel_mda(
        tmp_path / 'nomatch.mda',
        positioner=[1.0, 2.0],
        channels=[('unlabelled 0', [10.0, 20.0])],
    )
    reader = MDADetectorChannelPositionFileReader(
        MICROMETER_M, description=_TEST_DESCRIPTION, x_channel='absent', y_channel='absent'
    )

    with pytest.raises(ProbePositionParseError, match='detector0'):
        reader.read(mda)


def test_channel_reader_applies_signed_per_axis_scales(isn_mda: Path) -> None:
    positions = _isn_reader(x_scale_to_meters=-MICROMETER_M).read(isn_mda)

    assert positions[0].x_m == pytest.approx(-1.5 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)


def _write_raster(path: Path, *, num_rows: int = 3) -> Path:
    """A rank-2 26-ID-C-style raster: x and y encoders in every row's lower scan."""
    rows = [
        [[10.0 * row + column for column in range(4)], [100.0 + row] * 4, [-1.0] * 4]
        for row in range(num_rows)
    ]
    return write_nested_channel_mda(
        path,
        outer=[float(row) for row in range(num_rows)],
        channel_names=['26idcnpi:m34.RBV', '26idcnpi:m35.RBV', 'atto2:m3.RBV'],
        rows=rows,
    )


def test_channel_reader_flattens_a_raster_row_by_row(tmp_path: Path) -> None:
    mda = _write_raster(tmp_path / 'raster.mda')
    reader = MDADetectorChannelPositionFileReader(
        MICROMETER_M,
        description=_TEST_DESCRIPTION,
        x_channel='26idcnpi:m34.RBV',
        y_channel='26idcnpi:m35.RBV',
    )
    positions = reader.read(mda)

    assert [point.index for point in positions] == list(range(12))
    assert [point.x_m / MICROMETER_M for point in positions] == pytest.approx(
        [10.0 * row + column for row in range(3) for column in range(4)]
    )
    assert [point.y_m / MICROMETER_M for point in positions] == pytest.approx(
        [100.0 + row for row in range(3) for _ in range(4)]
    )


def test_channel_reader_refuses_a_deeper_rank(tmp_path: Path) -> None:
    mda = write_nested_mda(
        tmp_path / 'rank3.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        header_dimensions=[2, 2, 3],
    )

    with pytest.raises(ProbePositionParseError, match='rank-3'):
        _isn_reader().read(mda)


def test_read_detector_channel_concatenates_rows(tmp_path: Path) -> None:
    mda_file = MDAFile.read(_write_raster(tmp_path / 'raster.mda', num_rows=2))
    values = read_detector_channel(mda_file, '26idcnpi:m35.RBV')

    assert values is not None
    assert values.tolist() == pytest.approx([100.0] * 4 + [101.0] * 4)
    assert read_detector_channel(mda_file, 'absent:PV') is None


def test_cnm_registration_reads_the_zone_plate_encoders(tmp_path: Path) -> None:
    mda = _write_raster(tmp_path / '26idbSOFT_0001.mda')
    registry = PluginRegistry.load_plugins()
    reader = registry.probe_position_file_readers.get_strategy_by_name('CNM_APS_HXN')
    positions = reader.read(mda)

    assert len(positions) == 12
    assert abs(positions[5].x_m) == pytest.approx(11.0 * MICROMETER_M)
    assert abs(positions[5].y_m) == pytest.approx(101.0 * MICROMETER_M)


# --- Line-major scan indexes
#
# The 2-ID-E fly scan drives the positioner over more points per line than the detector
# records, so numbering either side by running count slides the pairing by the surplus,
# one line at a time, while every pattern still finds a match. These pin the numbering
# that makes the two agree, and the pairing that falls out of it.


def test_nested_reader_numbers_by_running_count_by_default(nested_mda: Path) -> None:
    """Every other instrument keeps the contiguous numbering, so the stride is opt-in."""
    positions = _mda_reader(scale_to_meters=MICROMETER_M).read(nested_mda)

    assert [point.index for point in positions] == [0, 1, 2, 3, 4, 5]


def test_nested_reader_numbers_line_major_when_given_a_stride(nested_mda: Path) -> None:
    """A point is named by its line and column, not by how many points preceded it."""
    positions = _mda_reader(
        scale_to_meters=MICROMETER_M,
        line_index_stride=RASTER_LINE_INDEX_STRIDE,
    ).read(nested_mda)

    assert [point.index for point in positions] == [
        0,
        1,
        2,
        RASTER_LINE_INDEX_STRIDE,
        RASTER_LINE_INDEX_STRIDE + 1,
        RASTER_LINE_INDEX_STRIDE + 2,
    ]

    # The coordinates are untouched; only the numbering changed.
    assert positions[3].y_m == pytest.approx(200.0 * MICROMETER_M)
    assert positions[3].x_m == pytest.approx(4.0 * MICROMETER_M)


def test_nested_reader_rejects_a_line_that_reaches_the_stride(nested_mda: Path) -> None:
    """A longer line would run into the next line's numbers, so it is refused."""
    reader = _mda_reader(scale_to_meters=MICROMETER_M, line_index_stride=3)

    with pytest.raises(ProbePositionParseError, match='reaches the line-major index stride'):
        reader.read(nested_mda)


def test_aps_2ide_registration_numbers_line_major(nested_mda: Path) -> None:
    """The stride is a property of the 2-ID-E registration, not of the reader class."""
    registry = PluginRegistry.load_plugins()

    line_major = registry.probe_position_file_readers.get_strategy_by_name('APS_2IDE')
    contiguous = registry.probe_position_file_readers.get_strategy_by_name('APS_2IDD')

    assert [point.index for point in line_major.read(nested_mda)][3] == RASTER_LINE_INDEX_STRIDE
    assert [point.index for point in contiguous.read(nested_mda)] == [0, 1, 2, 3, 4, 5]


# --- Pairing a line-major series with a line-major raster


def _write_line_series(directory: Path, frames_per_line: list[int]) -> Path:
    """One Eiger-style HDF5 per scan line, numbered from one as the instrument writes them."""
    directory.mkdir(parents=True, exist_ok=True)

    for line, num_frames in enumerate(frames_per_line):
        path = directory / f'fly007_data_{line + 1:03d}.h5'

        with h5py.File(path, 'w') as h5_file:
            h5_file.create_dataset(
                '/entry/data/data', data=numpy.zeros((num_frames, 2, 2), dtype=numpy.uint16)
            )

    return directory / 'fly007_data_001.h5'


def _pair(mda_path: Path, series_path: Path) -> list[tuple[float, float]]:
    """Read both sides the way 2-ID-E is registered and pair them by scan index."""
    dataset = _aps2id_reader(line_index_stride=RASTER_LINE_INDEX_STRIDE).read(series_path)
    indexes = numpy.concatenate([array.get_indexes() for array in dataset])
    pixel_geometry = PixelGeometry(width_m=1e-6, height_m=1e-6)

    assembled = AssembledDiffractionData(
        indexes=indexes.astype(numpy.intp),
        patterns=numpy.zeros((indexes.size, 2, 2), dtype=numpy.uint16),
        pixel_geometry=pixel_geometry,
        bad_pixels=numpy.zeros((2, 2), dtype=numpy.bool_),
    )
    positions = _mda_reader(
        scale_to_meters=MICROMETER_M,
        line_index_stride=RASTER_LINE_INDEX_STRIDE,
    ).read(mda_path)
    product = Product(
        metadata=ProductMetadata(
            name='test',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10_000.0,
            probe_photon_count=1,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(list(positions)),
        probes=ProbeEnsemble(
            array=numpy.zeros((1, 1, 2, 2), dtype=numpy.complex128),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.zeros((1, 8, 8), dtype=numpy.complex128),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
            layer_spacing_m=[],
        ),
        losses=[],
    )
    paired = prepare_reconstruct_input(assembled, product).product.probe_positions
    return [(point.x_m / MICROMETER_M, point.y_m / MICROMETER_M) for point in paired]


def test_surplus_commanded_points_go_unclaimed(tmp_path: Path) -> None:
    """The positioner is driven over five points per line; the detector records three.

    Every pattern must land on the column the detector actually exposed -- exactly, with
    no interpolation -- and the two points the detector never reached simply have no
    pattern. Numbering by running count instead would place line one's patterns two
    columns along, and line two's four columns along.
    """
    mda = write_nested_mda(
        tmp_path / 'raster.mda',
        ys=[10.0, 20.0, 30.0],
        xs_per_row=[[1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
    )
    series = _write_line_series(tmp_path / 'ptycho', [3, 3, 3])

    assert _pair(mda, series) == [
        (1.0, 10.0),
        (2.0, 10.0),
        (3.0, 10.0),
        (1.0, 20.0),
        (2.0, 20.0),
        (3.0, 20.0),
        (1.0, 30.0),
        (2.0, 30.0),
        (3.0, 30.0),
    ]


def test_a_line_the_detector_cut_short_claims_fewer_points(tmp_path: Path) -> None:
    """A short line must not shift the lines after it, which a running count would."""
    mda = write_nested_mda(
        tmp_path / 'raster.mda',
        ys=[10.0, 20.0, 30.0],
        xs_per_row=[[1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
    )
    series = _write_line_series(tmp_path / 'ptycho', [3, 1, 3])

    assert _pair(mda, series) == [
        (1.0, 10.0),
        (2.0, 10.0),
        (3.0, 10.0),
        (1.0, 20.0),
        (1.0, 30.0),
        (2.0, 30.0),
        (3.0, 30.0),
    ]


def test_a_detector_line_past_the_raster_is_dropped(tmp_path: Path) -> None:
    """An aborted scan keeps the line the detector was on; the MDA file truncates before it."""
    mda = write_nested_mda(
        tmp_path / 'raster.mda',
        ys=[10.0, 20.0, 30.0],
        xs_per_row=[[1.0, 2.0, 3.0, 4.0, 5.0]] * 3,
        current_point=2,
    )
    series = _write_line_series(tmp_path / 'ptycho', [3, 3, 3])

    # The MDA file kept two lines, so the third line's three patterns have no position.
    assert _pair(mda, series) == [
        (1.0, 10.0),
        (2.0, 10.0),
        (3.0, 10.0),
        (1.0, 20.0),
        (2.0, 20.0),
        (3.0, 20.0),
    ]
