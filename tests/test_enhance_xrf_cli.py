"""``ptychodus-enhance-xrf``: options round trip, path guards, and an end-to-end run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel
import numpy
import pytest
import yaml

from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    FluorescenceFileKeys,
    load_fluorescence_data,
    save_fluorescence_data,
    save_product,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.processing import RunState
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.storage import (
    DerivedFromRef,
    FluorescenceManifest,
    ResourceKind,
    StorageLayout,
    load_manifest,
    save_fluorescence_resource,
    save_product_resource,
)
from ptychodus.cli import CancellationToken, enhance_xrf

_PIXEL_SIZE_M = 1.0e-7
_CANVAS_PX = 24
_PROBE_PX = 6
_SCAN_SIDE = 6
_CHANNELS = ('Fe', 'Zn')


@pytest.fixture(autouse=True)
def cancellation(monkeypatch: pytest.MonkeyPatch) -> CancellationToken:
    """The run's cancellation token, which ``main()`` would otherwise bind to signals.

    Its root logger is left alone too, and a test cancels by setting the token.
    """
    token = CancellationToken()
    monkeypatch.setattr(enhance_xrf, 'install_signal_handlers', lambda: token)
    monkeypatch.setattr(enhance_xrf, 'configure_logging', lambda args: None)
    monkeypatch.setattr(enhance_xrf, 'open_action_log', lambda *args, **kwargs: None)
    return token


def _product() -> Product:
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    offsets_px = numpy.linspace(-6.0, 6.0, _SCAN_SIDE)
    positions = [
        ProbePosition(index=index, x_m=x_px * _PIXEL_SIZE_M, y_m=y_px * _PIXEL_SIZE_M)
        for index, (y_px, x_px) in enumerate(
            (y_px, x_px) for y_px in offsets_px for x_px in offsets_px
        )
    ]
    yy, xx = numpy.mgrid[:_PROBE_PX, :_PROBE_PX] - (_PROBE_PX - 1) / 2
    probe_array = numpy.exp(-(yy**2 + xx**2) / 4.0).astype(complex)

    return Product(
        metadata=ProductMetadata(
            name='enhance',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=5.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(
            array=probe_array[numpy.newaxis, :, :],
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.ones((_CANVAS_PX, _CANVAS_PX), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def _write_inputs(directory: Path) -> tuple[Path, Path]:
    rng = numpy.random.default_rng(7)
    dataset = FluorescenceDataset(
        _element_maps=[
            ElementMap(name, rng.uniform(1.0, 2.0, (_SCAN_SIDE, _SCAN_SIDE))) for name in _CHANNELS
        ],
        counts_per_second_path=FluorescenceFileKeys.COUNTS_PER_SECOND,
        channel_names_path=FluorescenceFileKeys.CHANNEL_NAMES,
    )
    product_file = directory / 'product.h5'
    fluorescence_file = directory / 'xrf.h5'
    save_product(product_file, _product())
    save_fluorescence_data(fluorescence_file, dataset)
    return product_file, fluorescence_file


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str | Path | UUID) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-enhance-xrf', *map(str, argv)])
    return enhance_xrf.main()


@pytest.mark.parametrize('algorithm', ['two_step', 'vspi'])
def test_printed_options_load_back_to_the_defaults(algorithm: str) -> None:
    plugin = PluginRegistry.load_plugins().fluorescence_enhancers.find_plugin(algorithm)
    assert plugin is not None

    document = yaml.safe_load(enhance_xrf.format_enhancer_options(plugin))

    assert document['algorithm'] == algorithm
    options = enhance_xrf._validate_options(plugin, document['options'])
    assert options == plugin.strategy.get_default_options()


def test_print_options_writes_to_stdout_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, '--print-options', 'two_step')

    assert exit_info.value.code == 0
    assert yaml.safe_load(capsys.readouterr().out)['algorithm'] == 'two_step'


def test_enhancement_writes_the_output_and_leaves_the_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    inputs_before = product_file.read_bytes(), fluorescence_file.read_bytes()
    output_file = tmp_path / 'out' / 'enhanced.h5'

    exit_code = _main(
        monkeypatch,
        '--product',
        product_file,
        '--fluorescence',
        fluorescence_file,
        '--algorithm',
        'two_step',
        '-o',
        output_file,
    )

    assert exit_code is ExitCode.SUCCESS
    enhanced = load_fluorescence_data(output_file)
    assert [emap.name for emap in enhanced] == list(_CHANNELS)
    assert enhanced[0].counts_per_second.shape == (_CANVAS_PX, _CANVAS_PX)
    assert (product_file.read_bytes(), fluorescence_file.read_bytes()) == inputs_before


def test_options_file_selects_the_enhancer_and_its_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    options_file = tmp_path / 'options.yaml'
    options_file.write_text(
        yaml.safe_dump({'algorithm': 'two_step', 'options': {'deconvolution_method': 'identity'}})
    )
    output_file = tmp_path / 'enhanced.h5'

    exit_code = _main(
        monkeypatch,
        '--product',
        product_file,
        '--fluorescence',
        fluorescence_file,
        '--options',
        options_file,
        '-o',
        output_file,
    )

    assert exit_code is ExitCode.SUCCESS
    assert output_file.is_file()


@pytest.mark.parametrize('target', ['fluorescence', 'product'])
def test_output_may_not_be_an_input(
    target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    inputs_before = product_file.read_bytes(), fluorescence_file.read_bytes()
    output_file = fluorescence_file if target == 'fluorescence' else product_file

    with pytest.raises(SystemExit) as exit_info:
        _main(
            monkeypatch,
            '--product',
            product_file,
            '--fluorescence',
            fluorescence_file,
            '--algorithm',
            'two_step',
            '-o',
            output_file,
        )

    assert exit_info.value.code == ExitCode.USAGE
    assert (product_file.read_bytes(), fluorescence_file.read_bytes()) == inputs_before


@pytest.mark.parametrize(
    'document',
    [
        {'algorithm': 'vspi', 'options': {'max_iteration': 5}},
        {'algorithm': 'vspi', 'options': {'max_iterations': 0}},
        {'algorithm': 'two_step'},
        {'options': {}},
        {'algorithm': 'vspi', 'extra': 1},
    ],
    ids=['unknown-field', 'out-of-bounds', 'disagrees-with-flag', 'no-algorithm', 'extra-key'],
)
def test_unusable_options_are_a_usage_error(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    options_file = tmp_path / 'options.yaml'
    options_file.write_text(yaml.safe_dump(document))
    output_file = tmp_path / 'enhanced.h5'

    with pytest.raises(SystemExit) as exit_info:
        _main(
            monkeypatch,
            '--product',
            product_file,
            '--fluorescence',
            fluorescence_file,
            '--algorithm',
            'vspi',
            '--options',
            options_file,
            '-o',
            output_file,
        )

    assert exit_info.value.code == ExitCode.USAGE
    assert not output_file.exists()


def test_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    output_file = tmp_path / 'enhanced.h5'

    exit_code = _main(
        monkeypatch,
        '--product',
        product_file,
        '--fluorescence',
        fluorescence_file,
        '--dry-run',
        '-o',
        output_file,
    )

    assert exit_code is ExitCode.SUCCESS
    assert not output_file.exists()


def _patch_enhance(
    monkeypatch: pytest.MonkeyPatch, cancellation: CancellationToken, behavior: str
) -> None:
    """Make two_step yield its first output and then behave as `behavior` says."""
    registry = PluginRegistry.load_plugins()
    plugin = registry.fluorescence_enhancers.find_plugin('two_step')
    assert plugin is not None
    original_enhance = plugin.strategy.enhance

    def patched_enhance(
        input_: FluorescenceEnhancerInput, *, options: BaseModel
    ) -> Iterator[FluorescenceEnhancerOutput]:
        if behavior == 'cancel before':
            cancellation.cancel('SIGTERM')
            return

        if behavior == 'fail before':
            raise RuntimeError('diverged')

        for output in original_enhance(input_, options=options):
            yield output

            if behavior == 'cancel after':
                cancellation.cancel('SIGTERM')
            elif behavior == 'fail after':
                raise RuntimeError('diverged')

    monkeypatch.setattr(plugin.strategy, 'enhance', patched_enhance)
    monkeypatch.setattr(enhance_xrf.PluginRegistry, 'load_plugins', lambda: registry)


def test_cancelled_enhancement_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cancellation: CancellationToken
) -> None:
    product_file, fluorescence_file = _write_inputs(tmp_path)
    output_file = tmp_path / 'enhanced.h5'
    _patch_enhance(monkeypatch, cancellation, 'cancel after')

    exit_code = _main(
        monkeypatch,
        '--product',
        product_file,
        '--fluorescence',
        fluorescence_file,
        '--algorithm',
        'two_step',
        '-o',
        output_file,
    )

    assert exit_code is ExitCode.CANCELLED
    assert not output_file.exists()


def _store_inputs(root: Path) -> tuple[UUID, UUID]:
    """A product and a measured fluorescence dataset in a storage root; their uuids."""
    staged = root.parent / 'staged'
    staged.mkdir()
    _, fluorescence_file = _write_inputs(staged)
    layout = StorageLayout(root)
    product = save_product_resource(layout, _product())
    fluorescence = save_fluorescence_resource(layout, load_fluorescence_data(fluorescence_file))
    return product, fluorescence


def _enhanced_resource(root: Path, before: set[str]) -> tuple[Path, FluorescenceManifest]:
    created = {path.name for path in (root / ResourceKind.FLUORESCENCE).iterdir()} - before
    assert len(created) == 1
    folder = root / ResourceKind.FLUORESCENCE / created.pop()
    manifest = load_manifest(
        folder / 'manifest.json',
        expected_kind=ResourceKind.FLUORESCENCE,
        expected_uuid=UUID(folder.name),
    )
    assert isinstance(manifest, FluorescenceManifest)
    return folder, manifest


@pytest.mark.parametrize(
    ('behavior', 'code', 'state', 'result'),
    [
        ('complete', ExitCode.SUCCESS, RunState.SUCCEEDED, 'fluorescence.h5'),
        ('cancel after', ExitCode.CANCELLED, RunState.CANCELLED, 'fluorescence.h5'),
        ('cancel before', ExitCode.CANCELLED, RunState.CANCELLED, None),
        ('fail after', ExitCode.FAILURE, RunState.FAILED, 'checkpoint'),
        ('fail before', ExitCode.FAILURE, RunState.FAILED, None),
    ],
)
def test_stored_inputs_enhance_into_a_new_resource(
    behavior: str,
    code: ExitCode,
    state: RunState,
    result: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    cancellation: CancellationToken,
) -> None:
    root = tmp_path / 'storage'
    product, fluorescence = _store_inputs(root)
    before = {path.name for path in (root / ResourceKind.FLUORESCENCE).iterdir()}
    _patch_enhance(monkeypatch, cancellation, behavior)

    exit_code = _main(
        monkeypatch,
        '--storage-root',
        root,
        '--product',
        product,
        '--fluorescence',
        fluorescence,
        '--algorithm',
        'two_step',
    )

    assert exit_code is code
    folder, manifest = _enhanced_resource(root, before)
    assert capsys.readouterr().out.strip() == folder.name
    assert manifest.derived_from == [
        DerivedFromRef(kind='product', uuid=product),
        DerivedFromRef(kind='fluorescence', uuid=fluorescence),
    ]
    assert manifest.files['options'] == 'enhance_xrf_options.yaml'
    assert (folder / 'enhance_xrf_options.yaml').is_file()
    assert manifest.run is not None
    assert manifest.run.state == state
    assert manifest.run.command == 'ptychodus-enhance-xrf'
    assert manifest.run.reason == ('RuntimeError' if behavior.startswith('fail') else None)

    if result is None:
        assert 'fluorescence' not in manifest.files
    elif result == 'checkpoint':
        assert manifest.files['fluorescence'].startswith('fluorescence.0')
        load_fluorescence_data(folder / manifest.files['fluorescence'])
    else:
        assert manifest.files['fluorescence'] == result
        enhanced = load_fluorescence_data(folder / result)
        assert [emap.name for emap in enhanced] == list(_CHANNELS)


@pytest.mark.parametrize('case', ['mixed', 'with output', 'with file type', 'no root'])
def test_a_bad_stored_request_is_a_usage_error(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / 'storage'
    product, fluorescence = _store_inputs(root)
    monkeypatch.delenv('PTYCHODUS_WEBAPP_STORAGE_ROOT', raising=False)
    stored = ['--storage-root', str(root), '--product', str(product)]
    argv = {
        'mixed': [*stored, '--fluorescence', str(tmp_path / 'staged' / 'xrf.h5')],
        'with output': [*stored, '--fluorescence', str(fluorescence), '-o', str(tmp_path / 'x.h5')],
        'with file type': [*stored, '--fluorescence', str(fluorescence), '--file-type', 'XRF-Maps'],
        'no root': ['--product', str(product), '--fluorescence', str(uuid4())],
    }[case]

    with pytest.raises(SystemExit) as caught:
        _main(monkeypatch, *argv)

    assert caught.value.code == 2
