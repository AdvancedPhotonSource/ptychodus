from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ptychodus_store.storage.layout import LayoutError, StoreLayout


def test_resource_folder_resolution(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    uuid = uuid4()
    folder = layout.resource_folder('product', uuid)
    assert folder == tmp_path.resolve() / 'product' / str(uuid)


def test_manifest_path_resolution(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    uuid = uuid4()
    m = layout.manifest_path('fluorescence', uuid)
    assert m.name == 'manifest.json'
    assert m.parent == tmp_path.resolve() / 'fluorescence' / str(uuid)


def test_parse_manifest_path_valid(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    layout.ensure_kind_dirs()
    uuid = uuid4()
    folder = tmp_path / 'campaign' / str(uuid)
    folder.mkdir()
    m = folder / 'manifest.json'
    m.touch()
    loc = layout.parse_manifest_path(m)
    assert loc.kind == 'campaign'
    assert loc.uuid == uuid
    assert loc.folder == folder.resolve()


def test_parse_manifest_rejects_outside_root(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path / 'inside')
    elsewhere = tmp_path / 'outside' / 'campaign' / str(uuid4()) / 'manifest.json'
    elsewhere.parent.mkdir(parents=True)
    elsewhere.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(elsewhere)


def test_parse_manifest_rejects_non_uuid_folder(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    layout.ensure_kind_dirs()
    bad = tmp_path / 'product' / 'not-a-uuid' / 'manifest.json'
    bad.parent.mkdir()
    bad.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(bad)


def test_parse_manifest_rejects_unknown_kind(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    layout.ensure_kind_dirs()
    bad = tmp_path / 'mystery' / str(uuid4()) / 'manifest.json'
    bad.parent.mkdir(parents=True)
    bad.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(bad)


def test_iter_manifest_paths_orders_campaigns_first(tmp_path: Path) -> None:
    layout = StoreLayout(tmp_path)
    layout.ensure_kind_dirs()

    for kind in ('product', 'diffraction', 'campaign', 'fluorescence'):
        folder = tmp_path / kind / str(uuid4())
        folder.mkdir()
        (folder / 'manifest.json').touch()

    paths = layout.iter_manifest_paths()
    kinds = [p.parent.parent.name for p in paths]
    # Campaign first; then diffraction, product, fluorescence in any order — the
    # spec orders them as campaign, diffraction, product, fluorescence.
    assert kinds[0] == 'campaign'
    assert kinds == ['campaign', 'diffraction', 'product', 'fluorescence']
