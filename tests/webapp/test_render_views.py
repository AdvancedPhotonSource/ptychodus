"""The render views, called as the worker thread calls them: positionally, on files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from ptychodus.webapp.rendering import coerce_render_params
from ptychodus.webapp.rendering.views import (
    RenderLookupError,
    diffraction_pattern_view,
    fluorescence_element_view,
    fluorescence_summary_view,
    object_layer_view,
    probe_view,
    scan_paths_view,
    view_as_png,
    view_as_rendered_image,
)

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_REAL = coerce_render_params('gray', 'identity', None, None, None, None, False)
_COMPLEX = coerce_render_params('gray', 'identity', 'amplitude', None, None, None, False)


def _file(root: Path, kind: str, uuid: UUID) -> Path:
    return root / kind / str(uuid) / f'{kind}.h5'


def test_a_view_renders_as_png_and_as_the_rest_response(
    tmp_storage_root: Path, seed_product: Callable[..., UUID]
) -> None:
    path = _file(tmp_storage_root, 'product', seed_product())

    assert view_as_png(object_layer_view, (path, 0), _COMPLEX).startswith(_PNG_SIGNATURE)

    image = view_as_rendered_image(object_layer_view, (path, 0), _COMPLEX, True)
    assert (image.shape_h_px, image.shape_w_px) == (16, 16)


@pytest.mark.parametrize(
    ('view', 'kind', 'argument', 'message'),
    [
        (diffraction_pattern_view, 'diffraction', 4, 'pattern index 4 out of range [0, 4)'),
        (probe_view, 'product', 1, 'incoherent mode 1 out of range [0, 1)'),
        (object_layer_view, 'product', -1, 'object layer -1 out of range [0, 1)'),
    ],
)
def test_an_index_outside_the_resource_is_a_lookup_error(
    tmp_storage_root: Path,
    seed_diffraction: Callable[..., UUID],
    seed_product: Callable[..., UUID],
    view: Callable[..., object],
    kind: str,
    argument: int,
    message: str,
) -> None:
    uuid = seed_diffraction() if kind == 'diffraction' else seed_product()
    params = _REAL if kind == 'diffraction' else _COMPLEX

    with pytest.raises(RenderLookupError) as caught:
        view(_file(tmp_storage_root, kind, uuid), argument, params)

    assert caught.value.message == message
    assert caught.value.available == []


def test_a_missing_element_names_the_ones_there_are(
    tmp_storage_root: Path, seed_fluorescence: Callable[..., UUID]
) -> None:
    path = _file(tmp_storage_root, 'fluorescence', seed_fluorescence(elements=['Fe', 'Cu']))

    with pytest.raises(RenderLookupError) as caught:
        fluorescence_element_view(path, 'Au', None, _REAL)

    assert caught.value.available == ['Fe', 'Cu']


def test_a_fluorescence_map_takes_the_paired_products_pixel_size(
    tmp_storage_root: Path,
    seed_fluorescence: Callable[..., UUID],
    seed_product: Callable[..., UUID],
) -> None:
    path = _file(tmp_storage_root, 'fluorescence', seed_fluorescence())
    product_path = _file(tmp_storage_root, 'product', seed_product())

    unpaired = fluorescence_summary_view(path, None, _REAL)
    paired = fluorescence_summary_view(path, product_path, _REAL)

    assert unpaired.get_pixel_geometry().width_m == pytest.approx(1e-6)
    assert paired.get_pixel_geometry().width_m == pytest.approx(1e-9)


def test_scan_paths_render_one_plot_for_every_product(
    tmp_storage_root: Path, seed_product: Callable[..., UUID]
) -> None:
    entries = [(_file(tmp_storage_root, 'product', seed_product()), label) for label in ('a', 'b')]

    plot = scan_paths_view(entries, connect_path=True, width_px=320, height_px=240)

    assert plot.png_base64
