"""The rendered views of stored resources, the bounded operations that draw them.

Each view loads a resource's current file, selects what it shows and builds the image,
synchronously: the HDF5 read and the rendering are blocking work, so each runs in a worker
thread (:class:`~ptychodus.webapp.rendering.offload.RenderRunner`). What a view was asked
to show but the resource lacks, an index out of range say, raises
:class:`~ptychodus.webapp.service.errors.ResourceNotFoundError`. A view takes the file to
read rather than a resource's uuid, because which file is current is the manifest's to say
and is looked up first.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.constants import LengthUnit
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_diffraction_data, load_fluorescence_data, load_product
from ptychodus.api.visualize import VisualizationProduct

from ptychodus.webapp.rendering.params import RenderParams
from ptychodus.webapp.rendering.plot import ProbePositionsTrace, render_probe_positions
from ptychodus.webapp.rendering.render import (
    build_visualization_complex,
    build_visualization_real,
    product_to_png_bytes,
    to_rendered_image,
)
from ptychodus.webapp.rendering.schemas import PlotImage, RenderedImage
from ptychodus.webapp.service.errors import ResourceNotFoundError

__all__ = [
    'DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY',
    'diffraction_aggregate_view',
    'diffraction_pattern_view',
    'fluorescence_element_view',
    'fluorescence_summary_view',
    'object_layer_view',
    'object_view',
    'probe_modes_view',
    'probe_positions_view',
    'probe_view',
    'view_as_png',
    'view_as_rendered_image',
]

DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY = PixelGeometry(
    width_m=LengthUnit.MICROMETER.meters_per_unit,
    height_m=LengthUnit.MICROMETER.meters_per_unit,
)
"""The pixel geometry of a fluorescence map rendered without a product to take one from."""


def diffraction_pattern_view(path: Path, index: int, params: RenderParams) -> VisualizationProduct:
    data = load_diffraction_data(path)
    num_patterns = data.get_patterns_shape()[0]

    if not 0 <= index < num_patterns:
        raise ResourceNotFoundError(f'pattern index {index} out of range [0, {num_patterns})')

    return build_visualization_real(
        data.get_pattern(index), data.get_pixel_geometry(), params, value_label='Counts'
    )


def diffraction_aggregate_view(path: Path, params: RenderParams) -> VisualizationProduct:
    data = load_diffraction_data(path)
    return build_visualization_real(
        data.get_mean_pattern(), data.get_pixel_geometry(), params, value_label='Mean Counts'
    )


def probe_view(path: Path, incoherent: int, params: RenderParams) -> VisualizationProduct:
    product = load_product(path)
    probe = product.probes.get_mean_probe()

    if not 0 <= incoherent < probe.num_incoherent_modes:
        raise ResourceNotFoundError(
            f'incoherent mode {incoherent} out of range [0, {probe.num_incoherent_modes})'
        )

    return build_visualization_complex(
        probe.get_incoherent_mode(incoherent), product.probes.get_pixel_geometry(), params
    )


def probe_modes_view(path: Path, params: RenderParams) -> VisualizationProduct:
    product = load_product(path)
    probe = product.probes.get_mean_probe()
    return build_visualization_complex(
        probe.get_incoherent_modes_flattened(), product.probes.get_pixel_geometry(), params
    )


def object_view(path: Path, params: RenderParams) -> VisualizationProduct:
    """The object as the product of its layers, the transmission of the whole stack."""
    product = load_product(path)
    return build_visualization_complex(
        product.object_.get_layers_flattened(), product.object_.get_pixel_geometry(), params
    )


def object_layer_view(path: Path, layer: int, params: RenderParams) -> VisualizationProduct:
    product = load_product(path)
    num_layers = product.object_.num_layers

    if not 0 <= layer < num_layers:
        raise ResourceNotFoundError(f'object layer {layer} out of range [0, {num_layers})')

    return build_visualization_complex(
        product.object_.get_layer(layer), product.object_.get_pixel_geometry(), params
    )


def _fluorescence_pixel_geometry(product_path: Path | None) -> PixelGeometry:
    if product_path is None:
        return DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY

    return load_product(product_path).object_.get_pixel_geometry()


def fluorescence_summary_view(
    path: Path, product_path: Path | None, params: RenderParams
) -> VisualizationProduct:
    """The elementwise sum of every element map, on a paired product's pixel geometry."""
    dataset = load_fluorescence_data(path)

    if not len(dataset):
        raise ResourceNotFoundError('fluorescence dataset has no element maps')

    summary = numpy.sum([emap.counts_per_second for emap in dataset], axis=0)
    return build_visualization_real(
        summary, _fluorescence_pixel_geometry(product_path), params, value_label='counts/s'
    )


def fluorescence_element_view(
    path: Path, name: str, product_path: Path | None, params: RenderParams
) -> VisualizationProduct:
    dataset = load_fluorescence_data(path)
    matches = [emap for emap in dataset if emap.name == name]

    if not matches:
        raise ResourceNotFoundError(
            f'element {name!r} not found', available=[emap.name for emap in dataset]
        )

    emap = matches[0]
    return build_visualization_real(
        emap.counts_per_second,
        _fluorescence_pixel_geometry(product_path),
        params,
        value_label=f'{emap.name} counts/s',
    )


def probe_positions_view(
    entries: Sequence[tuple[Path, str]], *, connect_path: bool, width_px: int, height_px: int
) -> PlotImage:
    """The probe positions of the products in `entries`, each a product file and its label."""
    traces: list[ProbePositionsTrace] = []

    for path, label in entries:
        product = load_product(path)
        positions = product.probe_positions

        if len(positions) == 0:
            raise ResourceNotFoundError(f'product {label} has no probe positions')

        traces.append(
            ProbePositionsTrace(
                label=product.metadata.name or label,
                x_m=[p.x_m for p in positions],
                y_m=[p.y_m for p in positions],
            )
        )

    return render_probe_positions(
        traces, connect_path=connect_path, width_px=width_px, height_px=height_px
    )


def view_as_rendered_image(
    view: Callable[..., VisualizationProduct],
    arguments: Sequence[Any],
    params: RenderParams,
    is_complex: bool,
) -> RenderedImage:
    """Run `view` on `arguments` and `params`, and encode it as the REST response.

    Positional throughout, so a worker thread can be handed it with its arguments as is.
    """
    return to_rendered_image(view(*arguments, params), params, is_complex=is_complex)


def view_as_png(
    view: Callable[..., VisualizationProduct], arguments: Sequence[Any], params: RenderParams
) -> bytes:
    """Run `view` on `arguments` and `params`, and encode it as PNG bytes."""
    return product_to_png_bytes(view(*arguments, params))
