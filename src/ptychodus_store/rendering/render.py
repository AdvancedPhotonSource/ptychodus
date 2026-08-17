"""Adapters from `ptychodus.api.visualize` to the store's HTTP and MCP response shapes."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy
from fastapi import HTTPException
from PIL import Image

from ptychodus.api.typing import ComplexArrayType, NumberArrayType, RealArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualize import (
    ComplexComponent,
    VisualizationProduct,
    visualize_complex_component,
    visualize_complex_values,
    visualize_real_values,
)

from ptychodus_store.rendering.params import RenderParams
from ptychodus_store.rendering.schemas import RenderedImage


class RenderParamsError(ValueError):
    """Raised when the RenderParams are inconsistent with the array being rendered."""


def build_visualization_real(
    values: NumberArrayType,
    pixel_geometry: PixelGeometry,
    params: RenderParams,
    value_label: str,
) -> VisualizationProduct:
    """Build a VisualizationProduct from a real-valued 2D array."""
    if params.component is not None or params.color_model is not None:
        raise RenderParamsError(
            'component and color_model are only valid for complex-valued arrays.'
        )

    values_real: RealArrayType = numpy.asarray(values, dtype=numpy.float64)
    return visualize_real_values(
        value_label=value_label,
        values=values_real,
        pixel_geometry=pixel_geometry,
        colormap=params.colormap,
        transform=params.transform,
        value_min=params.value_min,
        value_max=params.value_max,
        clip=params.clip,
    )


def build_visualization_complex(
    values: ComplexArrayType,
    pixel_geometry: PixelGeometry,
    params: RenderParams,
) -> VisualizationProduct:
    """Build a VisualizationProduct from a complex-valued 2D array.

    Dispatches on the provided render params: `color_model` → cylindrical encoding;
    `component` → single scalar component; neither → defaults to amplitude.
    """
    if params.component is not None and params.color_model is not None:
        raise RenderParamsError('component and color_model are mutually exclusive.')

    if params.color_model is not None:
        return visualize_complex_values(
            values=values,
            pixel_geometry=pixel_geometry,
            model=params.color_model,
            amplitude_transform=params.transform,
            value_min=params.value_min,
            value_max=params.value_max,
            clip=params.clip,
        )

    component = params.component or ComplexComponent.AMPLITUDE
    return visualize_complex_component(
        values=values,
        pixel_geometry=pixel_geometry,
        component=component,
        colormap=params.colormap,
        transform=params.transform,
        value_min=params.value_min,
        value_max=params.value_max,
        clip=params.clip,
    )


def product_to_png_bytes(vp: VisualizationProduct) -> bytes:
    """Encode a VisualizationProduct's RGBA image as PNG bytes."""
    rgba_uint8 = numpy.clip(vp.get_image_rgba() * 255.0, 0.0, 255.0).astype(numpy.uint8)
    image = Image.fromarray(rgba_uint8, mode='RGBA')
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def _product_to_response(vp: VisualizationProduct) -> RenderedImage:
    png_bytes = product_to_png_bytes(vp)
    pixel_geometry = vp.get_pixel_geometry()
    color_range = vp.get_color_value_range()
    rgba = vp.get_image_rgba()
    return RenderedImage(
        png_base64=base64.b64encode(png_bytes).decode('ascii'),
        value_label=vp.get_value_label(),
        color_value_min=float(color_range.lower),
        color_value_max=float(color_range.upper),
        pixel_width_m=float(pixel_geometry.width_m),
        pixel_height_m=float(pixel_geometry.height_m),
        shape_h_px=int(rgba.shape[0]),
        shape_w_px=int(rgba.shape[1]),
    )


def render_real(
    values: NumberArrayType,
    pixel_geometry: PixelGeometry,
    params: RenderParams,
    value_label: str,
) -> RenderedImage:
    """Render a real-valued (or integer) 2D array. Rejects component / color_model (400)."""
    try:
        vp = build_visualization_real(values, pixel_geometry, params, value_label)
    except RenderParamsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _product_to_response(vp)


def render_complex(
    values: ComplexArrayType,
    pixel_geometry: PixelGeometry,
    params: RenderParams,
) -> RenderedImage:
    """Render a complex-valued 2D array. Rejects component+color_model together (400)."""
    try:
        vp = build_visualization_complex(values, pixel_geometry, params)
    except RenderParamsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _product_to_response(vp)
