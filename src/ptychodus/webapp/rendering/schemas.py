"""Pydantic response models for the visualization endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RenderedImageInfo(BaseModel):
    """What a rendered image's scales are, for a reader that has the image itself apart.

    Everything :class:`RenderedImage` carries but the PNG and the legend's color stops.
    """

    value_label: str = Field(description='LaTeX-decorated label from the underlying transform.')
    color_value_min: float = Field(
        description='Lower bound of the color axis, in transformed units.'
    )
    color_value_max: float = Field(
        description='Upper bound of the color axis, in transformed units.'
    )
    pixel_width_m: float = Field(description='Physical pixel width in meters.')
    pixel_height_m: float = Field(description='Physical pixel height in meters.')
    shape_h_px: int = Field(description='Rendered image height in pixels.')
    shape_w_px: int = Field(description='Rendered image width in pixels.')
    legend_cyclic: bool = Field(
        default=False, description='True when the legend wraps around, as phase does.'
    )


class RenderedImage(RenderedImageInfo):
    """A rendered image plus the metadata a front-end needs to draw a colorbar."""

    png_base64: str = Field(description='Base64-encoded PNG bytes.')
    mime_type: Literal['image/png'] = 'image/png'
    legend_rgba: list[list[float]] = Field(
        default=[],
        description=(
            'Color legend as RGBA stops in [0, 1], evenly spaced from color_value_min '
            'to color_value_max, or once around the hue wheel when legend_cyclic.'
        ),
    )


class PlotImageInfo(BaseModel):
    """A rendered plot's size, for a reader that has the image itself apart.

    Distinct from :class:`RenderedImageInfo`, which describes a colormapped array: a plot
    carries its own axes, so it has neither a color axis nor a physical pixel size.
    Reporting one would invite a caption to read it as the sampling of the data.
    """

    shape_h_px: int = Field(description='Rendered image height in pixels.')
    shape_w_px: int = Field(description='Rendered image width in pixels.')


class PlotImage(PlotImageInfo):
    """A rendered plot."""

    png_base64: str = Field(description='Base64-encoded PNG bytes.')
    mime_type: Literal['image/png'] = 'image/png'


class OptionsRead(BaseModel):
    """Enumeration of valid choices a client may pass to visualization endpoints.

    Each list holds the values the render endpoints accept. The label maps carry the
    names the desktop shows for the same choices, keyed by those values.
    """

    colormaps_linear: list[str]
    colormaps_cyclic: list[str]
    transforms: list[str]
    components: list[str]
    color_models: list[str]
    transform_labels: dict[str, str] = {}
    component_labels: dict[str, str] = {}
    color_model_labels: dict[str, str] = {}
