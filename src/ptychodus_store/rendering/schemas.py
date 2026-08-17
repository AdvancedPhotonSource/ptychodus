"""Pydantic response models for the visualization endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RenderedImage(BaseModel):
    """A rendered image plus the metadata a front-end needs to draw a colorbar."""

    png_base64: str = Field(description='Base64-encoded PNG bytes.')
    mime_type: Literal['image/png'] = 'image/png'
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


class OptionsRead(BaseModel):
    """Enumeration of valid choices a client may pass to visualization endpoints."""

    colormaps_linear: list[str]
    colormaps_cyclic: list[str]
    transforms: list[str]
    components: list[str]
    color_models: list[str]
