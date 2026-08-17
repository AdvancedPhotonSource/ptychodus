"""Options-enumeration endpoint for the visualization API."""

from __future__ import annotations

from fastapi import APIRouter

from ptychodus.api.visualize import (
    ComplexComponent,
    CylindricalColorModel,
    ScalarTransformation,
    cyclic_colormap_names,
    linear_colormap_names,
)

from ptychodus_store.rendering.schemas import OptionsRead

router = APIRouter(prefix='/visualization', tags=['visualization'])


@router.get('/options', response_model=OptionsRead)
async def get_visualization_options() -> OptionsRead:
    """Enumerate the valid choices a client may pass to render endpoints."""
    return OptionsRead(
        colormaps_linear=list(linear_colormap_names()),
        colormaps_cyclic=list(cyclic_colormap_names()),
        transforms=[member.name.lower() for member in ScalarTransformation],
        components=[member.name.lower() for member in ComplexComponent],
        color_models=[member.name.lower() for member in CylindricalColorModel],
    )
