"""Options-enumeration endpoint for the visualization API."""

from __future__ import annotations

from fastapi import APIRouter

from ptychodus.webapp.rendering import visualization_options
from ptychodus.webapp.rendering.schemas import OptionsRead

router = APIRouter(prefix='/visualization', tags=['visualization'])


@router.get('/options', response_model=OptionsRead)
async def get_visualization_options() -> OptionsRead:
    """Enumerate the valid choices a client may pass to render endpoints."""
    return visualization_options()
