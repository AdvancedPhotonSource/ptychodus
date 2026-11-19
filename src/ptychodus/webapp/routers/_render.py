"""What the REST routes share in reaching a resource's file and rendering it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from fastapi import HTTPException

from ptychodus.api.storage import (
    LayoutError,
    ManifestLoadError,
    StorageLayout,
    resolve_resource_file,
)

from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.rendering.render import RenderParamsError
from ptychodus.webapp.rendering.views import RenderLookupError

_T = TypeVar('_T')


def resource_file_or_404(layout: StorageLayout, kind: str, uuid: UUID) -> Path:
    """The file holding the resource's current data, as its manifest names it, or a 404.

    A running resource names its newest checkpoint, and one with no result yet has none.
    """
    try:
        path = resolve_resource_file(layout, uuid, kind)
    except (LayoutError, ManifestLoadError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not path.is_file():
        raise HTTPException(status_code=404, detail=f'{path.name} not present on disk')

    return path


async def render_or_http_error(runner: RenderRunner, func: Callable[..., _T], /, *args: Any) -> _T:
    """Run a render through `runner`, mapping its errors to HTTP: lookups 404, params 400."""
    try:
        return await runner.run(func, *args)
    except RenderLookupError as exc:
        detail: Any = exc.message

        if exc.available:
            detail = {'error': exc.message, 'available': exc.available}

        raise HTTPException(status_code=404, detail=detail) from exc
    except RenderParamsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
