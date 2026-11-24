"""The service's refusals as HTTP responses, shaped as an `HTTPException` would shape them."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ptychodus.webapp.service.errors import InvalidRequestError, ResourceNotFoundError

__all__ = ['add_service_error_handlers']


async def _not_found(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ResourceNotFoundError)
    detail: Any = exc.message

    if exc.available:
        detail = {'error': exc.message, 'available': exc.available}

    return JSONResponse(status_code=404, content={'detail': detail})


async def _invalid(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, InvalidRequestError)
    detail: Any = exc.detail if exc.detail is not None else exc.message
    return JSONResponse(status_code=400, content={'detail': detail})


def add_service_error_handlers(app: FastAPI) -> None:
    """Answer an unknown resource with a 404 and a request it cannot serve with a 400."""
    app.add_exception_handler(ResourceNotFoundError, _not_found)
    app.add_exception_handler(InvalidRequestError, _invalid)
