"""Shared query-parameter dependency for the visualization endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, TypeVar

from fastapi import Depends, HTTPException, Query

from ptychodus.api.visualization import (
    ComplexComponent,
    CylindricalColorModel,
    ScalarTransformation,
    cyclic_colormap_names,
    linear_colormap_names,
)

E = TypeVar('E', bound=Enum)


class InvalidRenderParamError(ValueError):
    """Raised when a raw render-param string cannot be coerced to its enum / colormap."""

    def __init__(self, message: str, detail: dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True)
class RenderParams:
    """Parsed and validated visualization parameters shared by every render endpoint."""

    colormap: str
    transform: ScalarTransformation
    component: ComplexComponent | None
    color_model: CylindricalColorModel | None
    value_min: float | None
    value_max: float | None
    clip: bool


def _coerce_enum(name: str, enum_cls: type[E], arg_name: str) -> E:
    normalized = name.strip().upper().replace('-', '_')
    try:
        return enum_cls[normalized]
    except KeyError as exc:
        valid = [member.name.lower() for member in enum_cls]
        raise InvalidRenderParamError(
            f'invalid {arg_name}: {name!r}',
            detail={'error': f'invalid {arg_name}: {name!r}', 'valid': valid},
        ) from exc


def _coerce_colormap(name: str) -> str:
    valid_linear = set(linear_colormap_names())
    valid_cyclic = set(cyclic_colormap_names())
    if name in valid_linear or name in valid_cyclic:
        return name
    raise InvalidRenderParamError(
        f'invalid colormap: {name!r}',
        detail={
            'error': f'invalid colormap: {name!r}',
            'valid_linear': sorted(valid_linear),
            'valid_cyclic': sorted(valid_cyclic),
        },
    )


def coerce_render_params(
    colormap: str,
    transform: str,
    component: str | None,
    color_model: str | None,
    value_min: float | None,
    value_max: float | None,
    clip: bool,
) -> RenderParams:
    """Coerce raw values into a :class:`RenderParams`; raises :class:`InvalidRenderParamError`."""
    return RenderParams(
        colormap=_coerce_colormap(colormap),
        transform=_coerce_enum(transform, ScalarTransformation, 'transform'),
        component=_coerce_enum(component, ComplexComponent, 'component')
        if component is not None
        else None,
        color_model=_coerce_enum(color_model, CylindricalColorModel, 'color_model')
        if color_model is not None
        else None,
        value_min=value_min,
        value_max=value_max,
        clip=clip,
    )


def render_params_dep(
    colormap: Annotated[
        str, Query(description='Colormap name from /visualization/options.')
    ] = 'gray',
    transform: Annotated[str, Query(description='Scalar transformation enum name.')] = 'identity',
    component: Annotated[
        str | None, Query(description='ComplexComponent enum name; for complex arrays only.')
    ] = None,
    color_model: Annotated[
        str | None,
        Query(description='CylindricalColorModel enum name; for complex arrays only.'),
    ] = None,
    value_min: Annotated[float | None, Query(description='Color-axis lower bound.')] = None,
    value_max: Annotated[float | None, Query(description='Color-axis upper bound.')] = None,
    clip: Annotated[
        bool, Query(description='Clip out-of-range values to the axis bounds.')
    ] = False,
) -> RenderParams:
    try:
        return coerce_render_params(
            colormap=colormap,
            transform=transform,
            component=component,
            color_model=color_model,
            value_min=value_min,
            value_max=value_max,
            clip=clip,
        )
    except InvalidRenderParamError as exc:
        raise HTTPException(status_code=400, detail=exc.detail) from exc


RenderParamsDep = Annotated[RenderParams, Depends(render_params_dep)]
