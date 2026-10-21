"""Build live, observable settings parameters from a declared options model.

A reconstructor declares what it accepts as plain pydantic. That declaration is a
description: it has no identity, nothing observes it, and nothing persists it.
This module turns one into the live ``Parameter`` handles the settings layer and
the Qt panels bind to, under the flat key layout :mod:`.flatten` computes.

**Transitional, with the flattener.** When the Qt panels retire and the nested
model goes straight to the web UI, the declaration stops needing a flat live
mirror and this module goes with it.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
import copy
from types import UnionType
from typing import Any, Union, get_args, get_origin
import logging

from ptychodus.api.parameters import Parameter, ParameterGroup

from .flatten import FlatParameter, FlatteningPolicy, flatten_options

__all__ = [
    'UnsupportedParameterError',
    'populate_parameter_group',
    'apply_options',
    'restore_options',
]

logger = logging.getLogger(__name__)


class UnsupportedParameterError(TypeError):
    """Raised for a declared field the settings layer cannot represent."""


def _unwrap_optional(annotation: Any) -> Any:
    """Return the single non-None member of an optional annotation, else the input.

    Declarations routinely spell an unset bound as ``int | None``. The live
    parameter stores the inner type and treats absence as the default, so the
    wrapper carries no information worth preserving.
    """
    if get_origin(annotation) not in (Union, UnionType):
        return annotation

    members = [arg for arg in get_args(annotation) if arg is not type(None)]
    return members[0] if len(members) == 1 else annotation


def _create(group: ParameterGroup, flat: FlatParameter) -> Parameter[Any]:
    annotation = _unwrap_optional(flat.annotation)
    key = flat.ini_key
    default = flat.default

    # Enums persist as a string -- by member name for pty-chi, by value for the
    # models ptychodus owns -- so the live parameter is a string either way, and
    # the declared type is what restores meaning on the way back out.
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return group.create_string_parameter(key, flat.spelling.render(default))

    # bool is a subclass of int, so it has to be tested first.
    if annotation is bool:
        return group.create_boolean_parameter(key, bool(default))

    if annotation is int:
        return group.create_integer_parameter(
            key, int(default), minimum=flat.minimum, maximum=flat.maximum
        )

    if annotation is float:
        return group.create_real_parameter(
            key, float(default), minimum=flat.minimum, maximum=flat.maximum
        )

    if annotation is str:
        return group.create_string_parameter(key, str(default))

    if isinstance(annotation, type) and issubclass(annotation, Path):
        return group.create_path_parameter(key, Path(default))

    raise UnsupportedParameterError(
        f'Cannot represent "{flat.path}" ({annotation}) as a settings parameter.'
    )


def populate_parameter_group(
    group: ParameterGroup, options: Any, policy: FlatteningPolicy
) -> Mapping[str, Parameter[Any]]:
    """Add the parameters *options* declares to *group*, returning them.

    Takes an existing group rather than creating one, because a settings group is
    not always fed by a single declared object: pty-chi's per-algorithm groups
    draw their keys from four different branches of one declaration, and each
    contributes to the same group.

    Keyed by persisted name, so a caller holding the flattened description can
    pair each declared path with the handle that now backs it.

    Idempotent: a key the group already carries is reused, not recreated.
    """
    existing = group.parameters()
    parameters: dict[str, Parameter[Any]] = {}

    for flat in flatten_options(options, policy):
        # Reused rather than recreated, so that two reconstructors declaring
        # the same options share one set of live values. A backend's modes are
        # configured by one settings section, and a file holds one copy of it.
        parameter = existing.get(flat.ini_key)
        parameters[flat.ini_key] = _create(group, flat) if parameter is None else parameter

    return parameters


def _assign(root: Any, path: str, value: Any) -> None:
    *ancestors, leaf = path.split('.')
    owner = root

    for name in ancestors:
        owner = getattr(owner, name)

    setattr(owner, leaf, value)


def _accepted(flat: FlatParameter, value: Any) -> Any:
    """Return *value* if the declaration accepts it, else the default.

    A settings file is hand-edited and outlives the version that wrote it, so
    a key can hold something the declaration does not allow. Warning and
    falling back keeps a run going with a value that is at least valid, and
    says which key was ignored -- where passing it through would hand a
    backend a word it has no case for.

    Only a key that holds one of the choices is policed. A key naming a
    *subset* of them carries the same list and holds a number instead, which
    is never one of the names, so checking it would reject every value it can
    ever have.
    """
    if not flat.choices or not isinstance(value, str) or value in flat.choices:
        return value

    logger.warning(
        f'Ignoring "{flat.ini_key}" = "{value}": not one of '
        f'{", ".join(flat.choices)}. Using "{flat.default}".'
    )
    return flat.default


def _restore_value(flat: FlatParameter, value: Any) -> Any:
    """Turn a live parameter's value back into what the declaration expects.

    Only enums need turning back: they persist as a string, and which string
    depends on how the key was written.
    """
    annotation = _unwrap_optional(flat.annotation)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return flat.spelling.parse(annotation, value)
        except (KeyError, ValueError):
            members = ', '.join(flat.spelling.members(annotation))
            # The fallback is spelled the way the members are listed, so the
            # message does not appear to name a value it just rejected.
            logger.warning(
                f'Ignoring "{flat.ini_key}" = "{value}": not one of {members}. '
                f'Using "{flat.spelling.render(flat.default)}".'
            )
            return flat.default

    return _accepted(flat, value)


def apply_options(
    target: Any, parameters: Mapping[str, Parameter[Any]], policy: FlatteningPolicy
) -> None:
    """Write the values held in *parameters* into *target*, in place.

    Separate from :func:`restore_options` because one declaration can back
    several settings groups, and they are applied to parts of a single object
    that must be copied only once.
    """
    adapted: dict[str, dict[str, Any]] = {}

    for flat in flatten_options(target, policy):
        try:
            parameter = parameters[flat.ini_key]
        except KeyError:
            continue

        if flat.structural_role:
            # Collected rather than assigned: the declared value is a function of
            # every key the adapter owns, so none can be written on its own.
            adapted.setdefault(flat.path, {})[flat.structural_role] = _accepted(
                flat, parameter.get_value()
            )
            continue

        _assign(target, flat.path, _restore_value(flat, parameter.get_value()))

    for path, values in adapted.items():
        _assign(target, path, policy.structural[path].decode(values))


def restore_options(
    defaults: Any, parameters: Mapping[str, Parameter[Any]], policy: FlatteningPolicy
) -> Any:
    """Return a copy of *defaults* carrying the values now held in *parameters*.

    The inverse of :func:`populate_parameter_group`. The declared object is what a
    reconstructor is handed, so edits made against the flat live parameters have
    to be written back into its nested shape before the call.

    A copy, because the defaults are the store's record of where each field
    started and must survive being read repeatedly.
    """
    restored = copy.deepcopy(defaults)
    apply_options(restored, parameters, policy)
    return restored
