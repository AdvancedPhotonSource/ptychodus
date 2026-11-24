"""Options documents: a library's pydantic options as YAML, named beside what they configure.

A document names what it configures beside the values it gives that thing's fields::

    algorithm: vspi
    options:
      field: value
      section:
        nested_field: value

:func:`dump_options_document` writes one in canonical form -- only the fields that are set,
so a library that resolves the others at runtime still can -- and
:func:`read_options_document` and :func:`apply_options_document` take one back. Fields a
document omits keep their defaults, at any depth.

**Set means set by the defaults or changed by the document.** A library may read which
fields are explicitly set -- ``model_dump(exclude_unset=True)`` -- and resolve the others at
runtime. A document written with every field, as a commented template is, names them all;
were naming a field enough to set it, a round trip would silently end those runtime
resolutions. A document that leaves a value as it was leaves its set-ness as it was too.

**A frozen field is not a document's to change.** A document giving a field declared
``Field(frozen=True)`` any value is refused.

:func:`expect_options` narrows options typed as the base model to the type a plugin declared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar
import sys

from pydantic import BaseModel, ValidationError
import yaml

__all__ = [
    'OptionsError',
    'apply_options_document',
    'dump_options_document',
    'expect_options',
    'parse_options_document',
    'read_options_document',
]


_OptionsT = TypeVar('_OptionsT', bound=BaseModel)


class OptionsError(ValueError):
    """An options document that cannot be used, reported as a usage error."""


def expect_options(options: BaseModel, kind: type[_OptionsT]) -> _OptionsT:
    """Return *options* as *kind*, refusing options declared for another plugin.

    Every options-taking interface types ``options`` as the base model, so this is what ties
    an implementation to the options it declared; a mismatch fails here rather than as an
    attribute error several frames later.

    Raises:
        TypeError: if *options* is not an instance of *kind*.
    """
    if not isinstance(options, kind):
        raise TypeError(f'Expected {kind.__name__}, got {type(options).__name__}.')

    return options


def _frozen_fields(model: BaseModel) -> dict[str, Any]:
    """A ``model_dump`` exclusion naming every frozen field of `model`, at any depth."""
    excluded: dict[str, Any] = {}

    for name, field in type(model).model_fields.items():
        value = getattr(model, name)

        if field.frozen:
            excluded[name] = True
        elif isinstance(value, BaseModel):
            nested = _frozen_fields(value)

            if nested:
                excluded[name] = nested

    return excluded


def dump_options_document(selectors: Mapping[str, str], options: BaseModel) -> str:
    """`options` as a YAML document headed by `selectors`, holding only the fields set.

    The canonical form a run leaves behind and a job carries:
    ``options.model_dump(mode='json', exclude_unset=True)`` under ``options``, less the
    frozen fields a document may not give, so :func:`apply_options_document` onto the same
    defaults reproduces `options`, set-ness included.
    """
    fields = options.model_dump(mode='json', exclude_unset=True, exclude=_frozen_fields(options))
    document = {**selectors, 'options': fields}
    return yaml.safe_dump(document, sort_keys=False)


def read_options_document(
    options_file: Path, selector_keys: Sequence[str]
) -> tuple[dict[str, str], Mapping[str, Any]]:
    """The selectors an options document gives, and the field values it gives.

    `options_file` named ``-`` reads stdin, the conventional spelling once a bare ``-`` has
    been through ``type=Path``.

    Raises:
        OptionsError: if the document is not a mapping holding a string for each of
            `selector_keys` and, optionally, a mapping of ``options``.
        OSError: if `options_file` cannot be read.
    """
    text = sys.stdin.read() if options_file == Path('-') else options_file.read_text()
    return parse_options_document(text, selector_keys, source=str(options_file))


def parse_options_document(
    text: str, selector_keys: Sequence[str], *, source: str
) -> tuple[dict[str, str], Mapping[str, Any]]:
    """As :func:`read_options_document`, from the document's text; `source` names it in errors.

    Raises:
        OptionsError: as :func:`read_options_document` does.
    """
    options_file = source
    allowed_keys = (*selector_keys, 'options')

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OptionsError(f'{options_file} is not YAML: {exc}') from exc

    if not isinstance(document, Mapping):
        raise OptionsError(
            f'{options_file} must hold a mapping with '
            + ', '.join(f'"{key}"' for key in allowed_keys)
            + '.'
        )

    unknown_keys = sorted(str(key) for key in document if key not in allowed_keys)

    if unknown_keys:
        raise OptionsError(
            f'{options_file} has unexpected top-level keys {unknown_keys}; '
            f'only {list(allowed_keys)} are read.'
        )

    selectors: dict[str, str] = {}

    for key in selector_keys:
        value = document.get(key)

        if not isinstance(value, str):
            raise OptionsError(f'{options_file} must give "{key}" as a string.')

        selectors[key] = value

    fields = document.get('options')

    if fields is None:
        fields = {}
    elif not isinstance(fields, Mapping):
        raise OptionsError(f'"options" in {options_file} must be a mapping of field values.')

    return selectors, fields


def _find_refused(
    model: BaseModel, fields: Mapping[str, Any], prefix: str
) -> tuple[list[str], list[str]]:
    """Dotted paths in `fields` that `model` does not declare, and those it declares frozen."""
    declared = type(model).model_fields
    undeclared: list[str] = []
    frozen: list[str] = []

    for name, value in fields.items():
        path = f'{prefix}{name}'
        field = declared.get(str(name))

        if field is None:
            undeclared.append(path)
        elif field.frozen:
            frozen.append(path)
        else:
            current = getattr(model, name)

            if isinstance(current, BaseModel) and isinstance(value, Mapping):
                nested_undeclared, nested_frozen = _find_refused(current, value, f'{path}.')
                undeclared.extend(nested_undeclared)
                frozen.extend(nested_frozen)

    return undeclared, frozen


def _merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for name, value in overrides.items():
        current = merged.get(name)

        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[name] = _merge(current, value)
        else:
            merged[name] = value

    return merged


def _set_or_changed(candidate: BaseModel, defaults: BaseModel) -> dict[str, Any]:
    """The fields of `candidate` that `defaults` set, or that differ from `defaults`.

    A nested model `defaults` left unset stays unset unless something in it changed, even
    where its own defaults set fields inside it.
    """
    kept: dict[str, Any] = {}

    for name in type(defaults).model_fields:
        value = getattr(candidate, name)
        default = getattr(defaults, name)
        is_set = name in defaults.model_fields_set

        if isinstance(value, BaseModel) and type(value) is type(default):
            if is_set or value != default:
                kept[name] = _set_or_changed(value, default)
        elif is_set or value != default:
            kept[name] = value

    return kept


def apply_options_document(
    defaults: BaseModel, fields: Mapping[str, Any], *, subject: str
) -> BaseModel:
    """`defaults` with `fields` merged in at any depth, validated against its declaration.

    A field is set in the result if `defaults` set it or `fields` changes its value, so
    applying the document :func:`format_options_document` wrote reproduces `defaults`
    exactly. `subject` names what the options configure, in error messages.

    Raises:
        OptionsError: if a field is not declared or is frozen, at any depth, or a value is
            one the declaration rejects.
    """
    # Checked here because pydantic ignores names a model does not declare, or keeps them
    # as extras, either of which would let a misspelled field run silently at its default.
    undeclared, frozen = _find_refused(defaults, fields, '')

    if undeclared:
        raise OptionsError(
            f'{subject} does not declare {undeclared}; --print-options lists what it does.'
        )

    if frozen:
        raise OptionsError(f'{subject} fixes {frozen}; they cannot be set by a document.')

    try:
        candidate = type(defaults).model_validate(_merge(defaults.model_dump(), fields))
        return type(defaults).model_validate(_set_or_changed(candidate, defaults))
    except ValidationError as exc:
        raise OptionsError(f'Invalid options for {subject}: {exc}') from exc
