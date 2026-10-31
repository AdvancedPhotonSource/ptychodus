"""Options documents: a backend's pydantic options as YAML a person edits and passes back.

A document names what it configures beside the values it gives that thing's fields::

    algorithm: vspi
    options:
      field: value
      section:
        nested_field: value

:func:`format_options_document` writes one at the defaults, each field's description above
it as a comment, and :func:`read_options_document` and :func:`apply_options_document` take
one back. Fields a document omits keep their defaults, at any depth.

**Set means set by the defaults or changed by the document.** A backend may read which
fields are explicitly set -- ``model_dump(exclude_unset=True)`` -- and resolve the others at
runtime. Every field is printed, so a document passed back unchanged names them all; were
naming a field enough to set it, a round trip would silently end those runtime resolutions.
A document that leaves a value as it was leaves its set-ness as it was too.

**A frozen field is not a document's to change.** A field declared ``Field(frozen=True)``
is printed as a comment saying so, and a document giving it any value is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import sys

from pydantic import BaseModel, ValidationError
import yaml

__all__ = [
    'OptionsError',
    'apply_options_document',
    'format_options_document',
    'read_options_document',
]


class OptionsError(ValueError):
    """An options document that cannot be used, reported as a usage error."""


def _format_fields(model: BaseModel, values: Mapping[str, Any], depth: int) -> list[str]:
    indent = '  ' * depth
    lines: list[str] = []

    for name, field in type(model).model_fields.items():
        if field.description:
            lines.append(f'{indent}# {field.description}')

        value = getattr(model, name)

        if field.frozen:
            lines.append(f'{indent}# {name} is fixed by the backend and cannot be set here.')
        elif isinstance(value, BaseModel) and type(value).model_fields:
            lines.append(f'{indent}{name}:')
            lines.extend(_format_fields(value, values[name], depth + 1))
        else:
            entry = yaml.safe_dump({name: values[name]}, default_flow_style=False, sort_keys=False)
            lines.extend(f'{indent}{line}' for line in entry.splitlines())

    return lines


def format_options_document(title: str, selectors: Mapping[str, str], defaults: BaseModel) -> str:
    """`defaults` as a commented YAML document headed by `selectors`.

    `title` opens the header comment, and `selectors` name what the document configures,
    so it can be saved and passed straight back. Each field's description is written above
    it as a comment, which :func:`yaml.safe_load` discards; a nested model is written as a
    nested mapping, its own fields commented the same way.
    """
    lines = [
        f'# {title}, at their defaults. Save this,',
        '# edit it, and pass it back with --options.',
        yaml.safe_dump(dict(selectors), sort_keys=False).rstrip(),
        'options:',
    ]
    lines.extend(_format_fields(defaults, defaults.model_dump(mode='json'), 1))
    return '\n'.join(lines) + '\n'


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
    """The fields of `candidate` that `defaults` set, or that differ from `defaults`."""
    kept: dict[str, Any] = {}

    for name in type(defaults).model_fields:
        value = getattr(candidate, name)
        default = getattr(defaults, name)
        is_set = name in defaults.model_fields_set

        if isinstance(value, BaseModel) and type(value) is type(default):
            nested = _set_or_changed(value, default)

            if nested or is_set:
                kept[name] = nested
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
