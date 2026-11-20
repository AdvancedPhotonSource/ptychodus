"""The commented options template ``--print-options`` writes, and library selection.

:func:`format_options_document` writes an options document at the defaults, each field's
description above it as a comment, for a person to edit and pass back with ``--options``;
:mod:`ptychodus.api.options` reads it. A frozen field is printed as a comment saying so.

:func:`resolve_reconstructor` finds the library and algorithm a document or the command line
names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel
import yaml

from ptychodus.api.options import OptionsError
from ptychodus.api.plugins import Plugin, PluginCatalog
from ptychodus.api.reconstruct import Reconstructor, ReconstructorLibrary, TrainableReconstructor

__all__ = [
    'format_options_document',
    'resolve_reconstructor',
    'resolve_trainable_reconstructor',
]

ReconstructorT = TypeVar('ReconstructorT', bound=Reconstructor)


def _format_fields(model: BaseModel, values: Mapping[str, Any], depth: int) -> list[str]:
    indent = '  ' * depth
    lines: list[str] = []

    for name, field in type(model).model_fields.items():
        if field.description:
            lines.append(f'{indent}# {field.description}')

        value = getattr(model, name)

        if field.frozen:
            lines.append(f'{indent}# {name} is fixed by the library and cannot be set here.')
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


def _find_library(
    libraries: PluginCatalog[ReconstructorLibrary], library_name: str
) -> Plugin[ReconstructorLibrary]:
    plugin = libraries.find_plugin(library_name)

    if plugin is None:
        raise OptionsError(
            f'No library named "{library_name}"; registered are '
            f'{libraries.stringify_plugin_names()}.'
        )

    return plugin


def _select(
    plugin: Plugin[ReconstructorLibrary],
    candidates: Sequence[ReconstructorT],
    algorithm_name: str | None,
    verb: str,
) -> ReconstructorT:
    if not candidates:
        if plugin.strategy.is_available():
            raise OptionsError(f'{plugin.display_name} has nothing that {verb}.')

        raise OptionsError(f'{plugin.display_name} is not installed.')

    names = [candidate.name for candidate in candidates]

    if algorithm_name is None:
        if len(candidates) == 1:
            return candidates[0]

        raise OptionsError(f'{plugin.display_name} {verb} {names}; name one with --algorithm.')

    for candidate in candidates:
        if candidate.name.casefold() == algorithm_name.casefold():
            return candidate

    raise OptionsError(
        f'{plugin.display_name} has no algorithm "{algorithm_name}" that {verb}; it {verb} {names}.'
    )


def resolve_reconstructor(
    libraries: PluginCatalog[ReconstructorLibrary], library_name: str, algorithm_name: str | None
) -> tuple[Plugin[ReconstructorLibrary], Reconstructor]:
    """The library and algorithm named, or the library's only one when unnamed.

    Either name matches case-insensitively, and a library by its simple or display name.

    Raises:
        OptionsError: if no library or no algorithm answers to the names, the library is not
            installed, or the algorithm is left unnamed and the library offers more than one.
    """
    plugin = _find_library(libraries, library_name)
    return plugin, _select(plugin, list(plugin.strategy), algorithm_name, 'reconstructs')


def resolve_trainable_reconstructor(
    libraries: PluginCatalog[ReconstructorLibrary], library_name: str, algorithm_name: str | None
) -> tuple[Plugin[ReconstructorLibrary], TrainableReconstructor]:
    """As :func:`resolve_reconstructor`, among the algorithms that can train."""
    plugin = _find_library(libraries, library_name)
    trainable = [
        reconstructor
        for reconstructor in plugin.strategy
        if isinstance(reconstructor, TrainableReconstructor)
    ]
    return plugin, _select(plugin, trainable, algorithm_name, 'trains')
