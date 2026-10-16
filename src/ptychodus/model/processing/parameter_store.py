"""Live settings for each reconstructor, generated from the options it declares.

A reconstructor describes what it accepts and is handed values per call. This is
what sits between: it generates the live observable parameters from a
declaration, lets the GUI and the settings file edit them, and writes them back
into a declared object when a run starts.

**Transitional**, with the flat layout it serves. One declaration can back
several settings groups -- pty-chi's single options tree is persisted as fifteen
-- so a reconstructor registers one :class:`OptionsSection` per group, each
rooted at the part of the declaration it covers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import copy
import logging

from ptychodus.api.parameters import Parameter, ParameterGroup
from ptychodus.api.reconstruct import Reconstructor
from ptychodus.api.settings import SettingsRegistry

from .flatten import FlatteningPolicy, flatten_options
from .generate import apply_options, populate_parameter_group

__all__ = [
    'OptionsSection',
    'ProcessingParameterStore',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptionsSection:
    """One settings group generated from part of a declaration.

    ``root_path`` is the dotted path of the declared object this section covers;
    empty means the whole declaration. Several sections may name the same group:
    pty-chi's per-algorithm groups draw their keys from four separate branches of
    one declaration, so a group is assembled from however many sections feed it.
    """

    group_name: str
    root_path: str = ''
    policy: FlatteningPolicy = field(default_factory=FlatteningPolicy)


@dataclass(frozen=True)
class _BoundSection:
    section: OptionsSection
    parameters: Mapping[str, Parameter[Any]]


@dataclass(frozen=True)
class _Entry:
    defaults: Any
    sections: Sequence[_BoundSection]


def _resolve(root: Any, path: str) -> Any:
    target = root

    for name in path.split('.') if path else ():
        target = getattr(target, name)

    return target


class ProcessingParameterStore:
    """The live settings behind every registered reconstructor's declaration."""

    def __init__(self, settings_registry: SettingsRegistry) -> None:
        self._settings_registry = settings_registry
        self._entries: dict[str, _Entry] = {}

    def register(
        self, key: str, reconstructor: Reconstructor, sections: Sequence[OptionsSection] = ()
    ) -> None:
        """Generate the live parameters backing *reconstructor*, filed under *key*.

        A reconstructor that declares nothing registers no settings group at all:
        an empty section would appear in every saved settings file and mean
        nothing. Its options are still served, unchanged from their defaults.
        """
        defaults = reconstructor.get_default_options()

        if not sections and type(defaults).__pydantic_fields__:
            logger.warning(
                f'{key} declares options but no settings sections were registered for it; '
                'they will not be editable and every run will use the declared defaults.'
            )

        bound: list[_BoundSection] = []
        groups: dict[str, ParameterGroup] = {}

        for section in sections:
            root = _resolve(defaults, section.root_path)

            if not flatten_options(root, section.policy):
                logger.debug(f'{key}: section "{section.group_name}" declares nothing; skipped.')
                continue

            # Created on first use, so that several sections can feed one group
            # without the second attempt colliding with the first.
            group = groups.get(section.group_name)

            if group is None:
                group = self._settings_registry.create_group(section.group_name)
                groups[section.group_name] = group

            parameters = populate_parameter_group(group, root, section.policy)
            bound.append(_BoundSection(section, parameters))

        self._entries[key] = _Entry(defaults, bound)

    def get_options(self, key: str) -> Any:
        """Return *key*'s declared object carrying the values currently held.

        Falls back to the declared defaults for a reconstructor that was never
        registered, so a caller does not have to know which backends have
        converted.
        """
        try:
            entry = self._entries[key]
        except KeyError:
            raise KeyError(f'No options registered for "{key}".') from None

        options = copy.deepcopy(entry.defaults)

        for bound in entry.sections:
            apply_options(
                _resolve(options, bound.section.root_path),
                bound.parameters,
                bound.section.policy,
            )

        return options

    def __contains__(self, key: str) -> bool:
        return key in self._entries
