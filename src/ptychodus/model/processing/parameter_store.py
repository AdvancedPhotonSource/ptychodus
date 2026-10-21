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

from .flatten import FlatParameter, FlatteningPolicy, flatten_options
from .generate import apply_options, populate_parameter_group

__all__ = [
    'DescribedField',
    'DescribedGroup',
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
class DescribedField:
    """One persisted key: what the declaration says about it, and what holds it.

    Pairs the flattened description -- type, bounds, choices, label, tool tip --
    with the live parameter generated from it, which is everything needed to
    put an editor on screen without knowing which backend it came from.
    """

    described: FlatParameter
    parameter: Parameter[Any]


@dataclass(frozen=True)
class DescribedGroup:
    """One settings group, in the order the declaration lists its fields."""

    name: str
    fields: Sequence[DescribedField]


@dataclass(frozen=True)
class _BoundSection:
    section: OptionsSection
    parameters: Mapping[str, Parameter[Any]]


@dataclass(frozen=True)
class _Entry:
    defaults: Any
    sections: Sequence[_BoundSection]


def _compose_key(library: str, algorithm: str) -> str:
    """Name the entry holding one reconstructor's settings.

    The format is this module's own: nothing outside ``model/processing``
    composes one, and nothing outside it should. Callers name the pair and
    this decides how the two halves are spelled together -- which is what
    lets the spelling change without a search across the tree.
    """
    return f'{library.casefold()}_{algorithm.casefold()}'


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
        self._groups: dict[str, ParameterGroup] = {}

    def register(
        self,
        library: str,
        reconstructor: Reconstructor,
        sections: Sequence[OptionsSection] = (),
    ) -> None:
        """Generate the live parameters backing *reconstructor* of *library*.

        A reconstructor that declares nothing registers no settings group at all:
        an empty section would appear in every saved settings file and mean
        nothing. Its options are still served, unchanged from their defaults.
        """
        key = _compose_key(library, reconstructor.name)
        defaults = reconstructor.get_default_options()

        if not sections and type(defaults).__pydantic_fields__:
            logger.warning(
                f'{key} declares options but no settings sections were registered for it; '
                'they will not be editable and every run will use the declared defaults.'
            )

        bound: list[_BoundSection] = []

        for section in sections:
            root = _resolve(defaults, section.root_path)

            if not flatten_options(root, section.policy):
                logger.debug(f'{key}: section "{section.group_name}" declares nothing; skipped.')
                continue

            # Created on first use and remembered across registrations, so
            # that several sections can feed one group and several
            # reconstructors can share one. Two modes of the same backend are
            # configured by one set of keys, and a settings file has one
            # section for them.
            group = self._groups.get(section.group_name)

            if group is None:
                group = self._settings_registry.create_group(section.group_name)
                self._groups[section.group_name] = group

            parameters = populate_parameter_group(group, root, section.policy)
            bound.append(_BoundSection(section, parameters))

        self._entries[key] = _Entry(defaults, bound)

    def get_options(self, library: str, algorithm: str) -> Any:
        """Return the declared object carrying the values currently held.

        Falls back to the declared defaults for a reconstructor that was never
        registered, so a caller does not have to know which backends have
        converted.
        """
        key = _compose_key(library, algorithm)

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

    def describe(self, library: str, algorithm: str) -> Sequence[DescribedGroup]:
        """Return the settings groups, each field paired with its live parameter.

        What a renderer needs and all it needs: a declaration says what kind of
        value a field holds and what it may be, and the parameter beside it is
        what an editor binds to. Several sections feeding one group are merged,
        because the panel shows one group.

        A key this reconstructor persists but ignores is left out. It is
        described away rather than never generated, because the group holding
        it is shared: two reconstructors of one backend write the same section,
        and which of them was registered first must not decide what that
        section contains.
        """
        try:
            entry = self._entries[_compose_key(library, algorithm)]
        except KeyError:
            return []

        merged: dict[str, list[DescribedField]] = {}

        for bound in entry.sections:
            root = _resolve(entry.defaults, bound.section.root_path)
            fields = merged.setdefault(bound.section.group_name, [])

            for described in flatten_options(root, bound.section.policy):
                if described.panel_hidden:
                    continue

                parameter = bound.parameters.get(described.ini_key)

                if parameter is not None:
                    fields.append(DescribedField(described, parameter))

        return [DescribedGroup(name, fields) for name, fields in merged.items()]

    def is_registered(self, library: str, algorithm: str) -> bool:
        """Report whether this reconstructor's settings were generated."""
        return _compose_key(library, algorithm) in self._entries
