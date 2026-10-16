"""Project a nested options model onto the flat INI layout the Qt GUI binds to.

**This module is transitional.** ptychodus persists one flat section of prefixed
scalar keys per settings group because Qt controllers bind to a flat
``ParameterGroup``. A declared options model is nested instead, mirroring the
structure the backend actually accepts. Until the Qt panels retire, something has
to collapse one onto the other, and that something lives here rather than in the
plugin -- so a backend never names an INI key, and so this file can be deleted
whole rather than unpicked.

The projection is rule-driven, not a key-by-key map. A nested options object
contributes a prefix; its subfields append their own names to it; a subfield that
merely enables the group collapses onto the bare prefix. Only what the rules
cannot derive is tabulated: the prefix for each nested object, the handful of
fields renamed on the way out, and the few that expand into several keys.

Naming policy is per model, because no single convention fits: ptychodus's own
groups are PascalCase with spelled-out units, while two backends persist the
snake_case field names of the libraries they wrap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

__all__ = [
    'BoolEnum',
    'compose_key',
    'fields_of',
    'FlatParameter',
    'FlatteningPolicy',
    'OptionalGate',
    'OptionalRange',
    'Sentinel',
    'StructuralAdapter',
    'flatten_options',
    'render_value',
]


class StructuralAdapter(ABC):
    """Bridges one declared field and the key or keys ptychodus persists it as.

    Most fields persist as themselves. A few do not: an optional value becomes a
    switch and a value, an enum becomes a boolean, an absent value becomes a
    sentinel. Each such shape is a named kind implemented once here rather than a
    pair of functions written out per field.
    """

    @property
    @abstractmethod
    def fields(self) -> Mapping[str, type]:
        """Key suffix to the type the live parameter behind it should hold."""

    @abstractmethod
    def encode(self, value: Any) -> Mapping[str, Any]:
        """Split a declared value into a value per key suffix."""

    @abstractmethod
    def decode(self, values: Mapping[str, Any]) -> Any:
        """Rebuild the declared value from the values now held under each suffix."""


@dataclass(frozen=True)
class OptionalGate(StructuralAdapter):
    """An optional value persisted as a switch plus a value.

    ``None`` upstream means off, or automatic. Persisting it as a bare value
    would make the off state indistinguishable from a real one, so ptychodus
    keeps a boolean beside it and remembers the value the switch was last set to.
    """

    gate: str
    value: str
    value_type: type = float
    fallback: Any = 0.0
    # True when the switch being on is what makes the value absent upstream, as
    # for the 'Auto...' factors where setting one disables the automatic choice.
    inverted: bool = False

    @property
    def fields(self) -> Mapping[str, type]:
        return {self.gate: bool, self.value: self.value_type}

    def encode(self, value: Any) -> Mapping[str, Any]:
        present = value is not None
        return {
            self.gate: (not present) if self.inverted else present,
            self.value: self.fallback if value is None else value,
        }

    def decode(self, values: Mapping[str, Any]) -> Any:
        on = bool(values[self.gate])
        present = (not on) if self.inverted else on
        return values[self.value] if present else None


@dataclass(frozen=True)
class OptionalRange(StructuralAdapter):
    """An optional pair of bounds persisted as a switch plus a low and a high key."""

    gate: str
    minimum: str
    maximum: str
    value_type: type = float
    fallback: tuple[Any, Any] = (0.0, 0.0)

    @property
    def fields(self) -> Mapping[str, type]:
        return {self.gate: bool, self.minimum: self.value_type, self.maximum: self.value_type}

    def encode(self, value: Any) -> Mapping[str, Any]:
        low, high = self.fallback if value is None else (value[0], value[1])
        return {self.gate: value is not None, self.minimum: low, self.maximum: high}

    def decode(self, values: Mapping[str, Any]) -> Any:
        if not values[self.gate]:
            return None

        return (values[self.minimum], values[self.maximum])


@dataclass(frozen=True)
class Sentinel(StructuralAdapter):
    """An optional value persisted as a single key, with one value meaning None."""

    key: str
    value_type: type = int
    sentinel: Any = -1

    @property
    def fields(self) -> Mapping[str, type]:
        return {self.key: self.value_type}

    def encode(self, value: Any) -> Mapping[str, Any]:
        return {self.key: self.sentinel if value is None else value}

    def decode(self, values: Mapping[str, Any]) -> Any:
        held = values[self.key]
        return None if held == self.sentinel else held


@dataclass(frozen=True)
class BoolEnum(StructuralAdapter):
    """An enum persisted as a boolean, where ptychodus exposes only two members."""

    key: str
    when_true: Any = None
    when_false: Any = None

    @property
    def fields(self) -> Mapping[str, type]:
        return {self.key: bool}

    def encode(self, value: Any) -> Mapping[str, Any]:
        return {self.key: value == self.when_true}

    def decode(self, values: Mapping[str, Any]) -> Any:
        return self.when_true if values[self.key] else self.when_false


# Suffixes ptychodus spells out in persisted keys, longest first so that a
# compound suffix is not shadowed by a shorter one that also matches.
_UNIT_SUFFIXES: Final[Mapping[str, str]] = {
    'm2_per_kg': 'SquareMetersPerKilogram',
    'deg': 'InDegrees',
    'rad': 'InRadians',
    'eV': 'InElectronVolts',
    'px': 'InPixels',
    'tr': 'InTurns',
    'm': 'InMeters',
    's': 'InSeconds',
}


def to_pascal_case(name: str) -> str:
    """Render a snake_case field name in ptychodus's persisted-key style."""
    for suffix in sorted(_UNIT_SUFFIXES, key=len, reverse=True):
        if name.endswith(f'_{suffix}'):
            stem = name[: -len(suffix) - 1]
            return _join_words(stem) + _UNIT_SUFFIXES[suffix]

    return _join_words(name)


def _join_words(name: str) -> str:
    return ''.join(word[:1].upper() + word[1:] for word in name.split('_'))


# A name beginning with this escapes its group's prefix. A handful of pty-chi
# keys sit unprefixed inside an otherwise prefixed group -- UseTotalIntensityForCOM
# within ConstrainCenter, for one -- and nothing in the structure predicts which.
_ABSOLUTE = '/'


def compose_key(prefix: str, name: str) -> str:
    """Join a group prefix and a key name, honouring the absolute marker."""
    return name[1:] if name.startswith(_ABSOLUTE) else prefix + name


def _is_nested(value: Any) -> bool:
    """Return True for a nested pydantic object, whether model or dataclass.

    Tested on the value rather than the annotation, so an optional nesting left
    unset is treated as a leaf instead of being walked into.
    """
    return hasattr(value, '__pydantic_fields__')


def fields_of(model: type) -> Mapping[str, Any]:
    """Return a model's pydantic field map.

    Read reflectively because the two shapes a declaration may take -- a
    ``BaseModel`` subclass and a ``pydantic.dataclasses.dataclass`` -- share this
    attribute but no common base class.
    """
    fields: Mapping[str, Any] = getattr(model, '__pydantic_fields__')
    return fields


@dataclass(frozen=True)
class FlatParameter:
    """One persisted key, and where in the declared model its value comes from.

    Label, tool tip and bounds are read from the declaration itself: pydantic's
    ``Field(title=..., description=..., ge=..., le=...)`` already carries all four,
    so a model ptychodus owns states them inline and needs no sidecar.
    """

    ini_key: str
    path: str
    annotation: Any
    default: Any
    label: str = ''
    tool_tip: str = ''
    minimum: Any = None
    maximum: Any = None
    # Set when several keys share one declared field, naming this key's role.
    # The value adapter is the backend's, since only it knows the field's shape.
    structural_role: str = ''


@dataclass(frozen=True)
class FlatteningPolicy:
    """Everything the rules cannot derive, for one declared model.

    ``prefixes`` and ``renames`` are keyed by dotted path rather than by field
    name: the same leaf name can mean different things at different depths, and a
    name-keyed table would silently apply the wrong one.
    """

    # Prepended to every key this policy produces. A section covering an
    # algorithm's additions to a shared branch files them under its own name:
    # the object branch's extras persist as ObjectOptimalStepSizeScaler.
    base_prefix: str = ''
    # Dotted paths ptychodus does not surface. Hiding a nested object hides its
    # whole subtree. This exists for declarations ptychodus does not own -- a model
    # it writes simply omits the field instead.
    hidden: frozenset[str] = frozenset()
    # Dotted path of a nested model -> the segment it contributes. '' makes the
    # nesting transparent, so its subfields sit directly under the parent prefix.
    prefixes: Mapping[str, str] = field(default_factory=dict)
    # Dotted path of a leaf -> its persisted name, where the rules miss.
    renames: Mapping[str, str] = field(default_factory=dict)
    # Dotted path of a leaf -> the adapter bridging it and the keys it persists as.
    structural: Mapping[str, StructuralAdapter] = field(default_factory=dict)
    # Leaf names that merely switch their group on, and so take the bare prefix.
    collapse_onto_prefix: frozenset[str] = frozenset({'enabled', 'optimizable'})
    # ptychodus persists pty-chi enums by member name; the backends it owns
    # persist theirs by value.
    enum_by_name: bool = True
    # Rendered when the policy is applied to a model with no naming convention of
    # its own, i.e. one that persists its declared field names verbatim.
    verbatim: bool = False

    def is_hidden(self, path: str) -> bool:
        """Return True if *path*, or any ancestor of it, is hidden."""
        parts = path.split('.')
        return any('.'.join(parts[: index + 1]) in self.hidden for index in range(len(parts)))

    def render_name(self, path: str, name: str) -> str:
        try:
            return self.renames[path]
        except KeyError:
            return name if self.verbatim else to_pascal_case(name)


def flatten_options(options: Any, policy: FlatteningPolicy) -> list[FlatParameter]:
    """Return the persisted keys for *options*, in declaration order.

    Takes a populated object rather than its class: a field declared with a
    ``default_factory`` reports its class-level default as undefined, so only an
    instance carries the value the generated parameter should start at.
    """
    return list(_walk(options, policy, path='', prefix=policy.base_prefix))


def _walk(options: Any, policy: FlatteningPolicy, *, path: str, prefix: str) -> list[FlatParameter]:
    flat: list[FlatParameter] = []

    for name, info in fields_of(type(options)).items():
        child_path = f'{path}.{name}' if path else name

        if policy.is_hidden(child_path):
            continue

        value = getattr(options, name)

        if _is_nested(value):
            segment = policy.prefixes.get(child_path, policy.render_name(child_path, name))
            flat.extend(_walk(value, policy, path=child_path, prefix=prefix + segment))
            continue

        if prefix and name in policy.collapse_onto_prefix:
            flat.append(_leaf(prefix, child_path, info, value))
            continue

        try:
            adapter = policy.structural[child_path]
        except KeyError:
            key = compose_key(prefix, policy.render_name(child_path, name))
            flat.append(_leaf(key, child_path, info, value))
        else:
            encoded = adapter.encode(value)
            flat.extend(
                _leaf(
                    compose_key(prefix, suffix),
                    child_path,
                    info,
                    encoded[suffix],
                    role=suffix,
                    annotation=annotation,
                )
                for suffix, annotation in adapter.fields.items()
            )

    return flat


def _bounds(info: Any) -> tuple[Any, Any]:
    """Read inclusive bounds off a pydantic field.

    Only ``ge`` and ``le`` are mapped: ptychodus parameters clamp to an inclusive
    range and cannot express an exclusive one.
    """
    minimum: Any = None
    maximum: Any = None

    for constraint in info.metadata:
        # Compared against None rather than tested for truth: ge=0.0 is the most
        # common bound in this codebase and is falsy.
        lower = getattr(constraint, 'ge', None)
        upper = getattr(constraint, 'le', None)

        if lower is not None:
            minimum = lower

        if upper is not None:
            maximum = upper

    return minimum, maximum


def _leaf(
    key: str,
    path: str,
    info: Any,
    value: Any,
    *,
    role: str = '',
    annotation: Any = None,
) -> FlatParameter:
    minimum, maximum = _bounds(info)
    return FlatParameter(
        ini_key=key,
        path=path,
        annotation=info.annotation if annotation is None else annotation,
        default=value,
        label=info.title or '',
        tool_tip=info.description or '',
        minimum=minimum,
        maximum=maximum,
        structural_role=role,
    )


def render_value(value: Any, policy: FlatteningPolicy) -> str:
    """Render *value* the way the persisted file spells it.

    An enum is the case that matters: ptychodus writes pty-chi's enums by member
    name, so serializing a ``StrEnum`` as a plain string would silently rewrite
    every such key from ``SGD`` to ``sgd``.
    """
    if isinstance(value, Enum):
        return value.name if policy.enum_by_name else str(value.value)

    return str(value)
