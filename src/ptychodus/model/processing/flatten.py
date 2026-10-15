"""Project a nested parameter model onto the flat INI layout the Qt GUI binds to.

**This module is transitional.** ptychodus persists one flat section of prefixed
scalar keys per settings group because Qt controllers bind to a flat
``ParameterGroup``. A declared parameter model is nested instead, mirroring the
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

__all__ = [
    'FlatParameter',
    'FlatteningPolicy',
    'flatten_model',
    'render_value',
]

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


def _is_model(annotation: Any) -> bool:
    """Return True for a pydantic model or pydantic dataclass, which nest alike."""
    return hasattr(annotation, '__pydantic_fields__')


def _fields_of(model: type) -> Mapping[str, Any]:
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

    # Dotted paths ptychodus does not surface. Hiding a nested object hides its
    # whole subtree. This exists for declarations ptychodus does not own -- a model
    # it writes simply omits the field instead.
    hidden: frozenset[str] = frozenset()
    # Dotted path of a nested model -> the segment it contributes. '' makes the
    # nesting transparent, so its subfields sit directly under the parent prefix.
    prefixes: Mapping[str, str] = field(default_factory=dict)
    # Dotted path of a leaf -> its persisted name, where the rules miss.
    renames: Mapping[str, str] = field(default_factory=dict)
    # Dotted path of a leaf -> the several key suffixes it expands into.
    structural: Mapping[str, Sequence[str]] = field(default_factory=dict)
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


def flatten_model(model: type, policy: FlatteningPolicy) -> list[FlatParameter]:
    """Return the persisted keys for *model*, in declaration order."""
    return list(_walk(model, policy, path='', prefix=''))


def _walk(model: type, policy: FlatteningPolicy, *, path: str, prefix: str) -> list[FlatParameter]:
    flat: list[FlatParameter] = []

    for name, info in _fields_of(model).items():
        child_path = f'{path}.{name}' if path else name

        if policy.is_hidden(child_path):
            continue

        annotation = info.annotation

        if _is_model(annotation):
            segment = policy.prefixes.get(child_path, policy.render_name(child_path, name))
            flat.extend(_walk(annotation, policy, path=child_path, prefix=prefix + segment))
            continue

        if prefix and name in policy.collapse_onto_prefix:
            flat.append(_leaf(prefix, child_path, info))
            continue

        try:
            roles = policy.structural[child_path]
        except KeyError:
            key = prefix + policy.render_name(child_path, name)
            flat.append(_leaf(key, child_path, info))
        else:
            flat.extend(_leaf(prefix + role, child_path, info, role=role) for role in roles)

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


def _leaf(key: str, path: str, info: Any, *, role: str = '') -> FlatParameter:
    minimum, maximum = _bounds(info)
    return FlatParameter(
        ini_key=key,
        path=path,
        annotation=info.annotation,
        default=info.default,
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
