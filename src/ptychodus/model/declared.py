"""What a declared options model says about each of its fields.

A model ptychodus owns states everything an editor needs inline: ``pydantic.Field``
carries a title, a description and a pair of bounds, an enum annotation carries the
values a choice accepts, and the field's own name carries the unit it is held in.
This reads that back out, one record per field, and names the settings key each
field persists as.

Flat by design. A declaration this serves is a plain object of scalars, so there is
no nesting to project, no prefix to compose, no adapter to bridge one field onto
several keys, and no table of exceptions. A declaration that needs any of those is
not one ptychodus owns.

The primitives below -- reading a model's fields, its bounds, its display unit, and
spelling a key from a field name -- are shared with the flattener, which projects
*foreign* declarations onto the same settings layer. They live here because this is
the module that outlives it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ptychodus.api.constants import LengthUnit, Unit
from ptychodus.api.parameters import Parameter, ParameterGroup

__all__ = [
    'ParameterDescription',
    'UnsupportedParameterError',
    'declare_in',
    'describe',
    'fields_of',
    'read_bounds',
    'read_display_unit',
    'render_key',
    'restore',
]


class UnsupportedParameterError(TypeError):
    """Raised for a declared field the settings layer cannot represent."""


def fields_of(model: type) -> Mapping[str, Any]:
    """Return a model's pydantic field map.

    Read reflectively because the two shapes a declaration may take -- a
    ``BaseModel`` subclass and a ``pydantic.dataclasses.dataclass`` -- share this
    attribute but no common base class.
    """
    fields: Mapping[str, Any] = getattr(model, '__pydantic_fields__')
    return fields


def read_bounds(info: Any) -> tuple[Any, Any]:
    """Read an inclusive bound pair off a pydantic field.

    An exclusive constraint is moved inward, since the range a parameter clamps
    to includes its endpoints: a field declared strictly greater than zero
    would otherwise admit exactly zero and fail validation wherever it is used.
    The offset is small enough to be indistinguishable from the excluded value
    in use, and far below any default declared against such a bound. A field
    carrying both forms of a bound takes the exclusive one, the narrower.
    """
    exclusive_offset: Final = 1.0e-12
    minimum: Any = None
    maximum: Any = None

    for constraint in info.metadata:
        # Compared against None rather than tested for truth: ge=0.0 is the most
        # common bound in this codebase and is falsy.
        lower = getattr(constraint, 'ge', None)
        upper = getattr(constraint, 'le', None)
        above = getattr(constraint, 'gt', None)
        below = getattr(constraint, 'lt', None)

        if above is not None:
            lower = above + exclusive_offset

        if below is not None:
            upper = below - exclusive_offset

        if lower is not None:
            minimum = lower

        if upper is not None:
            maximum = upper

    return minimum, maximum


def read_display_unit(info: Any, unit: Unit | None, path: str) -> LengthUnit | None:
    """The length unit a field declares to be shown in, or None where it declares none.

    Raises:
        ValueError: if the declared name is no LengthUnit member, or the field is not
            held in meters, where a length unit would mislabel the value.
    """
    extra = info.json_schema_extra
    name = extra.get('display_unit') if isinstance(extra, Mapping) else None

    if name is None:
        return None

    if unit is not Unit.METER:
        raise ValueError(f'{path} declares a display unit but is not held in meters!')

    try:
        return LengthUnit[name]
    except KeyError:
        raise ValueError(f'{path} declares an unknown display unit {name!r}!') from None


def render_key(field_name: str, *, expand_num: bool = False) -> str:
    """Spell *field_name* the way ptychodus persists it: PascalCase, units in words.

    ``step_size_x_m`` becomes ``StepSizeXInMeters``, because the unit a value is
    held in is part of what the key says and a reader of a settings file should
    not have to know the suffix convention.

    *expand_num* additionally writes the word ``num`` out as ``NumberOf``, which
    is how ptychodus's hand-written settings classes spell it. Off by default,
    and deliberately so: two backends already persist ``NumEpochs`` from
    ``num_epochs``, and expanding theirs would rename a key that is already in
    every settings file in existence.
    """
    unit = Unit.of(field_name)
    stem = unit.strip_from(field_name) if unit else field_name
    words = stem.split('_')

    if expand_num:
        words = [
            part for word in words for part in (('number', 'of') if word == 'num' else (word,))
        ]

    joined = ''.join(word[:1].upper() + word[1:] for word in words)
    return joined + (unit.words if unit else '')


@dataclass(frozen=True)
class ParameterDescription:
    """One declared field: what it means, and the key it is stored under.

    Everything an editor needs to put the field on screen, and everything the
    settings layer needs to create a live parameter for it, read off the
    declaration alone.
    """

    name: str
    annotation: Any
    default: Any
    key: str
    label: str = ''
    tool_tip: str = ''
    minimum: Any = None
    maximum: Any = None
    choices: Sequence[str] = ()
    unit: Unit | None = None
    display_unit: LengthUnit | None = None


def _choices_of(annotation: Any) -> Sequence[str]:
    """The values an enum-typed field accepts, spelled as a settings file holds them.

    By value, which is how a model ptychodus owns spells its enums. A
    declaration spelling them by member name is a foreign one.
    """
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return [str(member.value) for member in annotation]

    return []


def describe(options: Any, *, expand_num: bool = False) -> Sequence[ParameterDescription]:
    """Describe every field of *options*, in declaration order.

    Takes a populated instance rather than its class: a field declared with a
    ``default_factory`` reports its class-level default as undefined, so only an
    instance carries the value the field actually starts at.

    *expand_num* is passed through to :func:`render_key`.
    """
    described: list[ParameterDescription] = []

    for name, info in fields_of(type(options)).items():
        unit = Unit.of(name)
        minimum, maximum = read_bounds(info)
        described.append(
            ParameterDescription(
                name=name,
                annotation=info.annotation,
                default=getattr(options, name),
                key=render_key(name, expand_num=expand_num),
                label=info.title or '',
                tool_tip=info.description or '',
                minimum=minimum,
                maximum=maximum,
                choices=_choices_of(info.annotation),
                unit=unit,
                display_unit=read_display_unit(info, unit, name),
            )
        )

    return described


def _create(group: ParameterGroup, described: ParameterDescription) -> Parameter[Any]:
    annotation = described.annotation
    key = described.key
    default = described.default

    # Enums persist as their value, which is how a model ptychodus owns spells
    # them, so the live parameter is a string and the declared type is what
    # restores meaning on the way back out.
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return group.create_string_parameter(key, str(default.value))

    # bool is a subclass of int, so it has to be tested first.
    if annotation is bool:
        return group.create_boolean_parameter(key, bool(default))

    if annotation is int:
        return group.create_integer_parameter(
            key, int(default), minimum=described.minimum, maximum=described.maximum
        )

    if annotation is float:
        return group.create_real_parameter(
            key, float(default), minimum=described.minimum, maximum=described.maximum
        )

    if annotation is str:
        return group.create_string_parameter(key, str(default))

    if isinstance(annotation, type) and issubclass(annotation, Path):
        return group.create_path_parameter(key, Path(default))

    raise UnsupportedParameterError(
        f'Cannot represent "{described.name}" ({annotation}) as a settings parameter.'
    )


def declare_in(
    group: ParameterGroup, described: Sequence[ParameterDescription]
) -> Mapping[str, Parameter[Any]]:
    """Add the parameters *described* names to *group*, returning them by key.

    Idempotent, and deliberately so: a key the group already holds is reused
    rather than recreated, which is what lets two declarations naming the same
    field share one live value. They must then agree on its default, since the
    first to arrive is the one that sets it.
    """
    existing = group.parameters()
    parameters: dict[str, Parameter[Any]] = {}

    for description in described:
        parameter = existing.get(description.key)
        parameters[description.key] = (
            _create(group, description) if parameter is None else parameter
        )

    return parameters


def restore(
    defaults: Any,
    described: Sequence[ParameterDescription],
    parameters: Mapping[str, Parameter[Any]],
) -> Any:
    """Return a copy of *defaults* carrying the values *parameters* now hold.

    The inverse of :func:`declare_in`. A declared object is what a generator is
    handed, so edits made against the live parameters have to be written back
    into one before the call. A copy, because the defaults are the record of
    where each field started and must survive being read repeatedly.
    """
    restored = defaults.model_copy(deep=True)

    for description in described:
        parameter = parameters.get(description.key)

        if parameter is None:
            continue

        value = parameter.get_value()
        annotation = description.annotation

        if isinstance(annotation, type) and issubclass(annotation, Enum):
            value = annotation(value)

        setattr(restored, description.name, value)

    return restored
