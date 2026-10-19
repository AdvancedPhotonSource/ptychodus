"""Render a reconstructor's panel from what the backend declares.

A declaration already says everything an editor needs: the kind of value a
field holds, the range it may take, the values it accepts, and the words to
put beside it. So the panel is derived rather than written, and a backend
gains one by declaring rather than by someone hand-wiring a widget per key.

Widget choice follows from the declaration alone. A field with a fixed set of
values is a combo box and a boolean is a check box. A number is editable
within its bounds where the declaration gives them: a whole number with a
floor is a spin box, and a real number with both ends is a slider. Without
those a spin box has nothing to count from and a slider nothing to slide
between, so both fall back to a line edit that accepts any value of the right
type.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from types import UnionType
from typing import Any, Union, get_args, get_origin
import logging

from PyQt5.QtWidgets import QLabel, QWidget

from ptychodus.api.parameters import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)
from ptychodus.model.processing import (
    DescribedField,
    ProcessingAlgorithmParameter,
    ProcessingParameterStore,
)

from ..parameters import ParameterViewBuilder
from .core import ReconstructorViewControllerFactory

__all__ = ['GeneratedViewControllerFactory']

logger = logging.getLogger(__name__)


def _held_type(annotation: Any) -> Any:
    """Return the type a field holds, looking through an optional wrapper.

    A declaration routinely spells an unset value as ``int | None``; the live
    parameter stores the inner type and treats absence as the default, so the
    wrapper says nothing about which editor to build.
    """
    if get_origin(annotation) not in (Union, UnionType):
        return annotation

    members = [arg for arg in get_args(annotation) if arg is not type(None)]
    return members[0] if len(members) == 1 else annotation


def _choices(field: DescribedField) -> Sequence[str]:
    """Return the values a field accepts, from wherever the declaration states them.

    Two sources, because the declarations differ in what they can carry. One
    ptychodus owns types a choice as an enum, so the members are in the
    annotation. One it does not own may spell every choice as a bare string,
    and then the policy supplies the list.
    """
    if field.described.choices:
        return field.described.choices

    annotation = _held_type(field.described.annotation)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return [str(member.value) for member in annotation]

    return []


def _add_field(builder: ParameterViewBuilder, field: DescribedField, group: str) -> None:
    """Put one editor on the panel, chosen by what the field is."""
    described = field.described
    parameter = field.parameter
    label = described.label or described.ini_key
    choices = _choices(field)

    if choices and isinstance(parameter, StringParameter):
        builder.add_combo_box(parameter, choices, label, tool_tip=described.tool_tip, group=group)
    elif isinstance(parameter, BooleanParameter):
        builder.add_check_box(parameter, label, tool_tip=described.tool_tip, group=group)
    elif isinstance(parameter, IntegerParameter):
        # A spin box needs somewhere to count from; it manages without a
        # ceiling, stopping at the largest int it can show.
        if parameter.get_minimum() is None:
            builder.add_integer_line_edit(
                parameter, label, tool_tip=described.tool_tip, group=group
            )
        else:
            builder.add_spin_box(parameter, label, tool_tip=described.tool_tip, group=group)
    elif isinstance(parameter, RealParameter):
        bounded = parameter.get_minimum() is not None and parameter.get_maximum() is not None

        if bounded:
            builder.add_decimal_slider(parameter, label, tool_tip=described.tool_tip, group=group)
        else:
            builder.add_decimal_line_edit(
                parameter, label, tool_tip=described.tool_tip, group=group
            )
    elif isinstance(parameter, StringParameter):
        builder.add_line_edit(parameter, label, tool_tip=described.tool_tip, group=group)
    else:
        # Reached only by a declared type no editor covers, which is a gap in
        # this function rather than a bad declaration -- so say which field.
        logger.warning(
            f'No editor for "{described.ini_key}" ({type(parameter).__name__}); omitted.'
        )


def _group_label(group_name: str, library_name: str) -> str:
    """Strip the backend's own name off a group, leaving what distinguishes it.

    A group is named for the settings section it persists as, which carries
    the backend name because an INI is flat. A panel already knows which
    backend it belongs to, so repeating it in every box title says nothing.
    """
    stripped = group_name
    for prefix in (library_name, library_name.replace('-', '')):
        if stripped.casefold().startswith(prefix.casefold()):
            stripped = stripped[len(prefix) :]
            break

    return stripped or group_name


class GeneratedViewControllerFactory(ReconstructorViewControllerFactory):
    """Builds every panel for one backend out of what its reconstructors declare."""

    def __init__(self, library_name: str, parameter_store: ProcessingParameterStore) -> None:
        super().__init__()
        self._library_name = library_name
        self._parameter_store = parameter_store

    @property
    def name(self) -> str:
        return self._library_name

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        key = ProcessingAlgorithmParameter.generate_key(self._library_name, reconstructor_name)
        groups = self._parameter_store.describe(key)

        if not groups:
            return QLabel(f'{self._library_name} declares no settings.')

        builder = ParameterViewBuilder()

        for group in groups:
            label = _group_label(group.name, self._library_name)

            for field in group.fields:
                _add_field(builder, field, label)

        return builder.build_widget()
