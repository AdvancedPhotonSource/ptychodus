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
type. A whole number carrying a list of choices is the odd one out: it is a
chosen subset rather than a single pick, one bit per name, and gets a list of
check boxes.

Layout follows from the declaration too, by two composite rules. A boolean
that switches a run of fields on encloses them in a group box it titles, so
turning it off folds them away instead of leaving two dozen dead rows on
screen. And a branch's start, stop and stride are one editor: they are a range
written as three numbers, and reading them apart is reading a range apart.

Two things a declaration does not state, and so two things no panel built here
can do. It says nothing about one field applying only while another holds a
particular value, so a field whose relevance is conditional is shown whatever
the condition; and it says nothing that reaches across settings groups, so a
value standing for "whatever that other group holds" is shown as the sentinel
it actually stores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, Final, Union, get_args, get_origin
import logging

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import BooleanParameter, IntegerParameter
from ptychodus.model.processing import (
    DescribedField,
    DescribedGroup,
    ProcessingParameterStore,
)

from ..parameters import (
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    ParameterWidget,
    create_parameter_view_controller,
    create_whole_number_view_controller,
)
from .core import ReconstructorViewControllerFactory

__all__ = ['GeneratedViewControllerFactory']

logger = logging.getLogger(__name__)

# The three fields of a plan, in the order they read, and what to call each on
# screen. Named by the declaration's own words rather than by persisted key, so
# a backend spelling its keys differently still gets one editor.
_PLAN_FIELDS: Final = ('start', 'stop', 'stride')
_PLAN_CAPTIONS: Final = ('Start', 'Stop', 'Stride')
_PLAN_LABEL: Final = 'Plan:'
_PLAN_TOOL_TIPS: Final = (
    'First epoch at which this is applied.',
    'Epoch after which this is no longer applied; -1 runs to the end.',
    'Number of epochs between applications.',
)


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
    """Return the values a field accepts, spelled as the settings file spells them.

    Two sources, because the declarations differ in what they can carry. One
    ptychodus owns types a choice as an enum, so the members are in the
    annotation. One it does not own may spell every choice as a bare string,
    and then the policy supplies the list.

    Which way an enum member is written is the key's own business and is
    settled before it gets here. Offering the other spelling would put a value
    in the box that never matches the one held, so the box would read as unset
    and picking from it would write something nothing can read back.
    """
    described = field.described

    if described.choices:
        return described.choices

    annotation = _held_type(described.annotation)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return described.spelling.members(annotation)

    return []


def _label(field: DescribedField) -> str:
    """What to write beside a field, falling back to the name it persists as."""
    return field.described.label or field.described.ini_key


def _branch(path: str) -> str:
    """The dotted path of the branch a leaf sits on, empty at the root."""
    branch, _, _ = path.rpartition('.')
    return branch


class _BitFieldListModel(QAbstractListModel):
    """One checkable row per bit of a whole number holding a chosen subset."""

    def __init__(
        self, parameter: IntegerParameter, names: Sequence[str], parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._parameter = parameter
        self._names = list(names)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self._names[index.row()]

        if role == Qt.ItemDataRole.CheckStateRole:
            is_set = self._parameter.get_value() & (1 << index.row())
            return Qt.CheckState.Checked if is_set else Qt.CheckState.Unchecked

        return None

    def setData(  # noqa: N802
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False

        mask = 1 << index.row()
        bits = self._parameter.get_value()
        self._parameter.set_value(bits | mask if value == Qt.CheckState.Checked else bits & ~mask)
        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._names)


class _BitFieldViewController(ParameterViewController, Observer):
    """Edit a whole number that names a subset, one check box per bit."""

    def __init__(
        self, parameter: IntegerParameter, names: Sequence[str], *, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._model = _BitFieldListModel(parameter, names)
        self._widget = QListView()
        self._widget.setModel(self._model)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._model.layoutChanged.emit()


class _PlanViewController(ParameterViewController):
    """A branch's start, stop and stride on one row, each captioned by what it is."""

    def __init__(self, parameters: Sequence[IntegerParameter]) -> None:
        super().__init__()
        self._controllers = [
            create_whole_number_view_controller(parameter, tool_tip=tool_tip)
            for parameter, tool_tip in zip(parameters, _PLAN_TOOL_TIPS)
        ]
        self._widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        for caption, controller in zip(_PLAN_CAPTIONS, self._controllers):
            layout.addWidget(QLabel(caption), 0)
            layout.addWidget(controller.get_widget(), 1)

        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


def _editor(field: DescribedField) -> ParameterViewController | None:
    """The editor one field gets, or None for a declared type none covers.

    Widget choice is shared with every other panel built from a declaration;
    only the one case this declaration shape has of its own is answered here.
    """
    described = field.described
    parameter = field.parameter

    # A whole number carrying a list of choices is a chosen subset rather than
    # a single pick, one bit per name. Nothing but a nested upstream
    # declaration spells a field that way, so it stays with the flattener.
    if isinstance(parameter, IntegerParameter) and described.choices:
        return _BitFieldViewController(parameter, described.choices, tool_tip=described.tool_tip)

    return create_parameter_view_controller(
        parameter,
        name=described.ini_key,
        label=_label(field),
        tool_tip=described.tool_tip,
        choices=_choices(field),
        unit=described.unit,
        display_unit=described.display_unit,
    )


@dataclass(frozen=True)
class _Leaf:
    """One field, rendered as itself."""

    field: DescribedField


@dataclass(frozen=True)
class _Plan:
    """A branch's start, stop and stride, rendered as one editor."""

    parameters: Sequence[IntegerParameter]


@dataclass(frozen=True)
class _Box:
    """A switch and the fields it turns on, rendered as a checkable group box."""

    switch: BooleanParameter
    title: str
    tool_tip: str
    items: Sequence[_Item]


_Item = _Leaf | _Plan | _Box


def _governs(gate: DescribedField, other: DescribedField) -> bool:
    """Whether *gate* is the switch deciding that *other* applies at all.

    Two shapes, because a declaration states the idea twice. A branch states
    its own enable flag as a field, and the rest of that branch is what the
    flag enables. An optional value persists instead as a flag beside the
    value it remembers, both from the one declared field, and there too the
    flag is what says whether the value applies.

    A flag at the root of a section enables the section rather than a branch
    of it, so it governs nothing here; the group box itself carries that one.
    """
    described = gate.described

    if described.gates_group:
        branch = _branch(described.path)
        return bool(branch) and other.described.path.startswith(f'{branch}.')

    return bool(described.structural_role) and other.described.path == described.path


def _switches_section(field: DescribedField) -> bool:
    """Whether this field is the flag enabling everything its section declares."""
    return (
        isinstance(field.parameter, BooleanParameter)
        and field.described.gates_group
        and not _branch(field.described.path)
    )


def _plan_parameters(items: Sequence[_Item]) -> list[IntegerParameter]:
    """These items as one branch's start, stop and stride, or empty if they are not."""
    if len(items) != len(_PLAN_FIELDS):
        return []

    parameters: list[IntegerParameter] = []
    paths: list[str] = []

    for item in items:
        if not isinstance(item, _Leaf) or not isinstance(item.field.parameter, IntegerParameter):
            return []

        parameters.append(item.field.parameter)
        paths.append(item.field.described.path)

    if len({_branch(path) for path in paths}) != 1:
        return []

    if [path.rpartition('.')[2] for path in paths] != list(_PLAN_FIELDS):
        return []

    return parameters


def _fold_plans(items: Sequence[_Item]) -> list[_Item]:
    """Collapse each start/stop/stride run into the single editor it is."""
    folded: list[_Item] = []
    index = 0

    while index < len(items):
        parameters = _plan_parameters(items[index : index + len(_PLAN_FIELDS)])

        if parameters:
            folded.append(_Plan(parameters))
            index += len(_PLAN_FIELDS)
        else:
            folded.append(items[index])
            index += 1

    return folded


def _split(fields: Sequence[DescribedField]) -> list[_Item]:
    """Nest each run of fields under the switch that turns it on.

    A switch governs only the run that follows it: a declaration lists a
    branch's enable flag before the branch, so a switch and what it switches
    are always adjacent, and a field reached by neither stays where it is.
    """
    items: list[_Item] = []
    index = 0

    while index < len(fields):
        gate = fields[index]
        switch = gate.parameter if isinstance(gate.parameter, BooleanParameter) else None
        span = 0

        if switch is not None:
            while index + 1 + span < len(fields) and _governs(gate, fields[index + 1 + span]):
                span += 1

        if switch is not None and span:
            items.append(_box(gate, switch, _split(fields[index + 1 : index + 1 + span])))
        else:
            items.append(_Leaf(gate))

        index += 1 + span

    return _fold_plans(items)


def _box(gate: DescribedField, switch: BooleanParameter, items: Sequence[_Item]) -> _Box:
    """A checkable box over *items*, titled by the field that switches them on."""
    return _Box(switch, _label(gate).rstrip(':'), gate.described.tool_tip, items)


def _checkable_box(box: _Box, controllers: list[ParameterViewController]) -> QWidget:
    """Realize a checkable box, collecting what has to stay alive."""
    controller = CheckableGroupBoxParameterViewController(
        box.switch,
        box.title,
        tool_tip=box.tool_tip,
        # The panels stack two dozen of these, most off in any given run, so
        # the ones that are off reduce to their title row.
        collapse_when_unchecked=True,
    )
    controllers.append(controller)
    controller.get_widget().set_contents_layout(_build_layout(box.items, controllers))
    return controller.get_widget()


def _build_layout(
    items: Sequence[_Item], controllers: list[ParameterViewController]
) -> QFormLayout:
    """Lay out one level of the panel, collecting what has to stay alive."""
    layout = QFormLayout()

    for item in items:
        if isinstance(item, _Box):
            layout.addRow(_checkable_box(item, controllers))
        elif isinstance(item, _Plan):
            plan = _PlanViewController(item.parameters)
            controllers.append(plan)
            layout.addRow(_PLAN_LABEL, plan.get_widget())
        else:
            editor = _editor(item.field)

            if editor is None:
                continue

            controllers.append(editor)
            # A check box already carries its text, so its row needs no label.
            label = '' if isinstance(item.field.parameter, BooleanParameter) else _label(item.field)
            layout.addRow(label, editor.get_widget())

    return layout


def _group_label(group_name: str, library_name: str, reconstructor_name: str = '') -> str:
    """Strip the backend's or the algorithm's own name off a group, leaving what
    distinguishes it.

    A group is named for the settings section it persists as, which carries
    the backend name -- or, in a library of several algorithms, the algorithm's --
    because an INI is flat. A panel already knows which backend and algorithm it
    belongs to, so repeating either in every box title says nothing.
    """
    stripped = group_name
    prefixes = (library_name, library_name.replace('-', ''), reconstructor_name)

    for prefix in filter(None, prefixes):
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
        groups = self._parameter_store.describe(self._library_name, reconstructor_name)

        if not groups:
            return QLabel(f'{self._library_name} declares no settings.')

        controllers: list[ParameterViewController] = []
        layout = QVBoxLayout()

        for group in groups:
            layout.addWidget(self._build_group(group, reconstructor_name, controllers))

        layout.addStretch()

        widget = ParameterWidget(controllers)
        widget.setLayout(layout)
        return widget

    def _build_group(
        self,
        group: DescribedGroup,
        reconstructor_name: str,
        controllers: list[ParameterViewController],
    ) -> QWidget:
        fields = list(group.fields)
        gate = fields[0] if fields and _switches_section(fields[0]) else None

        if gate is not None and isinstance(gate.parameter, BooleanParameter):
            # Its label says what the whole group is for, which is a better
            # title than the section name and leaves the group switchable.
            return _checkable_box(_box(gate, gate.parameter, _split(fields[1:])), controllers)

        box = QGroupBox(_group_label(group.name, self._library_name, reconstructor_name))
        box.setLayout(_build_layout(_split(fields), controllers))
        return box
