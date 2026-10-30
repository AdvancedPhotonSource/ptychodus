"""Unit tests for the panel a reconstructor's declaration is rendered into.

Two halves, split by what each needs. Nesting a run of fields under the switch
that turns it on, folding a plan's three numbers into one editor, and spelling
an enum's members the way the settings file does are decisions about a
declaration, not about widgets: they are a pure ``Sequence[DescribedField] ->
list[_Item]`` pipeline and are exercised as one. Which editor a field actually
gets is a widget, and needs a ``QApplication`` to build.

A `DescribedField` is assembled here rather than flattened from a model, so a
shape the backends do not currently produce can still be stated.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtWidgets import QApplication, QComboBox

from ptychodus.api.constants import LengthUnit
from ptychodus.model.parameters import (
    BooleanParameter,
    IntegerParameter,
    Parameter,
    RealParameter,
    StringParameter,
)
from ptychodus.controller.parameters import (
    AngleParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    IntegerLineEditParameterViewController,
    LengthParameterViewController,
    LineEditParameterViewController,
    SpinBoxParameterViewController,
)
from ptychodus.controller.processing.generated import (
    _Box,
    _Leaf,
    _Plan,
    _BitFieldListModel,
    _choices,
    _editor,
    _group_label,
    _split,
)
from ptychodus.api.constants import Unit
from ptychodus.model.processing.flatten import EnumSpelling, FlatParameter
from ptychodus.model.processing.parameter_store import DescribedField


class Optimizers(StrEnum):
    SGD = 'sgd'
    ADAM = 'adam'


def _field(
    path: str,
    parameter: Parameter[Any],
    *,
    ini_key: str = '',
    label: str = '',
    gates_group: bool = False,
    structural_role: str = '',
    choices: tuple[str, ...] = (),
    annotation: Any = None,
    spelling: EnumSpelling = EnumSpelling.BY_NAME,
    unit: Unit | None = None,
    display_unit: LengthUnit | None = None,
) -> DescribedField:
    return DescribedField(
        FlatParameter(
            ini_key=ini_key or path.replace('.', ''),
            path=path,
            annotation=annotation if annotation is not None else type(parameter.get_value()),
            default=parameter.get_value(),
            label=label,
            choices=choices,
            structural_role=structural_role,
            spelling=spelling,
            gates_group=gates_group,
            unit=unit,
            display_unit=display_unit,
        ),
        parameter,
    )


def _flag(path: str, *, gates_group: bool = False, label: str = '') -> DescribedField:
    return _field(path, BooleanParameter(False, None), gates_group=gates_group, label=label)


def _count(path: str, *, minimum: int | None = None) -> DescribedField:
    return _field(path, IntegerParameter(0, None, minimum=minimum))


def _plan_fields(branch: str) -> list[DescribedField]:
    return [_count(f'{branch}.{name}') for name in ('start', 'stop', 'stride')]


class TestNestingAFieldUnderItsSwitch:
    def test_a_branch_switch_encloses_the_rest_of_its_branch(self) -> None:
        items = _split(
            [
                _flag('l1.enabled', gates_group=True, label='Constrain L1'),
                _field('l1.weight', RealParameter(1.0, None)),
            ]
        )

        (box,) = items
        assert isinstance(box, _Box)
        assert box.title == 'Constrain L1'
        assert [item.field.described.path for item in box.items if isinstance(item, _Leaf)] == [
            'l1.weight'
        ]

    def test_a_field_on_another_branch_stays_outside(self) -> None:
        items = _split(
            [
                _flag('l1.enabled', gates_group=True),
                _field('l1.weight', RealParameter(1.0, None)),
                _field('l2.weight', RealParameter(1.0, None)),
            ]
        )

        box, outside = items
        assert isinstance(box, _Box)
        assert isinstance(outside, _Leaf)
        assert outside.field.described.path == 'l2.weight'

    def test_a_switch_at_the_root_governs_nothing_here(self) -> None:
        """The group box itself carries a section's own switch."""
        items = _split(
            [_flag('enabled', gates_group=True), _field('step', RealParameter(1.0, None))]
        )

        assert [type(item) for item in items] == [_Leaf, _Leaf]

    def test_an_adapter_gate_governs_only_the_value_it_remembers(self) -> None:
        """An optional value persists as a flag beside the value, from one field."""
        items = _split(
            [
                _field(
                    'blur',
                    BooleanParameter(False, None),
                    structural_role='EnableBlur',
                    label='Blur Patterns',
                ),
                _field('blur', RealParameter(1.0, None), structural_role='BlurSigma'),
                _field('other', RealParameter(1.0, None)),
            ]
        )

        box, outside = items
        assert isinstance(box, _Box)
        assert box.title == 'Blur Patterns'
        assert len(box.items) == 1
        assert isinstance(outside, _Leaf)

    def test_a_plain_boolean_encloses_nothing(self) -> None:
        items = _split([_flag('a'), _field('b', RealParameter(1.0, None))])

        assert [type(item) for item in items] == [_Leaf, _Leaf]

    def test_nesting_recurses(self) -> None:
        items = _split(
            [
                _flag('outer.enabled', gates_group=True),
                _flag('outer.inner.enabled', gates_group=True),
                _field('outer.inner.weight', RealParameter(1.0, None)),
            ]
        )

        (outer,) = items
        assert isinstance(outer, _Box)
        (inner,) = outer.items
        assert isinstance(inner, _Box)


class TestFoldingAPlan:
    def test_three_fields_of_one_branch_become_one_editor(self) -> None:
        items = _split(_plan_fields('object'))

        (plan,) = items
        assert isinstance(plan, _Plan)
        assert len(plan.parameters) == 3

    def test_the_run_must_be_in_order(self) -> None:
        start, stop, stride = _plan_fields('object')
        items = _split([stop, start, stride])

        assert [type(item) for item in items] == [_Leaf, _Leaf, _Leaf]

    def test_the_run_must_share_one_branch(self) -> None:
        start, stop, _ = _plan_fields('object')
        items = _split([start, stop, _count('probe.stride')])

        assert [type(item) for item in items] == [_Leaf, _Leaf, _Leaf]

    def test_a_plan_inside_a_box_folds_too(self) -> None:
        items = _split([_flag('l1.enabled', gates_group=True), *_plan_fields('l1')])

        (box,) = items
        assert isinstance(box, _Box)
        assert [type(item) for item in box.items] == [_Plan]

    def test_consecutive_plans_each_fold(self) -> None:
        items = _split([*_plan_fields('object'), *_plan_fields('probe')])

        assert [type(item) for item in items] == [_Plan, _Plan]


class TestHowAChoiceIsSpelled:
    def test_an_enum_persisted_by_name_offers_names(self) -> None:
        field = _field(
            'optimizer',
            StringParameter('SGD', None),
            annotation=Optimizers,
            spelling=EnumSpelling.BY_NAME,
        )

        assert list(_choices(field)) == ['SGD', 'ADAM']

    def test_an_enum_persisted_by_value_offers_values(self) -> None:
        field = _field(
            'optimizer',
            StringParameter('sgd', None),
            annotation=Optimizers,
            spelling=EnumSpelling.BY_VALUE,
        )

        assert list(_choices(field)) == ['sgd', 'adam']

    def test_an_optional_enum_is_looked_through(self) -> None:
        field = _field(
            'optimizer',
            StringParameter('SGD', None),
            annotation=Optimizers | None,
            spelling=EnumSpelling.BY_NAME,
        )

        assert list(_choices(field)) == ['SGD', 'ADAM']

    def test_the_policy_wins_where_it_supplies_a_list(self) -> None:
        """A declaration typing a choice as a bare string has nothing to offer."""
        field = _field('method', StringParameter('GD', None), choices=('GD', 'CG'))

        assert list(_choices(field)) == ['GD', 'CG']

    def test_a_plain_string_offers_nothing(self) -> None:
        assert list(_choices(_field('name', StringParameter('x', None)))) == []


class TestTitlingAGroup:
    def test_the_backend_name_is_stripped(self) -> None:
        assert _group_label('PtyChiObject', 'pty-chi') == 'Object'

    def test_a_group_that_is_only_the_backend_name_keeps_it(self) -> None:
        assert _group_label('PtyChi', 'pty-chi') == 'PtyChi'

    def test_a_group_naming_another_backend_is_left_alone(self) -> None:
        assert _group_label('PtychoFMInference', 'pty-chi') == 'PtychoFMInference'

    def test_the_algorithm_name_is_stripped_where_the_backend_name_is_absent(self) -> None:
        assert _group_label('LSQMLProbe', 'pty-chi', 'LSQML') == 'Probe'

    def test_a_group_that_is_only_the_algorithm_name_keeps_it(self) -> None:
        assert _group_label('LSQML', 'pty-chi', 'LSQML') == 'LSQML'


class TestWhichEditorAFieldGets:
    def test_a_bounded_real_slides(self, qapp: QApplication) -> None:
        field = _field('x', RealParameter(0.5, None, minimum=0.0, maximum=1.0))

        assert isinstance(_editor(field), DecimalSliderParameterViewController)

    def test_a_real_bounded_on_one_side_only_does_not(self, qapp: QApplication) -> None:
        field = _field('x', RealParameter(0.5, None, minimum=0.0))

        assert isinstance(_editor(field), DecimalLineEditParameterViewController)

    def test_a_length_gets_a_unit_ladder_whatever_its_bounds(self, qapp: QApplication) -> None:
        """A slider over a range spanning decades makes the small end unreachable."""
        field = _field(
            'period_x_m', RealParameter(1e-7, None, minimum=0.0, maximum=1.0), unit=Unit.METER
        )
        editor = _editor(field)

        assert isinstance(editor, LengthParameterViewController)

    def test_a_zero_length_reads_in_its_declared_display_unit(self, qapp: QApplication) -> None:
        def shown_unit(display_unit: LengthUnit | None) -> str:
            field = _field(
                'defocus_m',
                RealParameter(0.0, None, minimum=-1.0, maximum=1.0),
                unit=Unit.METER,
                display_unit=display_unit,
            )
            editor = _editor(field)
            assert editor is not None
            combo_box = editor.get_widget().findChild(QComboBox)
            assert combo_box is not None
            return combo_box.currentText()

        assert shown_unit(LengthUnit.MICROMETER) == LengthUnit.MICROMETER.label
        assert shown_unit(None) == LengthUnit.METER.label

    def test_an_angle_held_in_turns_does_too(self, qapp: QApplication) -> None:
        field = _field('phase_tr', RealParameter(0.25, None), unit=Unit.TURN)

        assert isinstance(_editor(field), AngleParameterViewController)

    def test_a_unit_no_controller_reads_falls_back(self, qapp: QApplication) -> None:
        """Degrees are not turns; a widget holding turns would rescale the value."""
        field = _field('tilt_deg', RealParameter(30.0, None), unit=Unit.DEGREE)

        assert isinstance(_editor(field), DecimalLineEditParameterViewController)

    def test_a_floored_integer_counts(self, qapp: QApplication) -> None:
        assert isinstance(_editor(_count('n', minimum=0)), SpinBoxParameterViewController)

    def test_an_unbounded_integer_does_not(self, qapp: QApplication) -> None:
        assert isinstance(_editor(_count('n')), IntegerLineEditParameterViewController)

    def test_a_string_with_choices_is_a_combo(self, qapp: QApplication) -> None:
        field = _field('method', StringParameter('GD', None), choices=('GD', 'CG'))

        assert isinstance(_editor(field), ComboBoxParameterViewController)

    def test_a_string_without_them_is_free_text(self, qapp: QApplication) -> None:
        field = _field('name', StringParameter('x', None))

        assert isinstance(_editor(field), LineEditParameterViewController)

    def test_a_combo_offers_the_spelling_the_file_holds(self, qapp: QApplication) -> None:
        """Offering the other spelling would write a value nothing reads back."""
        field = _field(
            'optimizer',
            StringParameter('SGD', None),
            annotation=Optimizers,
            spelling=EnumSpelling.BY_NAME,
        )
        editor = _editor(field)
        assert isinstance(editor, ComboBoxParameterViewController)
        combo = editor.get_widget()
        assert isinstance(combo, QComboBox)

        assert [combo.itemText(i) for i in range(combo.count())] == ['SGD', 'ADAM']
        assert combo.currentText() == 'SGD'


class TestAWholeNumberNamingASubset:
    def test_each_name_is_a_row(self, qapp: QApplication) -> None:
        model = _BitFieldListModel(IntegerParameter(0, None), ('X', 'Y', 'Z'))

        assert model.rowCount() == 3
        assert model.data(model.index(1, 0), Qt.ItemDataRole.DisplayRole) == 'Y'

    def test_a_set_bit_reads_as_checked(self, qapp: QApplication) -> None:
        model = _BitFieldListModel(IntegerParameter(0b101, None), ('X', 'Y', 'Z'))
        states = [
            model.data(model.index(row, 0), Qt.ItemDataRole.CheckStateRole) for row in range(3)
        ]

        assert states == [Qt.CheckState.Checked, Qt.CheckState.Unchecked, Qt.CheckState.Checked]

    def test_checking_a_row_sets_its_bit(self, qapp: QApplication) -> None:
        parameter = IntegerParameter(0b001, None)
        model = _BitFieldListModel(parameter, ('X', 'Y', 'Z'))

        model.setData(model.index(2, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)

        assert parameter.get_value() == 0b101

    def test_unchecking_clears_only_that_bit(self, qapp: QApplication) -> None:
        parameter = IntegerParameter(0b111, None)
        model = _BitFieldListModel(parameter, ('X', 'Y', 'Z'))

        model.setData(model.index(1, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)

        assert parameter.get_value() == 0b101

    def test_an_invalid_index_writes_nothing(self, qapp: QApplication) -> None:
        parameter = IntegerParameter(0b010, None)
        model = _BitFieldListModel(parameter, ('X', 'Y', 'Z'))

        assert not model.setData(
            QModelIndex(), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
        )
        assert parameter.get_value() == 0b010
