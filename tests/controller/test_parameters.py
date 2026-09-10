"""Unit tests for the parameter view controllers.

Each binds a parameter to one or more widgets, so the parameter -- not the widget -- is
the value under test. The unit-aware controllers bind a `RealParameter` in base SI units
to a magnitude line edit plus a unit combo box.
"""

from __future__ import annotations
from decimal import Decimal
from uuid import UUID

import pytest

from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QApplication

from ptychodus.api.constants import TWO_PI, AngleUnit, LengthUnit
from ptychodus.api.parameters import RealParameter, UUIDParameter
from ptychodus.controller.parameters import (
    AngleParameterViewController,
    LengthParameterViewController,
    UUIDLineEditParameterViewController,
)


def _real(value: float, *, minimum: float | None = None) -> RealParameter:
    return RealParameter(value, None, minimum=minimum)


def _uuid(value: str = '123e4567-e89b-12d3-a456-426614174000') -> UUIDParameter:
    return UUIDParameter(UUID(value), None)


def _activate_unit(view_controller, label: str) -> None:
    """Mimic a user picking a unit: set the index, then fire `activated` as Qt would."""
    combo_box = view_controller._units_combo_box
    index = combo_box.findText(label)
    assert index != -1, f'no {label!r} entry'
    combo_box.setCurrentIndex(index)
    combo_box.activated.emit(index)


class TestLength:
    def test_combo_box_lists_units_largest_first(self, qapp: QApplication) -> None:
        vc = LengthParameterViewController(_real(0.0))
        combo_box = vc._units_combo_box
        labels = [combo_box.itemText(i) for i in range(combo_box.count())]

        assert labels == ['m', 'mm', 'µm', 'nm', 'Å', 'pm']

    def test_default_unit_is_meters(self, qapp: QApplication) -> None:
        vc = LengthParameterViewController(_real(0.0))

        assert vc._units_combo_box.currentData() is LengthUnit.METER

    def test_zero_valued_parameter_honors_the_preferred_unit(self, qapp: QApplication) -> None:
        """The defocus regression: a parameter defaulting to zero used to fall back to meters,
        so typing 800 to mean microns silently set the parameter to 800 metres."""
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter, default_unit=LengthUnit.MICROMETER)

        assert vc._units_combo_box.currentData() is LengthUnit.MICROMETER

        vc._line_edit.set_value(Decimal(800))

        assert parameter.get_value() == pytest.approx(8e-4)

    def test_without_a_preferred_unit_zero_still_means_meters(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter)
        vc._line_edit.set_value(Decimal(800))

        assert parameter.get_value() == pytest.approx(800.0)

    def test_non_zero_parameter_auto_selects_a_unit(self, qapp: QApplication) -> None:
        vc = LengthParameterViewController(_real(1.5e-9))

        assert vc._units_combo_box.currentData() is LengthUnit.NANOMETER
        assert vc._line_edit.get_value() == Decimal('1.5')

    def test_auto_selection_overrides_the_preferred_unit(self, qapp: QApplication) -> None:
        vc = LengthParameterViewController(_real(2e-3), default_unit=LengthUnit.MICROMETER)

        assert vc._units_combo_box.currentData() is LengthUnit.MILLIMETER

    def test_model_change_rescales_the_display(self, qapp: QApplication) -> None:
        parameter = _real(1e-9)
        vc = LengthParameterViewController(parameter)
        parameter.set_value(2e-3)

        assert vc._units_combo_box.currentData() is LengthUnit.MILLIMETER
        assert vc._line_edit.get_value() == Decimal(2)

    def test_an_explicit_unit_choice_sticks(self, qapp: QApplication) -> None:
        """Picking mm then typing 0.5 must not rewrite the field to 500 µm."""
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter)
        _activate_unit(vc, 'mm')
        vc._line_edit.set_value(Decimal('0.5'))

        assert parameter.get_value() == pytest.approx(5e-4)
        assert vc._units_combo_box.currentData() is LengthUnit.MILLIMETER
        assert vc._line_edit.get_value() == Decimal('0.5')

    def test_an_explicit_choice_survives_a_model_change(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter)
        _activate_unit(vc, 'mm')
        parameter.set_value(2e-6)

        assert vc._units_combo_box.currentData() is LengthUnit.MILLIMETER
        assert vc._line_edit.get_value() == Decimal('0.002')

    def test_changing_the_unit_preserves_the_physical_length(self, qapp: QApplication) -> None:
        parameter = _real(2e-3)
        vc = LengthParameterViewController(parameter)
        _activate_unit(vc, 'µm')

        assert parameter.get_value() == pytest.approx(2e-3)
        assert vc._line_edit.get_value() == Decimal(2000)

    def test_signedness_is_inferred_from_an_absent_minimum(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter, default_unit=LengthUnit.MICROMETER)
        vc._line_edit.set_value(Decimal(-800))

        assert parameter.get_value() == pytest.approx(-8e-4)

    def test_signedness_is_inferred_from_a_zero_minimum(self, qapp: QApplication) -> None:
        parameter = _real(0.0, minimum=0.0)
        vc = LengthParameterViewController(parameter, default_unit=LengthUnit.MICROMETER)
        vc._line_edit.set_value(Decimal(-800))

        assert parameter.get_value() == pytest.approx(0.0)

    def test_an_explicit_flag_overrides_inference(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = LengthParameterViewController(parameter, is_signed=False)
        vc._line_edit.set_value(Decimal(-5))

        assert parameter.get_value() == pytest.approx(0.0)

    def test_meter_unit_does_not_add_a_trailing_zero(self, qapp: QApplication) -> None:
        vc = LengthParameterViewController(_real(0.0))
        vc._line_edit.set_value(Decimal('1.5'))

        assert str(vc._line_edit.get_value()) == '1.5'

    def test_non_finite_parameter_does_not_raise(self, qapp: QApplication) -> None:
        """Decimal('NaN').is_zero() is False, so NaN reaches the unit-selection path."""
        vc = LengthParameterViewController(_real(float('nan')))

        assert vc._units_combo_box.currentData() is LengthUnit.METER


class TestAngle:
    def test_combo_box_lists_the_angle_units(self, qapp: QApplication) -> None:
        vc = AngleParameterViewController(_real(0.0))
        combo_box = vc._units_combo_box
        labels = [combo_box.itemText(i) for i in range(combo_box.count())]

        assert labels == ['turn', 'deg', 'rad']

    def test_default_unit_is_turns(self, qapp: QApplication) -> None:
        vc = AngleParameterViewController(_real(0.0))

        assert vc._units_combo_box.currentData() is AngleUnit.TURN

    def test_preferred_unit_is_honored(self, qapp: QApplication) -> None:
        vc = AngleParameterViewController(_real(0.0), default_unit=AngleUnit.DEGREE)

        assert vc._units_combo_box.currentData() is AngleUnit.DEGREE

    def test_degrees_round_trip_exactly(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = AngleParameterViewController(parameter, default_unit=AngleUnit.DEGREE)
        vc._line_edit.set_value(Decimal(90))

        assert parameter.get_value() == 0.25

    def test_radians_round_trip(self, qapp: QApplication) -> None:
        parameter = _real(0.0)
        vc = AngleParameterViewController(parameter, default_unit=AngleUnit.RADIAN)
        vc._line_edit.set_value(Decimal.from_float(TWO_PI))

        assert parameter.get_value() == pytest.approx(1.0)

    def test_the_unit_is_never_auto_selected(self, qapp: QApplication) -> None:
        """turn/deg/rad is a presentation choice, not a magnitude ladder."""
        parameter = _real(0.001)
        vc = AngleParameterViewController(parameter)

        assert vc._units_combo_box.currentData() is AngleUnit.TURN

        parameter.set_value(0.25)

        assert vc._units_combo_box.currentData() is AngleUnit.TURN

    def test_changing_the_unit_preserves_the_angle(self, qapp: QApplication) -> None:
        parameter = _real(0.25)
        vc = AngleParameterViewController(parameter)
        _activate_unit(vc, 'deg')

        assert parameter.get_value() == 0.25
        assert vc._line_edit.get_value() == Decimal(90)

    def test_angles_are_signed_when_the_parameter_has_no_minimum(self, qapp: QApplication) -> None:
        """The angle editor used to hardcode unsigned, so a negative shift was unenterable."""
        parameter = _real(0.25)
        vc = AngleParameterViewController(parameter)
        vc._line_edit.set_value(Decimal('-0.25'))

        assert parameter.get_value() == -0.25


class TestUUID:
    def test_validator_accepts_a_well_formed_uuid(self, qapp: QApplication) -> None:
        view_controller = UUIDLineEditParameterViewController(_uuid())
        validator = view_controller._widget.validator()
        assert validator is not None

        state, _, _ = validator.validate('7c9e6679-7425-40de-944b-e07fc1f90ae7', 0)
        assert state == QValidator.State.Acceptable

    def test_validator_marks_a_truncated_uuid_intermediate(self, qapp: QApplication) -> None:
        """A partial UUID is still editable, which is why the model sync needs a guard."""
        view_controller = UUIDLineEditParameterViewController(_uuid())
        validator = view_controller._widget.validator()
        assert validator is not None

        state, _, _ = validator.validate('7c9e6679-7425-40de', 0)
        assert state == QValidator.State.Intermediate

    def test_validator_rejects_non_hex_text(self, qapp: QApplication) -> None:
        view_controller = UUIDLineEditParameterViewController(_uuid())
        validator = view_controller._widget.validator()
        assert validator is not None

        state, _, _ = validator.validate('not-a-uuid', 0)
        assert state == QValidator.State.Invalid

    def test_bad_text_leaves_the_parameter_unchanged(self, qapp: QApplication) -> None:
        """Regression: the view-to-model sync used to raise ValueError out of a Qt slot."""
        parameter = _uuid()
        view_controller = UUIDLineEditParameterViewController(parameter)
        original = parameter.get_value()

        view_controller._widget.setText('7c9e6679-7425-40de')
        view_controller._widget.editingFinished.emit()

        assert parameter.get_value() == original

    def test_round_trip(self, qapp: QApplication) -> None:
        parameter = _uuid()
        view_controller = UUIDLineEditParameterViewController(parameter)
        assert view_controller._widget.text() == '123e4567-e89b-12d3-a456-426614174000'

        parameter.set_value(UUID('7c9e6679-7425-40de-944b-e07fc1f90ae7'))
        assert view_controller._widget.text() == '7c9e6679-7425-40de-944b-e07fc1f90ae7'

        view_controller._widget.setText('123e4567-e89b-12d3-a456-426614174000')
        view_controller._widget.editingFinished.emit()
        assert parameter.get_value() == UUID('123e4567-e89b-12d3-a456-426614174000')
