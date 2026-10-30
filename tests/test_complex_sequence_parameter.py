"""ComplexSequenceParameter: one spelling for a settings file, a declaration and JSON.

The value is persisted the way pydantic writes a ``list[complex]`` as JSON, so what a
settings file holds is what a declared options type validates and what a JSON payload
carries. Values handed to it in code take the same inputs.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from ptychodus.model.parameters import ComplexSequenceParameter


def test_the_string_form_is_pydantic_json_and_round_trips_exactly() -> None:
    value = [complex(0.1, -1.0 / 3.0), 2.5j, -1.0 + 0j]
    parameter = ComplexSequenceParameter(value, None)
    text = parameter.get_value_as_string()

    restored = ComplexSequenceParameter([], None)
    restored.set_value_from_string(text)

    assert json.loads(text) == ['0.1-0.3333333333333333j', '2.5j', '-1+0j']
    assert restored.get_value() == value


def test_plain_numbers_and_literals_are_accepted_from_a_string() -> None:
    parameter = ComplexSequenceParameter([], None)

    parameter.set_value_from_string('[1, 0.5, "-2j"]')

    assert parameter.get_value() == [1 + 0j, 0.5 + 0j, -2j]


def test_values_set_in_code_are_converted_to_complex() -> None:
    """A workflow mapping passes values straight to set_value, never through a string."""
    parameter = ComplexSequenceParameter([1 + 0j], None)

    parameter.set_value(['1+0j', 0.5, '0.5j'])  # type: ignore[list-item]

    assert parameter.get_value() == [1 + 0j, 0.5 + 0j, 0.5j]
    assert all(isinstance(element, complex) for element in parameter.get_value())


def test_a_value_that_is_not_a_number_is_refused_in_code() -> None:
    parameter = ComplexSequenceParameter([1 + 0j], None)

    with pytest.raises(ValidationError):
        parameter.set_value(['not a number'])  # type: ignore[list-item]


def test_a_malformed_string_leaves_the_value_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    parameter = ComplexSequenceParameter([1 + 0j, 2j], None)

    with caplog.at_level(logging.WARNING):
        parameter.set_value_from_string('(1+0j),(2j)')

    assert parameter.get_value() == [1 + 0j, 2j]
    assert 'malformed complex sequence' in caplog.text


def test_a_copy_holds_its_own_list() -> None:
    parameter = ComplexSequenceParameter([1 + 0j], None)
    duplicate = parameter.copy()

    duplicate.set_value([3j])

    assert parameter.get_value() == [1 + 0j]
