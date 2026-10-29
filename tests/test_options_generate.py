"""Unit tests for building live settings parameters from a declared model."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import pytest

from ptychodus.model.parameters import (
    BooleanParameter,
    Parameter,
    IntegerParameter,
    PathParameter,
    RealParameter,
    StringParameter,
)
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.processing.flatten import (
    EnumSpelling,
    FlatteningPolicy,
    Overlay,
    Sentinel,
)
from ptychodus.model.processing.generate import (
    UnsupportedParameterError,
    populate_parameter_group,
    restore_options,
)


class Mode(StrEnum):
    FAST = 'fast'
    EXACT = 'exact'


class Declared(BaseModel):
    enabled: bool = True
    built: float = Field(default_factory=lambda: 0.25)
    count: int = Field(4, ge=1, le=16)
    rate: float = Field(0.5, ge=0.0)
    label: str = 'untitled'
    mode: Mode = Mode.FAST
    output: Path = Path('/path/to/out')
    stop: int | None = -1


class Inner(BaseModel):
    weight: float = 1.0


class Outer(BaseModel):
    inner: Inner = Inner()


class Unsupported(BaseModel):
    weights: dict[str, float] = Field(default_factory=dict)


@pytest.fixture
def built() -> dict[str, object]:
    registry = SettingsRegistry()
    return dict(
        populate_parameter_group(registry.create_group('Declared'), Declared(), FlatteningPolicy())
    )


class TestParameterKinds:
    def test_each_declared_type_becomes_its_matching_parameter(
        self, built: dict[str, object]
    ) -> None:
        assert isinstance(built['Enabled'], BooleanParameter)
        assert isinstance(built['Count'], IntegerParameter)
        assert isinstance(built['Rate'], RealParameter)
        assert isinstance(built['Label'], StringParameter)
        assert isinstance(built['Output'], PathParameter)

    def test_bool_is_not_mistaken_for_int(self, built: dict[str, object]) -> None:
        """bool subclasses int, so the order the kinds are tested in matters."""
        assert not isinstance(built['Enabled'], IntegerParameter)

    def test_declared_bounds_reach_the_live_parameter(self, built: dict[str, object]) -> None:
        count = built['Count']
        assert isinstance(count, IntegerParameter)
        assert count.get_minimum() == 1
        assert count.get_maximum() == 16

    def test_a_zero_lower_bound_survives(self, built: dict[str, object]) -> None:
        """ge=0.0 is falsy and the commonest bound here, so it is easily dropped."""
        rate = built['Rate']
        assert isinstance(rate, RealParameter)
        assert rate.get_minimum() == 0.0

    def test_optional_annotations_use_their_inner_type(self, built: dict[str, object]) -> None:
        assert isinstance(built['Stop'], IntegerParameter)

    def test_a_default_factory_value_is_resolved(self, built: dict[str, object]) -> None:
        """The reason the declaration is passed as an instance.

        A field built by a factory reports its class-level default as undefined,
        so reading defaults off the class would store that sentinel instead of
        the value the field actually starts at.
        """
        value = built['Built']
        assert isinstance(value, RealParameter)
        assert value.get_value() == 0.25


class TestEnums:
    def test_enum_persists_by_member_name_by_default(self, built: dict[str, object]) -> None:
        mode = built['Mode']
        assert isinstance(mode, StringParameter)
        assert mode.get_value() == 'FAST'

    def test_enum_persists_by_value_under_a_verbatim_policy(self) -> None:
        registry = SettingsRegistry()
        built = populate_parameter_group(
            registry.create_group('Declared'),
            Declared(),
            FlatteningPolicy(spelling=EnumSpelling.BY_VALUE),
        )
        assert built['Mode'].get_value() == 'fast'


class TestRegistryIntegration:
    def test_the_group_is_registered_under_its_name(self) -> None:
        registry = SettingsRegistry()
        populate_parameter_group(registry.create_group('Declared'), Declared(), FlatteningPolicy())
        assert 'Declared' in set(registry)

    def test_parameters_are_live_and_observable(self) -> None:
        registry = SettingsRegistry()
        built = populate_parameter_group(
            registry.create_group('Declared'), Declared(), FlatteningPolicy()
        )
        count = built['Count']
        count.set_value(9)
        assert registry['Declared'].parameters()['Count'].get_value_as_string() == '9'

    def test_bounds_clamp_a_value_loaded_from_file(self) -> None:
        registry = SettingsRegistry()
        built = populate_parameter_group(
            registry.create_group('Declared'), Declared(), FlatteningPolicy()
        )
        built['Count'].set_value_from_string('999')
        assert built['Count'].get_value() == 16


class TestUnrepresentableFields:
    def test_an_unsupported_type_is_rejected_by_name(self) -> None:
        registry = SettingsRegistry()
        with pytest.raises(UnsupportedParameterError, match='weights'):
            populate_parameter_group(
                registry.create_group('Unsupported'), Unsupported(), FlatteningPolicy()
            )

    def test_an_adapted_field_round_trips_through_its_keys(self) -> None:
        """A declared value is a function of every key its adapter owns."""
        registry = SettingsRegistry()
        policy = FlatteningPolicy(structural={'stop': Sentinel('Stop', sentinel=-1)})
        parameters = populate_parameter_group(registry.create_group('Declared'), Declared(), policy)
        parameters['Stop'].set_value(-1)
        assert restore_options(Declared(), parameters, policy).stop is None
        parameters['Stop'].set_value(12)
        assert restore_options(Declared(), parameters, policy).stop == 12


class TestRestoringOptions:
    """Edits made against the flat live parameters must reach the nested object."""

    def _round_trip(
        self, policy: FlatteningPolicy | None = None
    ) -> tuple[Declared, Mapping[str, Parameter[Any]], FlatteningPolicy]:
        registry = SettingsRegistry()
        defaults = Declared()
        used = policy or FlatteningPolicy()
        parameters = populate_parameter_group(registry.create_group('Declared'), defaults, used)
        return defaults, parameters, used

    def test_an_untouched_round_trip_reproduces_the_defaults(self) -> None:
        defaults, parameters, policy = self._round_trip()
        assert restore_options(defaults, parameters, policy) == defaults

    def test_an_edited_parameter_reaches_the_restored_object(self) -> None:
        defaults, parameters, policy = self._round_trip()
        parameters['Count'].set_value(11)
        assert restore_options(defaults, parameters, policy).count == 11

    def test_an_enum_is_restored_from_its_member_name(self) -> None:
        defaults, parameters, policy = self._round_trip()
        parameters['Mode'].set_value('EXACT')
        assert restore_options(defaults, parameters, policy).mode is Mode.EXACT

    def test_an_enum_is_restored_from_its_value_under_a_verbatim_policy(self) -> None:
        defaults, parameters, policy = self._round_trip(
            FlatteningPolicy(spelling=EnumSpelling.BY_VALUE)
        )
        parameters['Mode'].set_value('exact')
        assert restore_options(defaults, parameters, policy).mode is Mode.EXACT

    def test_the_defaults_are_not_mutated_by_restoring(self) -> None:
        """The store keeps the defaults as its record of where each field started."""
        defaults, parameters, policy = self._round_trip()
        parameters['Count'].set_value(11)
        restore_options(defaults, parameters, policy)
        assert defaults.count == 4

    def test_a_nested_edit_reaches_the_nested_object(self) -> None:
        registry = SettingsRegistry()
        defaults = Outer()
        policy = FlatteningPolicy(prefixes={'inner': 'Inner'})
        parameters = populate_parameter_group(registry.create_group('Outer'), defaults, policy)
        parameters['InnerWeight'].set_value(7.5)
        assert restore_options(defaults, parameters, policy).inner.weight == 7.5


class TestAValueTheDeclarationDoesNotAccept:
    """A settings file is hand-edited and outlives the version that wrote it."""

    def _bind(self, policy: FlatteningPolicy) -> tuple[Declared, Mapping[str, Any]]:
        registry = SettingsRegistry()
        declared = Declared()
        parameters = populate_parameter_group(registry.create_group('Declared'), declared, policy)
        return declared, parameters

    def test_a_choice_outside_the_list_falls_back_and_says_so(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        policy = FlatteningPolicy(overlays={'label': Overlay(choices=('untitled', 'named'))})
        declared, parameters = self._bind(policy)
        parameters['Label'].set_value_from_string('bogus')

        assert restore_options(declared, parameters, policy).label == 'untitled'
        assert 'bogus' in caplog.text
        assert 'Label' in caplog.text

    def test_an_enum_member_that_no_longer_exists_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A member removed upstream leaves the name it was persisted under."""
        policy = FlatteningPolicy()
        declared, parameters = self._bind(policy)
        parameters['Mode'].set_value_from_string('RETIRED')

        assert restore_options(declared, parameters, policy).mode is Mode.FAST
        assert 'RETIRED' in caplog.text

    def test_an_adapted_key_is_checked_too(self, caplog: pytest.LogCaptureFixture) -> None:
        """A Sentinel bypasses the ordinary restore path, so it needs its own check."""
        policy = FlatteningPolicy(
            structural={'label': Sentinel('Label', value_type=str, sentinel='none')},
            overlays={'Label': Overlay(choices=('none', 'named'))},
        )
        declared, parameters = self._bind(policy)
        parameters['Label'].set_value_from_string('bogus')

        restore_options(declared, parameters, policy)
        assert 'bogus' in caplog.text

    def test_a_value_within_the_list_is_left_alone(self) -> None:
        policy = FlatteningPolicy(overlays={'label': Overlay(choices=('untitled', 'named'))})
        declared, parameters = self._bind(policy)
        parameters['Label'].set_value_from_string('named')

        assert restore_options(declared, parameters, policy).label == 'named'
