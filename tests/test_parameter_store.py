"""Unit tests for the live settings behind each reconstructor's declaration."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field
import pytest

from ptychodus.api.reconstruct import (
    NoOptions,
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
)
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing.flatten import FlatteningPolicy
from ptychodus.model.processing.parameter_store import (
    OptionsSection,
    ProcessingParameterStore,
)


class ObjectBranch(BaseModel):
    weight: float = Field(1.0, ge=0.0)
    slices: int = 1


class ProbeBranch(BaseModel):
    power: float = 2.0


class Declared(BaseModel):
    epochs: int = 10
    object_: ObjectBranch = ObjectBranch()
    probe: ProbeBranch = ProbeBranch()


class FakeReconstructor(Reconstructor):
    """A reconstructor is only ever asked for its name and its declaration here."""

    def __init__(self, name: str, options: BaseModel) -> None:
        self._name = name
        self._options = options

    @property
    def name(self) -> str:
        return self._name

    def get_progress_goal(self) -> int:
        return 0

    def get_default_options(self) -> BaseModel:
        return self._options

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        yield from ()


@pytest.fixture
def registry() -> SettingsRegistry:
    return SettingsRegistry()


class TestUndeclaredReconstructors:
    def test_a_reconstructor_declaring_nothing_creates_no_settings_group(
        self, registry: SettingsRegistry
    ) -> None:
        """An empty section would appear in every saved settings file and mean nothing."""
        store = ProcessingParameterStore(registry)
        store.register('lib_algo', FakeReconstructor('algo', NoOptions()), [OptionsSection('Algo')])
        assert list(registry) == []

    def test_its_options_are_still_served(self, registry: SettingsRegistry) -> None:
        store = ProcessingParameterStore(registry)
        reconstructor = FakeReconstructor('algo', NoOptions())
        store.register('lib_algo', reconstructor, [OptionsSection('Algo')])
        assert isinstance(store.get_options('lib_algo'), NoOptions)


class TestSingleSection:
    def test_the_whole_declaration_becomes_one_group(self, registry: SettingsRegistry) -> None:
        store = ProcessingParameterStore(registry)
        policy = FlatteningPolicy(prefixes={'object_': 'Object', 'probe': 'Probe'})
        store.register(
            'lib_algo',
            FakeReconstructor('algo', Declared()),
            [OptionsSection('Algo', policy=policy)],
        )
        keys = set(registry['Algo'].parameters())
        assert keys == {'Epochs', 'ObjectWeight', 'ObjectSlices', 'ProbePower'}

    def test_an_edit_reaches_the_served_options(self, registry: SettingsRegistry) -> None:
        store = ProcessingParameterStore(registry)
        policy = FlatteningPolicy(prefixes={'object_': 'Object', 'probe': 'Probe'})
        store.register(
            'lib_algo',
            FakeReconstructor('algo', Declared()),
            [OptionsSection('Algo', policy=policy)],
        )
        registry['Algo'].parameters()['ObjectWeight'].set_value(4.5)
        assert store.get_options('lib_algo').object_.weight == 4.5


class TestSeveralSections:
    """pty-chi persists one declaration as fifteen groups; this is that shape."""

    def _store(self, registry: SettingsRegistry) -> ProcessingParameterStore:
        store = ProcessingParameterStore(registry)
        store.register(
            'lib_algo',
            FakeReconstructor('algo', Declared()),
            [
                OptionsSection(
                    'Algo', policy=FlatteningPolicy(hidden=frozenset({'object_', 'probe'}))
                ),
                OptionsSection('AlgoObject', root_path='object_'),
                OptionsSection('AlgoProbe', root_path='probe'),
            ],
        )
        return store

    def test_each_section_becomes_its_own_group(self, registry: SettingsRegistry) -> None:
        self._store(registry)
        assert set(registry) == {'Algo', 'AlgoObject', 'AlgoProbe'}

    def test_a_rooted_section_names_keys_relative_to_its_root(
        self, registry: SettingsRegistry
    ) -> None:
        self._store(registry)
        assert set(registry['AlgoObject'].parameters()) == {'Weight', 'Slices'}

    def test_edits_in_every_section_reach_one_served_object(
        self, registry: SettingsRegistry
    ) -> None:
        store = self._store(registry)
        registry['Algo'].parameters()['Epochs'].set_value(99)
        registry['AlgoObject'].parameters()['Weight'].set_value(4.5)
        registry['AlgoProbe'].parameters()['Power'].set_value(8.0)
        options = store.get_options('lib_algo')
        assert (options.epochs, options.object_.weight, options.probe.power) == (99, 4.5, 8.0)

    def test_the_defaults_survive_repeated_reads(self, registry: SettingsRegistry) -> None:
        store = self._store(registry)
        registry['AlgoObject'].parameters()['Weight'].set_value(4.5)
        store.get_options('lib_algo')
        registry['AlgoObject'].parameters()['Weight'].set_value(6.0)
        assert store.get_options('lib_algo').object_.weight == 6.0


class TestLookup:
    def test_an_unregistered_key_is_rejected_by_name(self, registry: SettingsRegistry) -> None:
        store = ProcessingParameterStore(registry)
        with pytest.raises(KeyError, match='lib_missing'):
            store.get_options('lib_missing')

    def test_membership_reports_what_is_registered(self, registry: SettingsRegistry) -> None:
        store = ProcessingParameterStore(registry)
        store.register('lib_algo', FakeReconstructor('algo', NoOptions()), [])
        assert 'lib_algo' in store and 'lib_other' not in store


class TestOneGroupFromSeveralSections:
    """pty-chi's per-algorithm groups draw keys from four branches of one declaration."""

    def _store(self, registry: SettingsRegistry) -> ProcessingParameterStore:
        store = ProcessingParameterStore(registry)
        store.register(
            'lib_algo',
            FakeReconstructor('algo', Declared()),
            [
                OptionsSection(
                    'Algo',
                    root_path='object_',
                    policy=FlatteningPolicy(hidden=frozenset({'slices'})),
                ),
                OptionsSection('Algo', root_path='probe'),
            ],
        )
        return store

    def test_both_sections_land_in_one_group(self, registry: SettingsRegistry) -> None:
        self._store(registry)
        assert set(registry) == {'Algo'}
        assert set(registry['Algo'].parameters()) == {'Weight', 'Power'}

    def test_edits_from_either_section_reach_the_served_object(
        self, registry: SettingsRegistry
    ) -> None:
        store = self._store(registry)
        registry['Algo'].parameters()['Weight'].set_value(3.5)
        registry['Algo'].parameters()['Power'].set_value(9.0)
        options = store.get_options('lib_algo')
        assert (options.object_.weight, options.probe.power) == (3.5, 9.0)
