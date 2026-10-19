"""Selecting a reconstructor, and what happens when the selection cannot resolve.

A settings file routinely names a backend the current install does not have --
one machine has pty-chi, another does not, and the same file is carried between
them. That is the normal case rather than an error, so the selection has to
survive being unresolvable: a bare install that rewrote the key would lose the
user's choice the first time they opened ptychodus on it.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel
import pytest

from ptychodus.api.reconstruct import (
    NullReconstructor,
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing.api import ProcessingAlgorithmParameter


class _Reconstructor(Reconstructor):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        yield from ()


class _Library(ReconstructorLibrary):
    def __init__(self, name: str, *reconstructors: Reconstructor) -> None:
        super().__init__(name, name)
        self._name = name
        self._built = reconstructors

    @property
    def name(self) -> str:
        return self._name

    def _build(self) -> tuple[Reconstructor, ...]:
        return self._built


@pytest.fixture
def selection() -> tuple[ProcessingAlgorithmParameter, SettingsRegistry]:
    registry = SettingsRegistry()
    parameter = registry.create_group('Processing').create_string_parameter('Algorithm', 'lib_algo')
    library = _Library('lib', _Reconstructor('algo'))
    return ProcessingAlgorithmParameter(parameter, [library]), registry


def test_a_registered_algorithm_is_selected(
    selection: tuple[ProcessingAlgorithmParameter, SettingsRegistry],
) -> None:
    algorithm, _ = selection
    assert algorithm.get_value() == 'lib_algo'
    assert algorithm.get_current_reconstructor().name == 'algo'


def test_an_unresolvable_name_leaves_the_setting_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The key a backendless install cannot resolve is the key it must not rewrite."""
    registry = SettingsRegistry()
    parameter = registry.create_group('Processing').create_string_parameter(
        'Algorithm', 'ptychi_LSQML'
    )
    ProcessingAlgorithmParameter(parameter, [])

    assert parameter.get_value() == 'ptychi_LSQML'
    assert 'ptychi_LSQML' in caplog.text


def test_an_unresolvable_name_warns_rather_than_whispering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At the default level, a debug line is no notice at all."""
    registry = SettingsRegistry()
    parameter = registry.create_group('Processing').create_string_parameter('Algorithm', 'nope')
    ProcessingAlgorithmParameter(parameter, [])

    assert [r.levelname for r in caplog.records if 'nope' in r.getMessage()] == ['WARNING']


def test_nothing_registered_resolves_to_a_reconstructor_that_cannot_train() -> None:
    """A bare install must not offer to train a no-op."""
    registry = SettingsRegistry()
    parameter = registry.create_group('Processing').create_string_parameter('Algorithm', 'nope')
    algorithm = ProcessingAlgorithmParameter(parameter, [])

    reconstructor = algorithm.get_current_reconstructor()
    assert isinstance(reconstructor, NullReconstructor)
    assert not isinstance(reconstructor, TrainableReconstructor)
