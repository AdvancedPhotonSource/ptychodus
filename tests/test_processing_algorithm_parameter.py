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
from ptychodus.api.parameters import StringParameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
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
    def __init__(self, distribution: str, *reconstructors: Reconstructor) -> None:
        super().__init__(distribution, distribution)
        self._built = reconstructors

    def _build(self) -> tuple[Reconstructor, ...]:
        return self._built


def _selection(
    algorithm: str,
    *libraries: tuple[str, _Library],
    library: str = '',
) -> tuple[ProcessingAlgorithmParameter, StringParameter, StringParameter]:
    """Build the two live selections over *libraries*, as ProcessingCore does."""
    chooser = PluginChooser[ReconstructorLibrary]()

    for name, offered in libraries:
        chooser.register_plugin(offered, simple_name=name, display_name=name)

    group = SettingsRegistry().create_group('Processing')
    library_setting = group.create_string_parameter('Library', library)
    algorithm_setting = group.create_string_parameter('Algorithm', algorithm)
    library_parameter = PluginChooserParameter(chooser, library_setting)

    return (
        ProcessingAlgorithmParameter(algorithm_setting, library_parameter),
        library_setting,
        algorithm_setting,
    )


def test_a_registered_algorithm_is_selected() -> None:
    algorithm, _, _ = _selection('algo', ('lib', _Library('lib', _Reconstructor('algo'))))

    assert algorithm.get_value() == 'algo'
    assert algorithm.get_current_reconstructor().name == 'algo'
    assert algorithm.get_current_parts() == ('lib', 'algo')


def test_a_name_is_resolved_case_insensitively_and_written_back_canonical() -> None:
    """A hand-edited file should work, and should be tidied where it is read."""
    algorithm, _, setting = _selection('LSQML', ('lib', _Library('lib', _Reconstructor('lsqml'))))

    assert algorithm.get_value() == 'lsqml'
    assert setting.get_value() == 'lsqml'


def test_an_unresolvable_name_leaves_the_setting_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The key a backendless install cannot resolve is the key it must not rewrite."""
    _, _, setting = _selection('ptychi_LSQML')

    assert setting.get_value() == 'ptychi_LSQML'


def test_an_unresolvable_name_warns_rather_than_whispering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At the default level, a debug line is no notice at all."""
    _selection('nope', ('lib', _Library('lib', _Reconstructor('algo'))))

    assert [r.levelname for r in caplog.records if 'nope' in r.getMessage()] == ['WARNING']


def test_an_unresolvable_name_still_leaves_something_usable() -> None:
    """Warned about, but stood in for, so the application is not left inert.

    Not written back, though: the file keeps the choice it carried until the
    user makes a real one.
    """
    algorithm, _, setting = _selection('nope', ('lib', _Library('lib', _Reconstructor('algo'))))

    assert algorithm.get_value() == 'algo'
    assert setting.get_value() == 'nope'


def test_a_library_offering_nothing_resolves_to_a_reconstructor_that_cannot_train() -> None:
    """A bare install must not offer to train a no-op.

    A backend registers whether or not its packages are installed, so the
    empty case is a library that offers nothing rather than no library.
    """
    algorithm, _, _ = _selection('nope', ('lib', _Library('lib')))

    reconstructor = algorithm.get_current_reconstructor()
    assert isinstance(reconstructor, NullReconstructor)
    assert not isinstance(reconstructor, TrainableReconstructor)


def test_no_backend_at_all_is_survivable() -> None:
    """Nothing registers in this test; production always registers something."""
    algorithm, _, _ = _selection('nope')

    assert algorithm.get_value() == ''
    assert list(algorithm.choices()) == []
    assert isinstance(algorithm.get_current_reconstructor(), NullReconstructor)


def test_the_choices_are_the_selected_library_s_algorithms() -> None:
    algorithm, library_setting, _ = _selection(
        'one',
        ('lib', _Library('lib', _Reconstructor('one'), _Reconstructor('two'))),
        ('other', _Library('other', _Reconstructor('three'))),
        library='lib',
    )

    assert list(algorithm.choices()) == ['one', 'two']

    library_setting.set_value('other')
    assert list(algorithm.choices()) == ['three']


def test_changing_library_settles_on_one_of_its_own_algorithms() -> None:
    algorithm, library_setting, algorithm_setting = _selection(
        'one',
        ('lib', _Library('lib', _Reconstructor('one'))),
        ('other', _Library('other', _Reconstructor('three'))),
        library='lib',
    )

    library_setting.set_value('other')

    assert algorithm.get_value() == 'three'
    assert algorithm_setting.get_value() == 'three'
    assert algorithm.get_current_parts() == ('other', 'three')


def test_returning_to_a_library_returns_to_what_was_chosen_there() -> None:
    """Flipping between backends to compare them must not lose the selection."""
    algorithm, library_setting, _ = _selection(
        'one',
        ('lib', _Library('lib', _Reconstructor('one'), _Reconstructor('two'))),
        ('other', _Library('other', _Reconstructor('three'))),
        library='lib',
    )
    algorithm.set_value('two')

    library_setting.set_value('other')
    library_setting.set_value('lib')

    assert algorithm.get_value() == 'two'


def test_the_two_keys_resolve_whichever_order_they_arrive_in() -> None:
    """A settings file is applied key by key, in an order this cannot control.

    The algorithm may land while the previous library is still selected, so
    it has to be re-resolved from what is persisted once the library catches
    up -- not from whatever it settled on in the meantime.
    """
    algorithm, library_setting, algorithm_setting = _selection(
        'one',
        ('lib', _Library('lib', _Reconstructor('one'))),
        ('other', _Library('other', _Reconstructor('three'), _Reconstructor('four'))),
        library='lib',
    )

    # Algorithm first, naming one the current library does not offer.
    algorithm_setting.set_value('four')
    # Library second, which is when 'four' becomes resolvable.
    library_setting.set_value('other')

    assert algorithm.get_value() == 'four'
