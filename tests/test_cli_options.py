"""Options documents: printed at the defaults, read back, and merged onto the defaults.

The models here are shaped like the backends' own: nested sections, defaults that differ
from the class's, a field left unset because the backend resolves it at runtime when
absent, and a section frozen against change.
"""

from __future__ import annotations

import io
from pathlib import Path

from pydantic import BaseModel, Field
import pytest
import yaml

from ptychodus.cli._options import (
    OptionsError,
    apply_options_document,
    format_options_document,
    read_options_document,
)


class _Training(BaseModel):
    epochs: int = Field(10, ge=1, description='Passes over the data.')
    batch_size: int | None = Field(None, description='Resolved at runtime when absent.')


class _Architecture(BaseModel):
    depth: int = 4


class _Options(BaseModel):
    learning_rate: float = Field(1.0e-3, gt=0.0, description='Step size.')
    training: _Training = Field(default_factory=_Training)
    architecture: _Architecture = Field(default_factory=_Architecture, frozen=True)


def _defaults() -> _Options:
    # Set explicitly and away from the class defaults, as a backend overlay does; batch_size
    # is left unset on purpose.
    return _Options(training=_Training(epochs=2), architecture=_Architecture(depth=12))


def _round_trip(defaults: BaseModel) -> BaseModel:
    document = yaml.safe_load(
        format_options_document('Fake options', {'algorithm': 'fake'}, defaults)
    )
    return apply_options_document(defaults, document['options'] or {}, subject='fake')


def test_the_printed_document_names_its_selectors_and_describes_nested_fields() -> None:
    text = format_options_document(
        'Fake options', {'library': 'lib', 'algorithm': 'fake'}, _defaults()
    )
    document = yaml.safe_load(text)

    assert document['library'] == 'lib'
    assert document['algorithm'] == 'fake'
    assert document['options']['training'] == {'epochs': 2, 'batch_size': None}
    assert '    # Passes over the data.' in text.splitlines()


def test_a_frozen_field_is_printed_only_as_a_comment() -> None:
    text = format_options_document('Fake options', {'algorithm': 'fake'}, _defaults())

    assert 'architecture' not in yaml.safe_load(text)['options']
    assert '  # architecture is fixed by the backend and cannot be set here.' in text.splitlines()


def test_an_unedited_document_reproduces_the_defaults_and_what_they_set() -> None:
    defaults = _defaults()
    options = _round_trip(defaults)

    assert options == defaults
    assert options.model_dump(exclude_unset=True) == defaults.model_dump(exclude_unset=True)
    assert 'batch_size' not in options.model_dump(exclude_unset=True)['training']


def test_a_nested_override_keeps_its_siblings_at_the_defaults() -> None:
    options = apply_options_document(_defaults(), {'training': {'batch_size': 64}}, subject='fake')

    assert isinstance(options, _Options)
    assert options.training.epochs == 2
    assert options.training.batch_size == 64
    assert options.architecture.depth == 12
    assert options.model_dump(exclude_unset=True)['training'] == {'epochs': 2, 'batch_size': 64}


def test_values_are_validated_against_the_declaration() -> None:
    with pytest.raises(OptionsError, match='Invalid options for fake'):
        apply_options_document(_defaults(), {'training': {'epochs': 0}}, subject='fake')


def test_an_undeclared_field_is_refused_at_any_depth() -> None:
    with pytest.raises(OptionsError, match=r"\['learning_rat', 'training.epoch'\]"):
        apply_options_document(
            _defaults(), {'learning_rat': 1.0, 'training': {'epoch': 3}}, subject='fake'
        )


def test_a_frozen_field_is_refused() -> None:
    with pytest.raises(OptionsError, match=r"fixes \['architecture'\]"):
        apply_options_document(_defaults(), {'architecture': {'depth': 4}}, subject='fake')


def test_a_document_is_read_from_a_file(tmp_path: Path) -> None:
    options_file = tmp_path / 'options.yaml'
    options_file.write_text('library: lib\nalgorithm: fake\noptions:\n  learning_rate: 0.5\n')

    selectors, fields = read_options_document(options_file, ('library', 'algorithm'))

    assert selectors == {'library': 'lib', 'algorithm': 'fake'}
    assert fields == {'learning_rate': 0.5}


def test_a_document_is_read_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('sys.stdin', io.StringIO('algorithm: fake\n'))

    selectors, fields = read_options_document(Path('-'), ('algorithm',))

    assert selectors == {'algorithm': 'fake'}
    assert fields == {}


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('- a list\n', 'must hold a mapping'),
        ('algorithm: fake\nextra: 1\n', 'unexpected top-level keys'),
        ('options: {}\n', 'must give "algorithm" as a string'),
        ('algorithm: fake\noptions: [1]\n', 'must be a mapping of field values'),
        ('algorithm: [unclosed\n', 'is not YAML'),
    ],
)
def test_a_malformed_document_is_refused(tmp_path: Path, text: str, message: str) -> None:
    options_file = tmp_path / 'options.yaml'
    options_file.write_text(text)

    with pytest.raises(OptionsError, match=message):
        read_options_document(options_file, ('algorithm',))
