"""Every registered library's options survive the trip through an options document.

A job, a prepared directory and a remote run all carry their options as a document rather
than as live objects, and the GUI hands live ones to a child process by pickle. So each
reconstructor's and enhancer's declared options must come back from either with the same
values, the same concrete types and the same set fields -- a library may resolve an unset
field at runtime, so losing set-ness would silently change what runs. Every options model
also forbids fields it does not declare, so a misspelled one is refused rather than run at
its default. A library that is not installed is skipped.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from ptychodus.api.options import apply_options_document, dump_options_document
from ptychodus.api.plugins import PluginRegistry


def _declared_options() -> list[tuple[str, BaseModel]]:
    registry = PluginRegistry.load_plugins()
    declared: list[tuple[str, BaseModel]] = []

    for plugin in registry.reconstructor_libraries:
        if plugin.strategy.is_available():
            for reconstructor in plugin.strategy:
                declared.append(
                    (
                        f'{plugin.simple_name}/{reconstructor.name}',
                        reconstructor.get_default_options(),
                    )
                )

    for enhancer in registry.fluorescence_enhancers:
        declared.append(
            (f'enhancer/{enhancer.simple_name}', enhancer.strategy.get_default_options())
        )

    return declared


_DECLARED = _declared_options()


def _comparable(options: BaseModel) -> Any:
    # pty-chi's task is a dataclass carrying no set-ness; its own wire format compares it.
    return options.model_dump(mode='json')


def _fields_set(options: BaseModel) -> dict[str, Any]:
    """Which fields are set, at every pydantic depth."""
    return {
        name: _fields_set(value) if isinstance(value, BaseModel) else True
        for name in sorted(options.model_fields_set)
        for value in (getattr(options, name),)
    }


@pytest.mark.parametrize(('name', 'defaults'), _DECLARED, ids=[name for name, _ in _DECLARED])
def test_the_defaults_survive_a_document(name: str, defaults: BaseModel) -> None:
    document = yaml.safe_load(dump_options_document({'algorithm': name}, defaults))
    restored = apply_options_document(defaults, document['options'], subject=name)

    assert type(restored) is type(defaults)
    assert _comparable(restored) == _comparable(defaults)
    assert _fields_set(restored) == _fields_set(defaults)


@pytest.mark.parametrize(('name', 'defaults'), _DECLARED, ids=[name for name, _ in _DECLARED])
def test_the_defaults_survive_pickling(name: str, defaults: BaseModel) -> None:
    restored = pickle.loads(pickle.dumps(defaults))

    assert type(restored) is type(defaults)
    assert _comparable(restored) == _comparable(defaults)
    assert _fields_set(restored) == _fields_set(defaults)


@pytest.mark.parametrize(('name', 'defaults'), _DECLARED, ids=[name for name, _ in _DECLARED])
def test_an_undeclared_field_is_refused(name: str, defaults: BaseModel) -> None:
    payload = defaults.model_dump()
    payload['not_a_declared_field'] = 1

    with pytest.raises(ValueError):
        type(defaults).model_validate(payload)


def test_a_ptychi_task_survives_an_edit() -> None:
    pytest.importorskip('ptychi')
    from ptychi.api import EPIEOptions

    from ptychodus.plugins.ptychi.options import default_options
    from ptychodus.plugins.ptychi.task import dump_task_options

    defaults = default_options(EPIEOptions)
    edited = defaults.model_copy(deep=True)
    edited.task.reconstructor_options.num_epochs = 7

    document = yaml.safe_load(dump_options_document({'algorithm': 'ePIE'}, edited))
    restored = apply_options_document(defaults, document['options'], subject='ePIE')

    # ePIE and PIE serialize the same fields; only the class stamp keeps them apart.
    assert type(restored.task) is EPIEOptions  # type: ignore[attr-defined]
    assert json.loads(dump_task_options(restored.task)) == json.loads(  # type: ignore[attr-defined]
        dump_task_options(edited.task)
    )
