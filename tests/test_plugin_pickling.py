"""Every reconstructor and enhancer survives the trip to the process that runs it.

The application runs ``reconstruct``, ``train`` and ``enhance`` in a spawned child
by pickling the plugin object, so a plugin that does not pickle cannot run at all
-- and nothing short of a real run would say so. Checked against whatever
registered, so a library installed here is covered and one that is not drops out.
"""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from ptychodus.api.plugins import PluginRegistry


def _plugins() -> list[tuple[str, Any]]:
    registry = PluginRegistry.load_plugins()
    found: list[tuple[str, Any]] = []

    for library in registry.reconstructor_libraries:
        for reconstructor in library.strategy:
            found.append((f'{library.display_name} {reconstructor.name}', reconstructor))

    for enhancer in registry.fluorescence_enhancers:
        found.append((enhancer.display_name, enhancer.strategy))

    return found


PLUGINS = _plugins()


def test_plugins_were_discovered() -> None:
    """The fluorescence enhancers need no optional dependency, so this is never empty."""
    assert PLUGINS


@pytest.mark.parametrize(('name', 'plugin'), PLUGINS, ids=[name for name, _ in PLUGINS])
def test_a_plugin_round_trips_through_pickle(name: str, plugin: Any) -> None:
    restored = pickle.loads(pickle.dumps(plugin))

    assert type(restored) is type(plugin), name
    assert restored.get_default_options() == plugin.get_default_options(), name
