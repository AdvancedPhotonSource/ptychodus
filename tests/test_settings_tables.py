"""The tables that route a backend's or an enhancer's options into settings groups.

Each table is keyed on a registered plugin name, so a rename on either side would
drop that plugin's settings without any error: a backend or enhancer whose name no
longer matches simply registers no groups, which reads exactly like one that
declares nothing.
"""

from __future__ import annotations

from ptychodus.api.plugins import PluginRegistry
from ptychodus.model.fluorescence.core import _BRANCHES_BY_ENHANCER
from ptychodus.model.processing.core import _BRANCHES_BY_LIBRARY, _SECTIONS_BY_LIBRARY


def test_every_backend_named_in_the_section_tables_is_a_registered_library() -> None:
    registered = {
        plugin.display_name for plugin in PluginRegistry.load_plugins().reconstructor_libraries
    }
    assert set(_SECTIONS_BY_LIBRARY) | set(_BRANCHES_BY_LIBRARY) <= registered


def test_every_registered_enhancer_names_its_settings_groups() -> None:
    registered = {
        plugin.simple_name for plugin in PluginRegistry.load_plugins().fluorescence_enhancers
    }
    assert registered <= set(_BRANCHES_BY_ENHANCER)
