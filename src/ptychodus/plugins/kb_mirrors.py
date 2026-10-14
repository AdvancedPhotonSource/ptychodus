from ptychodus.api.plugins import PluginRegistry


def register_plugins(registry: PluginRegistry) -> None:
    """Register Kirkpatrick-Baez mirror pairs as presets.

    Empty for now: a preset states an instrument's measured optics, and guessing those
    would be worse than offering none, since the numbers reach a reconstruction. Add a
    `registry.kb_mirrors.register_plugin(...)` call per instrument as the measurements
    become available; until then the builder works from its own settings.
    """
