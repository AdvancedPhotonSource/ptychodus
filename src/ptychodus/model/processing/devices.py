"""The devices a reconstruction may use, probed once and chosen by settings.

The probe costs a spawned child that imports torch, so it is taken on first
request rather than at startup: a headless batch run that never opens a device
panel should not pay for one, and a machine whose GPU stack is broken should
still start.

The selection is ptychodus's own, not any backend's. It is persisted as the
identifiers themselves rather than as indices, so a settings file written on a
four-GPU host still means something on a one-GPU host -- the entries that no
longer resolve are dropped with a warning instead of silently renumbering onto
whatever is there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ptychodus.api.devices import Device, probe_devices, visibility_environment
from ptychodus.api.parameters import Parameter

__all__ = ['DeviceSelection']


class DeviceSelection:
    """Which devices a spawned child may see, and the environment that enforces it."""

    def __init__(self, selection: Parameter[str]) -> None:
        self._selection = selection
        self._available: list[Device] | None = None

    def available_devices(self) -> Sequence[Device]:
        """Every device a child could use, probed on first request and remembered."""
        if self._available is None:
            self._available = list(probe_devices())

        return self._available

    def refresh(self) -> None:
        """Forget the probe, so the next request enumerates again."""
        self._available = None

    def selected_identifiers(self) -> Sequence[str]:
        """The identifiers currently chosen; empty means every device."""
        return [entry.strip() for entry in self._selection.get_value().split(',') if entry.strip()]

    def set_selected_identifiers(self, identifiers: Sequence[str]) -> None:
        self._selection.set_value(','.join(identifiers))

    def environment(self) -> Mapping[str, str]:
        """The environment entries that restrict a child to the chosen devices."""
        selected = self.selected_identifiers()

        if not selected:
            return {}

        return visibility_environment(
            selected, [device.identifier for device in self.available_devices()]
        )
