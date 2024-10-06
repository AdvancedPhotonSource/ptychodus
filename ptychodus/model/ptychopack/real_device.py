from collections.abc import Iterator
import logging

import ptychopack

from .device import PtychoPackDevice

logger = logging.getLogger(__name__)


class RealPtychoPackDevice(PtychoPackDevice):
    def __init__(self) -> None:
        self._available_devices = ptychopack.list_available_devices()
        self._device = self._available_devices[0]

    def get_available_devices(self) -> Iterator[str]:
        for device in self._available_devices:
            yield device.name

    def get_device(self) -> str:
        return self._device.name

    def set_device(self, name: str) -> None:
        for device in self._available_devices:
            if device.name == name:
                self._device = device
                self.notifyObservers()
                return

        logger.warning(f'Failed to set device "{name}"')

    def get_ptychopack_device(self) -> ptychopack.Device:
        return self._device
