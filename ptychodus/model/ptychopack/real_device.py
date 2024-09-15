from collections.abc import Iterator
import logging

from ptychopack import list_available_devices, Device

from .device import PtychoPackDevice

logger = logging.getLogger(__name__)


class RealPtychoPackDevice(PtychoPackDevice):

    def __init__(self) -> None:
        self._available_devices = list_available_devices()
        self._device = self._available_devices[0]

    def get_available_devices(self) -> Iterator[str]:
        return iter(self._available_devices)  # FIXME typing

    def get_device(self) -> str:
        return self._device.name

    def set_device(self, device: str) -> None:
        for dev in self._available_devices:
            if dev.torch_device == device:
                self._device = dev
                return

        logger.warning(f'Failed to set device \"{device}\"')
