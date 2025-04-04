from collections.abc import Sequence
from typing import overload
import logging

logger = logging.getLogger(__name__)


class PtyChiDeviceRepository(Sequence[str]):
    def __init__(self, *, is_developer_mode_enabled: bool) -> None:
        self._devices: list[str] = list()

        try:
            import ptychi
        except ModuleNotFoundError:
            if is_developer_mode_enabled:
                self._devices.extend(f'gpu:{n}' for n in range(4))
        else:
            for device in ptychi.list_available_devices():
                logger.info(device)
                self._devices.append(f'{device.name} ({device.torch_device})')

        if not self._devices:
            logger.info('No devices found!')

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        return self._devices[index]

    def __len__(self) -> int:
        return len(self._devices)
