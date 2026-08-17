"""Parent-safe device enumeration for the PtyChi backend.

Historically this module called ``ptychi.list_available_devices()`` at import
time inside the parent process, which pulled ptychi (and hence torch/CuPy)
into the parent's ``sys.modules``. The refactor moves the probe into a
one-shot spawned subprocess whose result is cached on the repository
instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from typing import Any, overload
import logging
import pickle

from ..processing._subprocess_protocol import ChildError, run_subprocess

logger = logging.getLogger(__name__)


_PROBE_ENTRY = 'ptychodus.model.ptychi._subprocess:probe_devices'


def _probe_devices_via_subprocess() -> list[str]:
    """Spawn a child that calls ``ptychi.list_available_devices()`` and return the list.

    Any failure -- ptychi not installed, subprocess crash, timeout -- is
    logged and an empty list is returned. This keeps device probing
    non-fatal for parents whose GPU stack is temporarily broken.
    """
    try:
        with run_subprocess(_PROBE_ENTRY, None, terminate_grace_sec=5.0) as events:
            for event in events:
                if event[0] == 'output':
                    devices = pickle.loads(event[1])
                    if isinstance(devices, list):
                        return list(devices)
    except ChildError as exc:
        logger.warning('Device probe subprocess failed: %s', exc.child_exception_type)
    except Exception:
        logger.exception('Device probe subprocess raised in the parent.')
    return []


class PtyChiDeviceRepository(Sequence[str]):
    def __init__(self, *, is_developer_mode_enabled: bool) -> None:
        self._devices: list[str] = list()

        if find_spec('ptychi') is None:
            if is_developer_mode_enabled:
                self._devices.extend(f'gpu:{n}' for n in range(4))
        else:
            for device in _probe_devices_via_subprocess():
                logger.info(device)
                self._devices.append(device)

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

    # Kept for callers that want to force a re-enumeration after e.g. plugging
    # in a new device. Not used inside ptychodus today.
    def refresh(self) -> None:
        if find_spec('ptychi') is None:
            return
        self._devices = _probe_devices_via_subprocess()


# `_PROBE_PAYLOAD` is None; keeping this alias documents that the entry point
# ignores its payload argument.
_PROBE_PAYLOAD: Any = None
