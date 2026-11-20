"""Work run in a spawned child by the ``iterate_in_subprocess`` and device tests.

What crosses to the child is pickled by reference, so everything here is defined
at module scope and the module must be resolvable on the child's ``sys.path``;
the test modules put the tests directory there before spawning. Not collected by
pytest (the filename does not match ``test_*.py``). Only the device-probing
functions import a GPU framework, and only when they run, so importing this
module never does. Do not import from application code.

A child imports this module before it runs anything, so it imports nothing from
ptychodus at module scope and a child starts in a fraction of a second. Work that
needs a product is in ``subprocess_reconstructor_fixtures``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy


def raise_immediately(message: str) -> Iterator[Any]:
    """Raise a :class:`ValueError` before producing anything."""
    raise ValueError(message)


def yield_environment(name: str) -> Iterator[str | None]:
    """Yield the value of environment variable *name* as the child sees it."""
    yield os.environ.get(name)


def yield_unpicklable() -> Iterator[Any]:
    """Yield an item that cannot be pickled; putting it must raise."""
    yield lambda: None


def yield_then_modify_in_place() -> Iterator[Any]:
    """Yield an array, then overwrite it; the parent must see what was yielded."""
    array = numpy.zeros(1 << 20)
    yield array
    array += 1.0


def probe_cuda_uuids() -> Iterator[list[str]]:
    """Yield the UUID of every CUDA device visible to this child.

    A name does not identify a device when a host holds two of the same model,
    so proving a restriction reached the right one takes something unique.
    """
    import torch

    yield [
        str(torch.cuda.get_device_properties(index).uuid)
        for index in range(torch.cuda.device_count())
    ]


def probe_devices_from_a_clean_process() -> Iterator[tuple[list[str], bool, bool]]:
    """Probe the devices from a process that has never touched CUDA, and say whether it now has.

    Yields the identifiers, whether every device carries a name, and whether this
    process holds a CUDA context once the probe returns.
    """
    import torch

    from ptychodus.api.devices import probe_devices

    devices = probe_devices()
    yield (
        [device.identifier for device in devices],
        all(device.name for device in devices),
        torch.cuda.is_initialized(),
    )
