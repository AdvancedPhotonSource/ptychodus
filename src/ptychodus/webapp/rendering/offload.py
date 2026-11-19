"""Running rendering off the event loop, a bounded number at a time."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

import anyio
import anyio.to_thread

__all__ = ['RenderRunner']

_T = TypeVar('_T')


class RenderRunner:
    """Runs blocking render work in worker threads, at most `concurrency` at once.

    A render reads an HDF5 file and builds an image, which would stall every other request
    if it ran on the event loop; in a thread it does not, and the cap bounds how many
    products are held in memory at once. One runner serves every transport in a process.
    """

    def __init__(self, concurrency: int) -> None:
        if concurrency < 1:
            raise ValueError(f'concurrency must be at least 1, not {concurrency}.')

        self._limiter = anyio.CapacityLimiter(concurrency)

    @property
    def concurrency(self) -> int:
        return int(self._limiter.total_tokens)

    async def run(self, func: Callable[..., _T], /, *args: Any) -> _T:
        """Call ``func(*args)`` in a worker thread once a slot is free, and return its value."""
        return await anyio.to_thread.run_sync(partial(func, *args), limiter=self._limiter)
