"""Renders run in worker threads, never more at once than the runner's cap."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from ptychodus.webapp.rendering.offload import RenderRunner


class _SlowRender:
    """Sleeps like a blocking read, recording its thread and how many overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0
        self.threads: set[int] = set()

    def __call__(self, value: int) -> int:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.threads.add(threading.get_ident())

        time.sleep(0.05)

        with self._lock:
            self._active -= 1

        return value * 2


@pytest.mark.asyncio
async def test_renders_run_off_the_event_loop_within_the_cap() -> None:
    runner = RenderRunner(2)
    render = _SlowRender()

    results = await asyncio.gather(*(runner.run(render, value) for value in range(6)))

    assert results == [0, 2, 4, 6, 8, 10]
    assert render.peak == 2
    assert threading.get_ident() not in render.threads


@pytest.mark.asyncio
async def test_the_loop_answers_while_a_render_runs() -> None:
    runner = RenderRunner(1)
    render = _SlowRender()
    task = asyncio.ensure_future(runner.run(render, 1))
    await asyncio.sleep(0)

    assert not task.done()
    assert await task == 2


@pytest.mark.asyncio
async def test_an_error_reaches_the_caller() -> None:
    with pytest.raises(ZeroDivisionError):
        await RenderRunner(1).run(divmod, 1, 0)


def test_a_runner_needs_room_for_one_render() -> None:
    with pytest.raises(ValueError, match='at least 1'):
        RenderRunner(0)

    assert RenderRunner(3).concurrency == 3
