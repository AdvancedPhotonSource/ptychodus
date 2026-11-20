"""Pytest plugin recording the error each tolerance call site actually achieves.

Wraps ``numpy.testing.assert_allclose`` and scalar ``pytest.approx`` comparisons,
keeps the largest absolute and relative error seen per ``tests/<file>:<line>``, and
writes them as JSON to ``$CURATE_TOLERANCE_LOG`` when the session ends. Loaded only
on request, never from the suite's own configuration:

    CURATE_TOLERANCE_LOG=<out>.json PYTHONPATH=.claude/skills/curate-tests \
      uv run --no-sync python -m pytest -p no:cacheprovider -p tolerance_recorder <files>
"""

from __future__ import annotations

import json
import os
import sys
from types import FrameType
from typing import Any

import _pytest.approx
import numpy
import numpy.testing
import pytest

_achieved: dict[str, tuple[float, float]] = {}
_assert_allclose = numpy.testing.assert_allclose
_approx_scalar_eq = _pytest.approx.ApproxScalar.__eq__


def _call_site() -> str:
    frame: FrameType | None = sys._getframe(2)

    while frame is not None and '/tests/' not in frame.f_code.co_filename:
        frame = frame.f_back

    if frame is None:
        return '?'

    return f'tests/{frame.f_code.co_filename.rsplit("/tests/", 1)[-1]}:{frame.f_lineno}'


def _record(site: str, actual: Any, desired: Any) -> None:
    try:
        a = numpy.asarray(actual, dtype=complex)
        d = numpy.asarray(desired, dtype=complex)
        difference = numpy.abs(a - d)
        magnitude = numpy.abs(numpy.broadcast_to(d, difference.shape))
    except (TypeError, ValueError):
        return

    if difference.size == 0:
        return

    with numpy.errstate(divide='ignore', invalid='ignore'):
        relative = numpy.where(magnitude > 0.0, difference / magnitude, 0.0)

    previous_abs, previous_rel = _achieved.get(site, (0.0, 0.0))
    _achieved[site] = (
        max(previous_abs, float(numpy.nanmax(difference))),
        max(previous_rel, float(numpy.nanmax(relative))),
    )


def _recording_assert_allclose(actual: Any, desired: Any, *args: Any, **kwargs: Any) -> None:
    _record(_call_site(), actual, desired)
    _assert_allclose(actual, desired, *args, **kwargs)


def _recording_approx_scalar_eq(self: _pytest.approx.ApproxScalar, actual: object) -> bool:
    _record(_call_site(), actual, self.expected)
    return _approx_scalar_eq(self, actual)


numpy.testing.assert_allclose = _recording_assert_allclose
_pytest.approx.ApproxScalar.__eq__ = _recording_approx_scalar_eq  # type: ignore[method-assign]  # the recorder wraps pytest's own comparison


def pytest_sessionfinish(session: pytest.Session) -> None:
    path = os.environ.get('CURATE_TOLERANCE_LOG')

    if path:
        with open(path, 'w') as stream:
            json.dump(_achieved, stream, indent=1, sort_keys=True)
