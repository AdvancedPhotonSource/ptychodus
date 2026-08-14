"""Child-side entry points used by ``test_subprocess_reconstructor``.

The ``spawn``-context child imports this module by dotted path, so it must be
resolvable on the child's ``sys.path``; the test module puts the tests
directory there before spawning. Not collected by pytest (the filename does
not match ``test_*.py``). Uses no GPU frameworks. Do not import from
application code.
"""

from __future__ import annotations

import logging
import pickle
import time
from typing import Any

from ptychodus.api.reconstruct import ReconstructOutput, TrainOutput
from ptychodus.model.processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_SETTINGS_SYNC,
    TAG_TRAIN_OUTPUT,
)

logger = logging.getLogger(__name__)


def yield_n_outputs(payload: Any, queue: Any) -> None:
    """Emit ``payload['n']`` :class:`ReconstructOutput`s with product=None."""
    n = int(payload['n'])
    for i in range(n):
        output = ReconstructOutput(product=payload['product'], progress=i + 1)
        queue.put((TAG_OUTPUT, pickle.dumps(output)))


def raise_immediately(payload: Any, queue: Any) -> None:
    """Raise a :class:`ValueError` before emitting anything."""
    raise ValueError(payload['message'])


def hang_forever(payload: Any, queue: Any) -> None:
    """Sleep so the parent must terminate us."""
    time.sleep(3600.0)


def emit_log_then_output(payload: Any, queue: Any) -> None:
    """Log a message, then emit one output; parent must see log before output."""
    logger.warning(payload['log_message'])
    output = ReconstructOutput(product=payload['product'], progress=1)
    queue.put((TAG_OUTPUT, pickle.dumps(output)))


def emit_settings_sync_then_output(payload: Any, queue: Any) -> None:
    """Emit a settings-sync message, then an output."""
    queue.put((TAG_SETTINGS_SYNC, payload['settings']))
    output = ReconstructOutput(product=payload['product'], progress=1)
    queue.put((TAG_OUTPUT, pickle.dumps(output)))


def train_and_save(payload: Any, queue: Any) -> None:
    """Emit one :class:`TrainOutput` and a model-saved path."""
    queue.put((TAG_TRAIN_OUTPUT, pickle.dumps(TrainOutput(progress=1))))
    queue.put((TAG_MODEL_SAVED, payload['saved_path']))
