"""Work run in a spawned child by the ``iterate_in_subprocess`` and device tests.

What crosses to the child is pickled by reference, so everything here is defined
at module scope and the module must be resolvable on the child's ``sys.path``;
the test modules put the tests directory there before spawning. Not collected by
pytest (the filename does not match ``test_*.py``). Only the device-probing
functions import a GPU framework, and only when they run, so importing this
module never does. Do not import from application code.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
from pydantic import BaseModel

from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    NoOptions,
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)

logger = logging.getLogger(__name__)


def yield_outputs(product: Product, count: int) -> Iterator[ReconstructOutput]:
    """Yield *count* outputs, numbered from one."""
    for index in range(count):
        yield ReconstructOutput(product=product, progress=index + 1)


def raise_immediately(message: str) -> Iterator[Any]:
    """Raise a :class:`ValueError` before producing anything."""
    raise ValueError(message)


def yield_one_then_hang(product: Product) -> Iterator[ReconstructOutput]:
    """Yield one output, then sleep so the parent must terminate us.

    The output is what lets the parent reach its first ``yield``: a child that
    only sleeps leaves the parent generator un-started, and closing a generator
    that never ran executes no cleanup at all.
    """
    yield ReconstructOutput(product=product, progress=1)
    time.sleep(3600.0)


def log_then_yield(message: str, product: Product) -> Iterator[ReconstructOutput]:
    """Log a message, then yield one output; the parent must see the log first."""
    logger.warning(message)
    yield ReconstructOutput(product=product, progress=1)


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


class FakeTrainableReconstructor(TrainableReconstructor):
    """A trainable reconstructor touching no GPU framework, as a backend would be written.

    ``reconstruct`` yields the input product ``num_outputs`` times. ``train``
    writes a model file into the output directory and names it in its last
    output, mutating ``self`` along the way so a test can show the mutation is
    not seen by whoever ran it in a child.
    """

    def __init__(self, num_outputs: int = 1) -> None:
        super().__init__()
        self._num_outputs = num_outputs
        self.loaded_model: Path | None = None
        self.trained_in_this_process = False

    @property
    def name(self) -> str:
        return 'Fake'

    def get_default_options(self) -> BaseModel:
        return NoOptions()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        for index in range(self._num_outputs):
            yield ReconstructOutput(
                product=input_.product, progress=index + 1, progress_goal=self._num_outputs
            )

    def is_model_loaded(self) -> bool:
        return self.loaded_model is not None

    def get_model_file_description(self) -> str:
        return 'Fake Model'

    def load_model(self, file_path: Path) -> None:
        self.loaded_model = file_path

    def get_model_file_extensions(self) -> tuple[str, ...]:
        return ('.fake',)

    def save_model(self, file_path: Path) -> None:
        if self.loaded_model is None:
            raise RuntimeError('No model has been loaded or trained.')

        file_path.write_bytes(self.loaded_model.read_bytes())

    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        self.trained_in_this_process = True
        yield TrainOutput(progress=1, progress_goal=2)

        model_path = output_directory / 'model.fake'
        model_path.write_bytes(b'trained')
        yield TrainOutput(progress=2, progress_goal=2, trained_model=model_path)


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
