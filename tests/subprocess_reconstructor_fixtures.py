"""Work run in a spawned child that yields reconstructor outputs.

Split from ``subprocess_child_fixtures`` because a child must import the module
holding its work, and importing ``ptychodus.api.product`` costs a child about two
seconds. Work that needs no product stays there; work that does lives here. The
same rules apply: everything at module scope, resolvable on the child's
``sys.path``, not collected by pytest, not imported from application code.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path

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


class FakeTrainableReconstructor(TrainableReconstructor):
    """A trainable reconstructor touching no GPU framework, written as a library's would be.

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
