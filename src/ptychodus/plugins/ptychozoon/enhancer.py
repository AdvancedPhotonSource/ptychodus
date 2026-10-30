"""Parent side of the ptychozoon enhancer: hand the inputs to a child and relay what it reports.

The CuPy computation runs in a freshly spawned child (see :mod:`._subprocess`), so
each run gets a clean GPU context and all GPU memory is released when the run
finishes or is stopped. Nothing here imports ptychozoon.
"""

from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
import logging

from pydantic import BaseModel

from ptychodus.api.fluorescence import (
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
    SubprocessFluorescenceEnhancer,
)
from ptychodus.api.spawn import run_subprocess

from ._payload import PtychozoonMessage, PtychozoonPayload
from .options import PtychozoonOptions

__all__ = [
    'PtychozoonFluorescenceEnhancer',
]

logger = logging.getLogger(__name__)

# Module scope for the reason tests/test_subprocess_entry_points.py states: it
# discovers entry points by AST-scanning module-level assignments whose name carries
# ENTRY.
_CHILD_ENTRY = 'ptychodus.plugins.ptychozoon._subprocess:run_enhancement'


class PtychozoonFluorescenceEnhancer(SubprocessFluorescenceEnhancer):
    """Virtual single pixel imaging as ptychozoon implements it, on a GPU where one is visible."""

    def get_default_options(self) -> BaseModel:
        return PtychozoonOptions()

    def enhance(
        self, input_: FluorescenceEnhancerInput, *, options: BaseModel
    ) -> Iterator[FluorescenceEnhancerOutput]:
        if not isinstance(options, PtychozoonOptions):
            raise TypeError(f'Expected PtychozoonOptions, got {type(options).__name__}.')

        # Which build produced a result is the first question asked when two machines
        # disagree, so it is recorded where the run's own log records it.
        try:
            logger.info(f'ptychozoon {version("ptychozoon")}')
        except PackageNotFoundError:
            logger.info('ptychozoon unknown')

        payload = PtychozoonPayload(input_=input_, options=options)

        with run_subprocess(_CHILD_ENTRY, payload, environment=self.device_environment()) as events:
            for event in events:
                if event[0] == PtychozoonMessage.OUTPUT:
                    yield event[1]
                else:
                    logger.debug(f'Dropping unrecognized subprocess message tag {event[0]!r}')
