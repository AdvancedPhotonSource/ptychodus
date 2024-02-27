from __future__ import annotations
from dataclasses import dataclass
import logging

from ..product import ProbeRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbePropagation:
    itemIndex: int  # FIXME


class ProbePropagator:

    def __init__(self, repository: ProbeRepository) -> None:
        self._repository = repository

    def propagate(self, itemIndex: int) -> ProbePropagation:  # FIXME
        return ProbePropagation(itemIndex)
