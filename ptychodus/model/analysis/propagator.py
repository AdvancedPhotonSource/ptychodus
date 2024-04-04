from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeAlias
import logging

import numpy
import numpy.typing

from ..product import ProbeRepository

ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbePropagation:
    itemIndex: int  # FIXME


class ProbePropagator:

    def __init__(self, repository: ProbeRepository) -> None:
        self._repository = repository

    def getName(self, itemIndex: int) -> str:
        return self._repository.getName(itemIndex)

    def propagate(self, itemIndex: int) -> ProbePropagation:  # FIXME
        return ProbePropagation(itemIndex)
