from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import overload, Union

import numpy

from ...api.data import (DiffractionArrayType, DiffractionData, DiffractionDataState,
                         DiffractionDataset, DiffractionMetadata)
from ...api.tree import SimpleTreeNode


class NullDiffractionData(DiffractionData):

    @property
    def name(self) -> str:
        return str()

    def getState(self) -> DiffractionDataState:
        return DiffractionDataState.MISSING

    def getStartIndex(self) -> int:
        return 0

    def getArray(self) -> DiffractionArrayType:
        return numpy.empty((0, 0, 0), dtype=numpy.uint16)

    @overload
    def __getitem__(self, index: int) -> DiffractionArrayType:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArrayType]:
        ...

    def __getitem__(
            self,
            index: Union[int,
                         slice]) -> Union[DiffractionArrayType, Sequence[DiffractionArrayType]]:

        if isinstance(index, slice):
            return list()
        else:
            return numpy.empty((0, 0), dtype=numpy.uint16)

    def __len__(self) -> int:
        return 0


class NullDiffractionDataset(DiffractionDataset):

    @property
    def metadata(self) -> DiffractionMetadata:
        return DiffractionMetadata(Path('/dev/null'), 0, 0, 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(list())

    @overload
    def __getitem__(self, index: int) -> DiffractionData:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionData]:
        ...

    def __getitem__(self,
                    index: Union[int, slice]) -> Union[DiffractionData, Sequence[DiffractionData]]:
        if isinstance(index, slice):
            return list()
        else:
            return NullDiffractionData()

    def __len__(self) -> int:
        return 0
