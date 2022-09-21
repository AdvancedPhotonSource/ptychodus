from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import overload, Union

import numpy

from ...api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                         DiffractionDataset, DiffractionMetadata)
from ...api.observer import Observable, Observer
from ...api.tree import SimpleTreeNode
from .assembler import DiffractionDataAssembler


class ActiveDiffractionDataset(DiffractionDataset, Observer):

    def __init__(self, assembler: DiffractionDataAssembler) -> None:
        super().__init__()
        self._assembler = assembler
        self._metadata = DiffractionMetadata(Path('/dev/null'), 0, 0, 0)
        self._contentsTree = SimpleTreeNode.createRoot(list())
        self._arrayDict: dict[int, DiffractionArray] = dict()

    @classmethod
    def createInstance(cls, assembler: DiffractionDataAssembler) -> ActiveDiffractionDataset:
        dataset = cls(assembler)
        assembler.addObserver(dataset)
        return dataset

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]:
        ...

    def __getitem__(
            self, index: Union[int, slice]) -> Union[DiffractionArray, Sequence[DiffractionArray]]:
        if isinstance(index, slice):
            return [self._arrayDict[idx] for idx in range(index.start, index.stop, index.step)]
        else:
            return self._arrayDict[index]

    def __len__(self) -> int:
        return max(self._arrayDict.keys(), default=0)

    def switchTo(self, dataset: DiffractionDataset) -> None:
        self._metadata = self._diffractionDataset.getMetadata()
        self._contentsTree = self._diffractionDataset.getContentsTree()

        self._assembler.start(self._metadata.totalNumberOfImages)

        for array in dataset:
            self._assembler.assemble(array)

        # FIXME when new frames come in, update arrayDict and notifyObservers

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._assembler:
            self.notifyObservers()  # FIXME
