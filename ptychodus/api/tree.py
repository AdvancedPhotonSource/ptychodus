from __future__ import annotations
from collections.abc import Sequence


class SimpleTreeNode:

    def __init__(self, parentItem: SimpleTreeNode | None, itemData: Sequence[str]) -> None:
        self.parentItem = parentItem
        self.itemData = itemData
        self.childItems: list[SimpleTreeNode] = list()

    @classmethod
    def createRoot(cls, itemData: Sequence[str]) -> SimpleTreeNode:
        return cls(None, itemData)

    def createChild(self, itemData: Sequence[str]) -> SimpleTreeNode:
        childItem = SimpleTreeNode(self, itemData)
        self.childItems.append(childItem)
        return childItem

    @property
    def isRoot(self) -> bool:
        return (self.parentItem is None)

    @property
    def isLeaf(self) -> bool:
        return not self.childItems

    def data(self, column: int) -> str | None:
        try:
            return self.itemData[column]
        except IndexError:
            return None

    def row(self) -> int:
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0
