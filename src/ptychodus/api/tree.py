from __future__ import annotations
from collections.abc import Sequence


class SimpleTreeNode:
    def __init__(self, parent_item: SimpleTreeNode | None, item_data: Sequence[str]) -> None:
        self.parent_item = parent_item
        self.item_data = item_data
        self.child_items: list[SimpleTreeNode] = list()

    @classmethod
    def create_root(cls, item_data: Sequence[str]) -> SimpleTreeNode:
        return cls(None, item_data)

    def create_child(self, item_data: Sequence[str]) -> SimpleTreeNode:
        child_item = SimpleTreeNode(self, item_data)
        self.child_items.append(child_item)
        return child_item

    @property
    def is_root(self) -> bool:
        return self.parent_item is None

    @property
    def is_leaf(self) -> bool:
        return not self.child_items

    def data(self, column: int) -> str | None:
        try:
            return self.item_data[column]
        except IndexError:
            return None

    def row(self) -> int:
        if self.parent_item:
            return self.parent_item.child_items.index(self)

        return 0
