from typing import Any
import logging

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, Qt

from ...model.product import ProductRepository

logger = logging.getLogger(__name__)


class ProductRepositoryListModel(QAbstractListModel):
    def __init__(
        self,
        repository: ProductRepository,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
            else:
                return item.get_name()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._repository)
