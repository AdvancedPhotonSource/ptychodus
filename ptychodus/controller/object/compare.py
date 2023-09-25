from __future__ import annotations
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.object import ObjectRepositoryItemPresenter, CompareObjectInitializer
from ...view.object import ObjectEditorDialog, CompareObjectView

logger = logging.getLogger(__name__)


class CompareObjectViewController(Observer):

    def __init__(self, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = CompareObjectView.createInstance()
        self._dialog = ObjectEditorDialog.createInstance(presenter.name, self._view, parent)
        self._nameListModel = QStringListModel()
        self._initializer: CompareObjectInitializer | None = None

    @classmethod
    def editParameters(cls, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        controller._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, CompareObjectInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.name1ComboBox.setModel(self._nameListModel)
        self._view.name1ComboBox.currentTextChanged.connect(initializer.setName1)
        self._view.name2ComboBox.setModel(self._nameListModel)
        self._view.name2ComboBox.currentTextChanged.connect(initializer.setName2)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.name1ComboBox.blockSignals(True)
            self._view.name2ComboBox.blockSignals(True)
            self._nameListModel.setStringList(self._initializer.getComparableNames())
            self._view.name1ComboBox.setCurrentText(self._initializer.getName1())
            self._view.name2ComboBox.setCurrentText(self._initializer.getName2())
            self._view.name2ComboBox.blockSignals(False)
            self._view.name1ComboBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
