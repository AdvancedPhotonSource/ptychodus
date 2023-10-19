from __future__ import annotations
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.object import ObjectRepositoryItemPresenter, CompareObjectInitializer
from ...view.object import (ObjectEditorDialog, CompareObjectView, CompareObjectParametersView,
                            CompareObjectPlotView)

logger = logging.getLogger(__name__)


class CompareObjectParametersController(Observer):

    def __init__(self, presenter: ObjectRepositoryItemPresenter,
                 view: CompareObjectParametersView) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = view
        self._nameListModel = QStringListModel()
        self._initializer: CompareObjectInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ObjectRepositoryItemPresenter,
                       view: CompareObjectParametersView) -> None:
        controller = cls(presenter, view)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, CompareObjectInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.name1ComboBox.setModel(self._nameListModel)
        self._view.name1ComboBox.textActivated.connect(initializer.setName1)
        self._view.name2ComboBox.setModel(self._nameListModel)
        self._view.name2ComboBox.textActivated.connect(initializer.setName2)

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


class CompareObjectPlotController(Observer):

    def __init__(self, presenter: ObjectRepositoryItemPresenter,
                 view: CompareObjectPlotView) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = view
        self._initializer: CompareObjectInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ObjectRepositoryItemPresenter,
                       view: CompareObjectPlotView) -> None:
        controller = cls(presenter, view)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, CompareObjectInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.axes.clear()

            sf = self._initializer.getSpatialFrequency()  # FIXME units
            frc = self._initializer.getFourierRingCorrelation()
            self._view.axes.plot(sf, frc, '-', linewidth=1.5)
            self._view.axes.grid(True)
            self._view.axes.set_xlabel('Spatial Frequency')
            self._view.axes.set_ylabel('Fourier Ring Correlation')

            self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()


class CompareObjectViewController:

    def __init__(self, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._view = CompareObjectView.createInstance()
        self._parametersController = CompareObjectParametersController.createInstance(
            presenter, self._view.parametersView)
        self._plotController = CompareObjectPlotController.createInstance(
            presenter, self._view.plotView)
        self._dialog = ObjectEditorDialog.createInstance(presenter.name, self._view, parent)

    @classmethod
    def editParameters(cls, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._dialog.open()
