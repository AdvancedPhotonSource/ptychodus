from __future__ import annotations
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.object import CompareObjectBuilder
from ...model.product import ObjectRepository
from ...view.object import (ObjectEditorDialog, CompareObjectView, CompareObjectParametersView,
                            CompareObjectPlotView)

logger = logging.getLogger(__name__)


class CompareObjectParametersController(Observer):

    def __init__(self, repository: ObjectRepository, view: CompareObjectParametersView) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = view
        self._nameListModel = QStringListModel()
        self._initializer: CompareObjectBuilder | None = None

    @classmethod
    def createInstance(cls, repository: ObjectRepository,
                       view: CompareObjectParametersView) -> None:
        controller = cls(repository, view)
        controller._updateBuilder()
        controller._syncModelToView()
        presenter.item.addObserver(controller)

    def _updateBuilder(self) -> None:
        initializer = self._item.getBuilder()

        if isinstance(initializer, CompareObjectBuilder):
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

    def __init__(self, repository: ObjectRepository, view: CompareObjectPlotView) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = view
        self._initializer: CompareObjectBuilder | None = None

    @classmethod
    def createInstance(cls, repository: ObjectRepository, view: CompareObjectPlotView) -> None:
        controller = cls(repository, view)
        controller._updateBuilder()
        controller._syncModelToView()
        presenter.item.addObserver(controller)

    def _updateBuilder(self) -> None:
        initializer = self._item.getBuilder()

        if isinstance(initializer, CompareObjectBuilder):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            plot2D = self._initializer.getFourierRingCorrelation().getPlot()
            axisX = plot2D.axisX
            axisY = plot2D.axisY

            ax = self._view.axes
            ax.clear()
            ax.set_xlabel(axisX.label)
            ax.set_ylabel(axisY.label)
            ax.grid(True)

            if len(axisX.series) == 1:
                sx = axisX.series[0]

                for sy in axisY.series:
                    ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
            else:
                logger.error('Failed to broadcast plot series!')

            if len(axisX.series) > 1:
                ax.legend(loc='upper right')

            self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()


class CompareObjectViewController:

    def __init__(self, repository: ObjectRepository, parent: QWidget) -> None:
        super().__init__()
        self._view = CompareObjectView.createInstance()
        self._parametersController = CompareObjectParametersController.createInstance(
            repository, self._view.parametersView)
        self._plotController = CompareObjectPlotController.createInstance(
            repository, self._view.plotView)
        self._dialog = ObjectEditorDialog.createInstance(presenter.name, self._view, parent)

    @classmethod
    def editParameters(cls, repository: ObjectRepository, parent: QWidget) -> None:
        controller = cls(repository, parent)
        controller._dialog.open()
