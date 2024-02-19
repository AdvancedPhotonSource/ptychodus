from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.patterns import DiffractionDatasetPresenter
from ...view.patterns import PatternsInfoDialog
from .tree import SimpleTreeModel


class PatternsInfoViewController(Observer):

    def __init__(self, presenter: DiffractionDatasetPresenter, treeModel: SimpleTreeModel) -> None:
        super().__init__()
        self._presenter = presenter
        self._treeModel = treeModel

    @classmethod
    def showInfo(cls, presenter: DiffractionDatasetPresenter, parent: QWidget) -> None:
        treeModel = SimpleTreeModel(presenter.getContentsTree())
        controller = cls(presenter, treeModel)
        presenter.addObserver(controller)

        dialog = PatternsInfoDialog.createInstance(parent)
        dialog.setWindowTitle('Patterns Info')
        dialog.treeView.setModel(treeModel)
        dialog.finished.connect(controller._finish)

        controller._syncModelToView()
        dialog.open()

    def _finish(self, result: int) -> None:
        self._presenter.removeObserver(self)

    def _syncModelToView(self) -> None:
        self._treeModel.setRootNode(self._presenter.getContentsTree())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
