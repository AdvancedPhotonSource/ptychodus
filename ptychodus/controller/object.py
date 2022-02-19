from pathlib import Path

from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QFileDialog

from ..model import Observer, Observable, ObjectIO, ObjectPresenter
from ..view import ImageView, ObjectInitializerView, ObjectParametersView

from .image import ImageController


class ObjectInitializerController(Observer):
    def __init__(self, presenter: ObjectPresenter, view: ObjectInitializerView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._initComboBoxModel = QStandardItemModel()

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ObjectInitializerView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for initializer in presenter.getInitializerList():
            view.initializerComboBox.addItem(initializer)

        view.initializerComboBox.currentTextChanged.connect(presenter.setCurrentInitializer)
        view.initializeButton.clicked.connect(presenter.initializeObject)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getCurrentInitializer())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ObjectParametersController:
    def __init__(self, presenter: ObjectPresenter, view: ObjectParametersView) -> None:
        self._presenter = presenter
        self._view = view
        self._initializerController = ObjectInitializerController.createInstance(
            presenter, view.initializerView)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ObjectParametersView):
        controller = cls(presenter, view)
        return controller

    def openObject(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(self._view, 'Open Object', str(Path.home()),
                                                  ObjectIO.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.openObject(filePath)

    def saveObject(self) -> None:
        fileName, _ = QFileDialog.getSaveFileName(self._view, 'Save Object', str(Path.home()),
                                                  ObjectIO.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.saveObject(filePath)


class ObjectImageController(Observer):
    def __init__(self, presenter: ObjectPresenter, view: ImageView) -> None:
        self._presenter = presenter
        self._image_controller = ImageController.createInstance(view)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ImageView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller.renderImageData()
        view.imageRibbon.frameGroupBox.setVisible(False)
        return controller

    def renderImageData(self) -> None:
        estimate = self._presenter.getObject()
        self._image_controller.renderImageData(estimate)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self.renderImageData()
