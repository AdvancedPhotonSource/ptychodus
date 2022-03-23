from __future__ import annotations

from ..model import Observer, Observable, Object, ObjectPresenter
from ..view import ImageView, ObjectInitializerView, ObjectParametersView
from .data import FileDialogFactory
from .image import ImageController


class ObjectInitializerController(Observer):
    def __init__(self, presenter: ObjectPresenter, view: ObjectInitializerView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter,
                       view: ObjectInitializerView) -> ObjectInitializerController:
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
    def __init__(self, presenter: ObjectPresenter, view: ObjectParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._initializerController = ObjectInitializerController.createInstance(
            presenter, view.initializerView)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ObjectParametersView,
                       fileDialogFactory: FileDialogFactory) -> ObjectParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        return controller

    def openObject(self) -> None:
        filePath = self._fileDialogFactory.getOpenFilePath(self._view, 'Open Object',
                                                           Object.FILE_FILTER)

        if filePath:
            self._presenter.openObject(filePath)

    def saveObject(self) -> None:
        filePath = self._fileDialogFactory.getSaveFilePath(self._view, 'Save Object',
                                                           Object.FILE_FILTER)

        if filePath:
            self._presenter.saveObject(filePath)


class ObjectImageController(Observer):
    def __init__(self, presenter: ObjectPresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._image_controller = ImageController.createInstance(view, fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectImageController:
        controller = cls(presenter, view, fileDialogFactory)
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
