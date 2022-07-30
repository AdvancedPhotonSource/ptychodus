from __future__ import annotations
from typing import Callable

from ..model import ImagePresenter, Observer, Observable, Object, ObjectPresenter
from ..view import ImageView, ObjectParametersView
from .data import FileDialogFactory
from .image import ImageController


class ObjectParametersController(Observer):

    def __init__(self, presenter: ObjectPresenter, view: ObjectParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, view: ObjectParametersView,
                       fileDialogFactory: FileDialogFactory) -> ObjectParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.objectView.pixelSizeXWidget.setReadOnly(True)
        view.objectView.pixelSizeYWidget.setReadOnly(True)

        for name in presenter.getInitializerNameList():
            initAction = view.initializerView.buttonBox.initializeMenu.addAction(name)
            initAction.triggered.connect(controller._createInitializerLambda(name))

        view.initializerView.buttonBox.saveButton.clicked.connect(controller._saveObject)

        controller._syncModelToView()

        return controller

    def _initializeObject(self, name: str) -> None:
        if name == 'Open File...':
            self._openObject()

        self._presenter.setInitializer(name)
        self._presenter.initializeObject()

    def _createInitializerLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeObject(name)

    def _openObject(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Object',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.setOpenFilePath(filePath)
            self._presenter.setOpenFileFilter(nameFilter)

    def _saveObject(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Object',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveObject(filePath, nameFilter)

    def _syncModelToView(self) -> None:
        self._view.objectView.pixelSizeXWidget.setLengthInMeters(
            self._presenter.getPixelSizeXInMeters())
        self._view.objectView.pixelSizeYWidget.setLengthInMeters(
            self._presenter.getPixelSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ObjectImageController(Observer):

    def __init__(self, presenter: ObjectPresenter, imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, imagePresenter: ImagePresenter,
                       view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectImageController:
        controller = cls(presenter, imagePresenter, view, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setVisible(False)
        return controller

    def _syncModelToView(self) -> None:
        array = self._presenter.getObject()
        self._imagePresenter.setArray(array)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
