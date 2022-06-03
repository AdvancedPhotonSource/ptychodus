from __future__ import annotations

from ..model import ImagePresenter, Observer, Observable, Object, ObjectPresenter
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

        view.initializerComboBox.currentTextChanged.connect(presenter.setInitializer)
        view.initializeButton.clicked.connect(presenter.initializeObject)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.initializerComboBox.setCurrentText(self._presenter.getInitializer())

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
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Object',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openObject(filePath, nameFilter)

    def saveObject(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Object',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveObject(filePath, nameFilter)


class ObjectImageController(Observer):

    def __init__(self, presenter: ObjectPresenter, imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
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
