from __future__ import annotations
from typing import Callable

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QWidget

from ..model import (FileProbeInitializer, FresnelZonePlateProbeInitializer, ImagePresenter,
                     Observable, Observer, Probe, ProbePresenter, SuperGaussianProbeInitializer)
from ..view import (FresnelZonePlateProbeDialog, FresnelZonePlateProbeView, ImageView,
                    ProbeParametersView, ProbeView, ProgressBarItemDelegate,
                    SuperGaussianProbeDialog, SuperGaussianProbeView)
from .data import FileDialogFactory
from .image import ImageController


class ProbeController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeView) -> ProbeController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.sizeSpinBox.valueChanged.connect(presenter.setProbeSize)
        view.sizeSpinBox.autoToggled.connect(presenter.setAutomaticProbeSizeEnabled)
        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.sizeSpinBox.setAutomatic(self._presenter.isAutomaticProbeSizeEnabled())
        self._view.sizeSpinBox.setValueAndRange(self._presenter.getProbeSize(),
                                                self._presenter.getProbeMinSize(),
                                                self._presenter.getProbeMaxSize())
        self._view.energyWidget.setEnergyInElectronVolts(
            self._presenter.getProbeEnergyInElectronVolts())
        self._view.wavelengthWidget.setLengthInMeters(self._presenter.getProbeWavelengthInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class SuperGaussianProbeController(Observer):

    def __init__(self, initializer: SuperGaussianProbeInitializer,
                 view: SuperGaussianProbeView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: SuperGaussianProbeInitializer,
                       view: SuperGaussianProbeView) -> SuperGaussianProbeController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.annularRadiusWidget.lengthChanged.connect(initializer.setAnnularRadiusInMeters)
        view.probeWidthWidget.lengthChanged.connect(initializer.setProbeWidthInMeters)
        view.orderParameterWidget.valueChanged.connect(initializer.setOrderParameter)
        view.numberOfModesSpinBox.valueChanged.connect(initializer.setNumberOfProbeModes)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.annularRadiusWidget.setLengthInMeters(
            self._initializer.getAnnularRadiusInMeters())
        self._view.probeWidthWidget.setLengthInMeters(self._initializer.getProbeWidthInMeters())
        self._view.orderParameterWidget.setValue(self._initializer.getOrderParameter())
        self._view.numberOfModesSpinBox.setValue(self._initializer.getNumberOfProbeModes())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()


class FresnelZonePlateProbeController(Observer):

    def __init__(self, initializer: FresnelZonePlateProbeInitializer,
                 view: FresnelZonePlateProbeView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: FresnelZonePlateProbeInitializer,
                       view: FresnelZonePlateProbeView) -> FresnelZonePlateProbeController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.zonePlateRadiusWidget.lengthChanged.connect(initializer.setZonePlateRadiusInMeters)
        view.outermostZoneWidthWidget.lengthChanged.connect(
            initializer.setOutermostZoneWidthInMeters)
        view.beamstopDiameterWidget.lengthChanged.connect(
            initializer.setCentralBeamstopDiameterInMeters)
        view.defocusDistanceWidget.lengthChanged.connect(initializer.setDefocusDistanceInMeters)
        view.numberOfModesSpinBox.valueChanged.connect(initializer.setNumberOfProbeModes)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.zonePlateRadiusWidget.setLengthInMeters(
            self._initializer.getZonePlateRadiusInMeters())
        self._view.outermostZoneWidthWidget.setLengthInMeters(
            self._initializer.getOutermostZoneWidthInMeters())
        self._view.beamstopDiameterWidget.setLengthInMeters(
            self._initializer.getCentralBeamstopDiameterInMeters())
        self._view.defocusDistanceWidget.setLengthInMeters(
            self._initializer.getDefocusDistanceInMeters())
        self._view.numberOfModesSpinBox.setValue(self._initializer.getNumberOfProbeModes())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()


class ProbeModesTableModel(QAbstractTableModel):

    def __init__(self, presenter: ProbePresenter, parent: QObject = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section == 0:
                value = QVariant('Mode')
            elif section == 1:
                value = QVariant('Relative Power')

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole and index.column() == 0:
                value = QVariant(index.row())
            if role == Qt.UserRole and index.column() == 1:
                power = self._presenter.getProbeModeRelativePower(index.row())
                powerPct = int((100 * power).to_integral_value())
                value = QVariant(powerPct)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfProbeModes()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2


class ProbeParametersController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeParametersView,
                 imagePresenter: ImagePresenter, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._imagePresenter = imagePresenter
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._probeController = ProbeController.createInstance(presenter, view.probeView)
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._modesTableModel = ProbeModesTableModel(presenter)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeParametersView,
                       imagePresenter: ImagePresenter, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeParametersController:
        controller = cls(presenter, view, imagePresenter, imageView, fileDialogFactory)
        presenter.addObserver(controller)

        delegate = ProgressBarItemDelegate(view.modesView.tableView)
        view.modesView.tableView.setItemDelegateForColumn(1, delegate)
        view.modesView.tableView.setModel(controller._modesTableModel)
        view.modesView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.modesView.tableView.selectionModel().currentChanged.connect(
            controller._displayProbeMode)

        imageView.imageRibbon.indexGroupBox.setVisible(False)

        for name in presenter.getInitializerNameList():
            initAction = view.modesView.buttonBox.initializeMenu.addAction(name)
            initAction.triggered.connect(controller._createInitializerLambda(name))

        view.modesView.buttonBox.saveButton.clicked.connect(controller._saveProbe)

        controller._syncModelToView()

        return controller

    def _editProbe(self) -> None:
        initializerName = self._presenter.getInitializerName()
        initializer = self._presenter.getInitializer()

        # TODO show live update while editing
        if isinstance(initializer, FileProbeInitializer):
            filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
                self._view,
                'Open Probe',
                nameFilters=initializer.getOpenFileFilterList(),
                selectedNameFilter=initializer.getOpenFileFilter())

            if filePath:
                initializer.setOpenFilePath(filePath)
                initializer.setOpenFileFilter(nameFilter)
                self._presenter.initializeProbe()
        elif isinstance(initializer, SuperGaussianProbeInitializer):
            sgDialog = SuperGaussianProbeDialog.createInstance(self._view)
            sgDialog.setWindowTitle(initializerName)
            sgDialog.finished.connect(self._finishInitialization)
            sgController = SuperGaussianProbeController.createInstance(
                initializer, sgDialog.editorView)
            sgDialog.open()
        elif isinstance(initializer, FresnelZonePlateProbeInitializer):
            fzpDialog = FresnelZonePlateProbeDialog.createInstance(self._view)
            fzpDialog.setWindowTitle(initializerName)
            fzpDialog.finished.connect(self._finishInitialization)
            fzpController = FresnelZonePlateProbeController.createInstance(
                initializer, fzpDialog.editorView)
            fzpDialog.open()
        else:
            self._finishInitialization(QDialog.Accepted)

    def _startInitialization(self, name: str) -> None:
        self._presenter.setInitializerByName(name)
        self._editProbe()

    def _finishInitialization(self, result: int) -> None:
        if result == QDialog.Accepted:
            self._presenter.initializeProbe()

    def _createInitializerLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._startInitialization(name)

    def _renderImageData(self, index: int) -> None:
        array = self._presenter.getProbeMode(index)
        self._imagePresenter.setArray(array)

    def _displayProbeMode(self, current: QModelIndex, previous: QModelIndex) -> None:
        self._renderImageData(current.row())

    def _saveProbe(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Probe',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveProbe(filePath, nameFilter)

    def _syncModelToView(self) -> None:
        self._modesTableModel.refresh()

        current = self._view.modesView.tableView.currentIndex()
        self._renderImageData(current.row())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeImageController(Observer):

    def __init__(self, presenter: ProbePresenter, imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, imagePresenter: ImagePresenter,
                       view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeImageController:
        controller = cls(presenter, imagePresenter, view, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setTitle('Probe Mode')
        view.imageRibbon.indexGroupBox.indexSpinBox.valueChanged.connect(
            controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._presenter.getProbeMode(index)
        self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        numberOfProbeModes = self._presenter.getNumberOfProbeModes()
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setEnabled(numberOfProbeModes > 0)
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setRange(0, numberOfProbeModes - 1)

        index = self._view.imageRibbon.indexGroupBox.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
