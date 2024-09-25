from __future__ import annotations
from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QDir, QFileInfo, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView, QFileSystemModel, QWizard

from ptychodus.api.observer import Observable, Observer

from ...model.patterns import (
    DiffractionDatasetInputOutputPresenter,
    DiffractionDatasetPresenter,
    DiffractionMetadataPresenter,
    DiffractionPatternPresenter,
)
from ...view.patterns import (
    OpenDatasetWizard,
    OpenDatasetWizardFilesPage,
    OpenDatasetWizardMetadataPage,
    OpenDatasetWizardPatternsPage,
    OpenDatasetWizardPatternCropView,
    OpenDatasetWizardPatternLoadView,
    OpenDatasetWizardPatternMemoryMapView,
    OpenDatasetWizardPatternTransformView,
)
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)

__all__ = [
    "OpenDatasetWizardController",
]


class OpenDatasetWizardFilesController(Observer):
    def __init__(
        self,
        presenter: DiffractionDatasetInputOutputPresenter,
        page: OpenDatasetWizardFilesPage,
        fileDialogFactory: FileDialogFactory,
        fileSystemModel: QFileSystemModel,
        fileSystemProxyModel: QSortFilterProxyModel,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = page
        self._fileDialogFactory = fileDialogFactory
        self._fileSystemModel = fileSystemModel
        self._fileSystemProxyModel = fileSystemProxyModel

    @classmethod
    def createInstance(
        cls,
        presenter: DiffractionDatasetInputOutputPresenter,
        page: OpenDatasetWizardFilesPage,
        fileDialogFactory: FileDialogFactory,
    ) -> OpenDatasetWizardFilesController:
        fileSystemModel = QFileSystemModel()
        fileSystemProxyModel = QSortFilterProxyModel()
        fileSystemModel.setFilter(QDir.Filter.AllEntries | QDir.Filter.AllDirs)
        fileSystemModel.setNameFilterDisables(False)
        fileSystemProxyModel.setSourceModel(fileSystemModel)

        controller = cls(presenter, page, fileDialogFactory, fileSystemModel, fileSystemProxyModel)
        presenter.addObserver(controller)

        page.directoryComboBox.addItem(str(fileDialogFactory.getOpenWorkingDirectory()))
        page.directoryComboBox.addItem(str(Path.home()))
        page.directoryComboBox.setEditable(True)
        page.directoryComboBox.textActivated.connect(controller._handleDirectoryComboBoxActivated)

        page.fileSystemTableView.setModel(controller._fileSystemProxyModel)
        page.fileSystemTableView.setSortingEnabled(True)
        page.fileSystemTableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        page.fileSystemTableView.verticalHeader().hide()
        page.fileSystemTableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        page.fileSystemTableView.doubleClicked.connect(
            controller._handleFileSystemTableDoubleClicked
        )
        page.fileSystemTableView.selectionModel().currentChanged.connect(
            controller._checkIfComplete
        )

        for fileFilter in presenter.getOpenFileFilterList():
            page.fileTypeComboBox.addItem(fileFilter)

        page.fileTypeComboBox.textActivated.connect(controller._setNameFiltersInFileSystemModel)

        controller._setRootPath(fileDialogFactory.getOpenWorkingDirectory())
        controller._syncModelToView()

        return controller

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._fileSystemModel.setRootPath(str(rootPath))
        proxyIndex = self._fileSystemProxyModel.mapFromSource(index)
        self._page.fileSystemTableView.setRootIndex(proxyIndex)
        self._page.directoryComboBox.setCurrentText(str(rootPath))
        self._fileDialogFactory.setOpenWorkingDirectory(rootPath)

    def _handleDirectoryComboBoxActivated(self, text: str) -> None:
        fileInfo = QFileInfo(text)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def _handleFileSystemTableDoubleClicked(self, proxyIndex: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        fileInfo = self._fileSystemModel.fileInfo(index)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def openDataset(self) -> None:
        proxyIndex = self._page.fileSystemTableView.currentIndex()
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        filePath = Path(self._fileSystemModel.filePath(index))
        self._fileDialogFactory.setOpenWorkingDirectory(filePath.parent)

        fileFilter = self._page.fileTypeComboBox.currentText()
        self._presenter.openDiffractionFile(filePath, fileFilter)

    def _checkIfComplete(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(current)
        fileInfo = self._fileSystemModel.fileInfo(index)
        self._page._setComplete(fileInfo.isFile())

    def _setNameFiltersInFileSystemModel(self, currentText: str) -> None:
        z = re.search(r"\((.+)\)", currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f"Dataset File Name Filters: {nameFilters}")
            self._fileSystemModel.setNameFilters(nameFilters)

    def _syncModelToView(self) -> None:
        self._page.fileTypeComboBox.setCurrentText(self._presenter.getOpenFileFilter())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class OpenDatasetWizardMetadataController(Observer):
    def __init__(
        self, presenter: DiffractionMetadataPresenter, page: OpenDatasetWizardMetadataPage
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = page

    @classmethod
    def createInstance(
        cls, presenter: DiffractionMetadataPresenter, page: OpenDatasetWizardMetadataPage
    ) -> OpenDatasetWizardMetadataController:
        controller = cls(presenter, page)
        presenter.addObserver(controller)
        controller._syncModelToView()
        page._setComplete(True)
        return controller

    def importMetadata(self) -> None:
        if self._page.detectorPixelCountCheckBox.isChecked():
            self._presenter.syncDetectorPixelCount()

        if self._page.detectorPixelSizeCheckBox.isChecked():
            self._presenter.syncDetectorPixelSize()

        if self._page.detectorBitDepthCheckBox.isChecked():
            self._presenter.syncDetectorBitDepth()

        if self._page.detectorDistanceCheckBox.isChecked():
            self._presenter.syncDetectorDistance()

        self._presenter.syncPatternCrop(
            syncCenter=self._page.patternCropCenterCheckBox.isChecked(),
            syncExtent=self._page.patternCropExtentCheckBox.isChecked(),
        )

        if self._page.probeEnergyCheckBox.isChecked():
            self._presenter.syncProbeEnergy()

    def _syncModelToView(self) -> None:
        canSyncDetectorPixelCount = self._presenter.canSyncDetectorPixelCount()
        self._page.detectorPixelCountCheckBox.setVisible(canSyncDetectorPixelCount)
        self._page.detectorPixelCountCheckBox.setChecked(canSyncDetectorPixelCount)

        canSyncDetectorPixelSize = self._presenter.canSyncDetectorPixelSize()
        self._page.detectorPixelSizeCheckBox.setVisible(canSyncDetectorPixelSize)
        self._page.detectorPixelSizeCheckBox.setChecked(canSyncDetectorPixelSize)

        canSyncDetectorBitDepth = self._presenter.canSyncDetectorBitDepth()
        self._page.detectorBitDepthCheckBox.setVisible(canSyncDetectorBitDepth)
        self._page.detectorBitDepthCheckBox.setChecked(canSyncDetectorBitDepth)

        canSyncDetectorDistance = self._presenter.canSyncDetectorDistance()
        self._page.detectorDistanceCheckBox.setVisible(canSyncDetectorDistance)
        self._page.detectorDistanceCheckBox.setChecked(canSyncDetectorDistance)

        canSyncPatternCropCenter = self._presenter.canSyncPatternCropCenter()
        self._page.patternCropCenterCheckBox.setVisible(canSyncPatternCropCenter)
        self._page.patternCropCenterCheckBox.setChecked(canSyncPatternCropCenter)

        canSyncPatternCropExtent = self._presenter.canSyncPatternCropExtent()
        self._page.patternCropExtentCheckBox.setVisible(canSyncPatternCropExtent)
        self._page.patternCropExtentCheckBox.setChecked(canSyncPatternCropExtent)

        canSyncProbeEnergy = self._presenter.canSyncProbeEnergy()
        self._page.probeEnergyCheckBox.setVisible(canSyncProbeEnergy)
        self._page.probeEnergyCheckBox.setChecked(canSyncProbeEnergy)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternLoadController(Observer):
    def __init__(
        self, presenter: DiffractionDatasetPresenter, view: OpenDatasetWizardPatternLoadView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls, presenter: DiffractionDatasetPresenter, view: OpenDatasetWizardPatternLoadView
    ) -> PatternLoadController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        view.numberOfThreadsSpinBox.valueChanged.connect(presenter.setNumberOfDataThreads)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfThreadsSpinBox.blockSignals(True)
        self._view.numberOfThreadsSpinBox.setRange(
            self._presenter.getNumberOfDataThreadsLimits().lower,
            self._presenter.getNumberOfDataThreadsLimits().upper,
        )
        self._view.numberOfThreadsSpinBox.setValue(self._presenter.getNumberOfDataThreads())
        self._view.numberOfThreadsSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternMemoryMapController(Observer):
    def __init__(
        self,
        presenter: DiffractionDatasetPresenter,
        view: OpenDatasetWizardPatternMemoryMapView,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(
        cls,
        presenter: DiffractionDatasetPresenter,
        view: OpenDatasetWizardPatternMemoryMapView,
        fileDialogFactory: FileDialogFactory,
    ) -> PatternMemoryMapController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.setCheckable(True)
        controller._syncModelToView()
        view.toggled.connect(presenter.setMemmapEnabled)
        view.scratchDirectoryLineEdit.editingFinished.connect(
            controller._syncScratchDirectoryToModel
        )
        view.scratchDirectoryBrowseButton.clicked.connect(controller._browseScratchDirectory)

        return controller

    def _syncScratchDirectoryToModel(self) -> None:
        scratchDirectory = Path(self._view.scratchDirectoryLineEdit.text())
        self._presenter.setScratchDirectory(scratchDirectory)

    def _browseScratchDirectory(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(
            self._view, "Choose Scratch ScratchDirectory"
        )

        if dirPath:
            self._presenter.setScratchDirectory(dirPath)

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isMemmapEnabled())
        scratchDirectory = self._presenter.getScratchDirectory()

        if scratchDirectory:
            self._view.scratchDirectoryLineEdit.setText(str(scratchDirectory))
        else:
            self._view.scratchDirectoryLineEdit.clear()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternCropController(Observer):
    def __init__(
        self, presenter: DiffractionPatternPresenter, view: OpenDatasetWizardPatternCropView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls, presenter: DiffractionPatternPresenter, view: OpenDatasetWizardPatternCropView
    ) -> PatternCropController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setCropEnabled)

        view.centerXSpinBox.valueChanged.connect(presenter.setCropCenterXInPixels)
        view.centerYSpinBox.valueChanged.connect(presenter.setCropCenterYInPixels)
        view.extentXSpinBox.valueChanged.connect(presenter.setCropWidthInPixels)
        view.extentYSpinBox.valueChanged.connect(presenter.setCropHeightInPixels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isCropEnabled())

        self._view.centerXSpinBox.blockSignals(True)
        self._view.centerXSpinBox.setRange(
            self._presenter.getCropCenterXLimitsInPixels().lower,
            self._presenter.getCropCenterXLimitsInPixels().upper,
        )
        self._view.centerXSpinBox.setValue(self._presenter.getCropCenterXInPixels())
        self._view.centerXSpinBox.blockSignals(False)

        self._view.centerYSpinBox.blockSignals(True)
        self._view.centerYSpinBox.setRange(
            self._presenter.getCropCenterYLimitsInPixels().lower,
            self._presenter.getCropCenterYLimitsInPixels().upper,
        )
        self._view.centerYSpinBox.setValue(self._presenter.getCropCenterYInPixels())
        self._view.centerYSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(
            self._presenter.getCropWidthLimitsInPixels().lower,
            self._presenter.getCropWidthLimitsInPixels().upper,
        )
        self._view.extentXSpinBox.setValue(self._presenter.getCropWidthInPixels())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(
            self._presenter.getCropHeightLimitsInPixels().lower,
            self._presenter.getCropHeightLimitsInPixels().upper,
        )
        self._view.extentYSpinBox.setValue(self._presenter.getCropHeightInPixels())
        self._view.extentYSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternTransformController(Observer):
    def __init__(
        self, presenter: DiffractionPatternPresenter, view: OpenDatasetWizardPatternTransformView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls, presenter: DiffractionPatternPresenter, view: OpenDatasetWizardPatternTransformView
    ) -> PatternTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.valueLowerBoundCheckBox.toggled.connect(presenter.setValueLowerBoundEnabled)
        view.valueLowerBoundSpinBox.valueChanged.connect(presenter.setValueLowerBound)
        view.valueUpperBoundCheckBox.toggled.connect(presenter.setValueUpperBoundEnabled)
        view.valueUpperBoundSpinBox.valueChanged.connect(presenter.setValueUpperBound)
        view.flipXCheckBox.toggled.connect(presenter.setFlipXEnabled)
        view.flipYCheckBox.toggled.connect(presenter.setFlipYEnabled)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.valueLowerBoundCheckBox.setChecked(self._presenter.isValueLowerBoundEnabled())

        self._view.valueLowerBoundSpinBox.blockSignals(True)
        self._view.valueLowerBoundSpinBox.setRange(
            self._presenter.getValueLowerBoundLimits().lower,
            self._presenter.getValueLowerBoundLimits().upper,
        )
        self._view.valueLowerBoundSpinBox.setValue(self._presenter.getValueLowerBound())
        self._view.valueLowerBoundSpinBox.blockSignals(False)

        self._view.valueUpperBoundCheckBox.setChecked(self._presenter.isValueUpperBoundEnabled())

        self._view.valueUpperBoundSpinBox.blockSignals(True)
        self._view.valueUpperBoundSpinBox.setRange(
            self._presenter.getValueUpperBoundLimits().lower,
            self._presenter.getValueUpperBoundLimits().upper,
        )
        self._view.valueUpperBoundSpinBox.setValue(self._presenter.getValueUpperBound())
        self._view.valueUpperBoundSpinBox.blockSignals(False)

        self._view.flipXCheckBox.setChecked(self._presenter.isFlipXEnabled())
        self._view.flipYCheckBox.setChecked(self._presenter.isFlipYEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class OpenDatasetWizardPatternsController:
    def __init__(
        self,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        page: OpenDatasetWizardPatternsPage,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        self._datasetPresenter = datasetPresenter
        self._patternPresenter = patternPresenter
        self._page = page
        self._loadController = PatternLoadController.createInstance(
            datasetPresenter, page.loadView
        )
        self._memoryMapController = PatternMemoryMapController.createInstance(
            datasetPresenter, page.memoryMapView, fileDialogFactory
        )
        self._cropController = PatternCropController.createInstance(
            patternPresenter, page.cropView
        )
        self._transformController = PatternTransformController.createInstance(
            patternPresenter, page.transformView
        )

    @classmethod
    def createInstance(
        cls,
        ioPresenter: DiffractionDatasetInputOutputPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        page: OpenDatasetWizardPatternsPage,
        fileDialogFactory: FileDialogFactory,
    ) -> OpenDatasetWizardPatternsController:
        controller = cls(datasetPresenter, patternPresenter, page, fileDialogFactory)
        page._setComplete(True)
        return controller


class OpenDatasetWizardController:
    def __init__(
        self,
        ioPresenter: DiffractionDatasetInputOutputPresenter,
        metadataPresenter: DiffractionMetadataPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        wizard: OpenDatasetWizard,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        self._ioPresenter = ioPresenter
        self._wizard = wizard
        self._filesController = OpenDatasetWizardFilesController.createInstance(
            ioPresenter, wizard.filesPage, fileDialogFactory
        )
        self._metadataController = OpenDatasetWizardMetadataController.createInstance(
            metadataPresenter, wizard.metadataPage
        )
        self._patternsController = OpenDatasetWizardPatternsController.createInstance(
            ioPresenter, datasetPresenter, patternPresenter, wizard.patternsPage, fileDialogFactory
        )

    @classmethod
    def createInstance(
        cls,
        ioPresenter: DiffractionDatasetInputOutputPresenter,
        metadataPresenter: DiffractionMetadataPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        wizard: OpenDatasetWizard,
        fileDialogFactory: FileDialogFactory,
    ) -> OpenDatasetWizardController:
        controller = cls(
            ioPresenter,
            metadataPresenter,
            datasetPresenter,
            patternPresenter,
            wizard,
            fileDialogFactory,
        )
        wizard.button(QWizard.WizardButton.NextButton).clicked.connect(
            controller._executeNextButtonAction
        )
        wizard.button(QWizard.WizardButton.FinishButton).clicked.connect(
            controller._executeFinishButtonAction
        )
        return controller

    def _executeNextButtonAction(self) -> None:
        page = self._wizard.currentPage()

        if page is self._wizard.metadataPage:
            self._filesController.openDataset()
        elif page is self._wizard.patternsPage:
            self._metadataController.importMetadata()

    def _executeFinishButtonAction(self) -> None:
        self._ioPresenter.startAssemblingDiffractionPatterns()

    def openDataset(self) -> None:
        self._ioPresenter.stopAssemblingDiffractionPatterns(finishAssembling=False)
        self._wizard.restart()
        self._wizard.show()
