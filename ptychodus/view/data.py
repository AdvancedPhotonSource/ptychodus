from __future__ import annotations
from typing import Callable, Generic, Optional, TypeVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QPushButton, QSpinBox,
                             QStackedWidget, QTreeView, QVBoxLayout, QWidget)

T = TypeVar('T', bound=QGroupBox)


class DataNavigationPage(Generic[T], QWidget):

    def __init__(self, contentsView: T, backwardText: str, forwardText: str,
                 parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.contentsView = contentsView
        self.buttonBox = QWidget()
        self.backwardButton = QPushButton('< ' + backwardText)
        self.forwardButton = QPushButton(forwardText + ' >')

    @classmethod
    def createInstance(cls, contentsView: T, backwardText: str, forwardText: str, \
                       parent: Optional[QWidget] = None) -> DataNavigationPage[T]:
        view = cls(contentsView, backwardText, forwardText, parent)

        buttonBoxLayout = QHBoxLayout()
        buttonBoxLayout.setContentsMargins(0, 0, 0, 0)
        buttonBoxLayout.addWidget(view.backwardButton)
        buttonBoxLayout.addWidget(view.forwardButton)
        view.buttonBox.setLayout(buttonBoxLayout)

        layout = QVBoxLayout()
        layout.addWidget(view.contentsView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class DatasetFileView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Dataset', parent)
        self.fileTypeComboBox = QComboBox()
        self.fileSystemTreeView = QTreeView()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DatasetFileView:
        view = cls(parent)

        view.fileSystemTreeView.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(view.fileTypeComboBox)
        layout.addWidget(view.fileSystemTreeView)
        view.setLayout(layout)

        return view


class MetadataView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Metadata', parent)
        self.detectorPixelCountCheckBox = QCheckBox('Detector Pixel Count')
        self.detectorPixelSizeCheckBox = QCheckBox('Detector Pixel Size')
        self.detectorDistanceCheckBox = QCheckBox('Detector Distance')
        self.imageCropCenterCheckBox = QCheckBox('Image Crop Center')
        self.imageCropExtentCheckBox = QCheckBox('Image Crop Extent')
        self.probeEnergyCheckBox = QCheckBox('Probe Energy')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> MetadataView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.detectorPixelCountCheckBox)
        layout.addWidget(view.detectorPixelSizeCheckBox)
        layout.addWidget(view.detectorDistanceCheckBox)
        layout.addWidget(view.imageCropCenterCheckBox)
        layout.addWidget(view.imageCropExtentCheckBox)
        layout.addWidget(view.probeEnergyCheckBox)
        layout.addStretch()
        view.setLayout(layout)

        return view


class PatternLoadView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Load', parent)
        self.numberOfThreadsSpinBox = QSpinBox()
        self.numberOfPatternsSpinBox = QSpinBox()
        # TODO controls to assemble in-memory or use scratch directory

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PatternLoadView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', view.numberOfThreadsSpinBox)
        layout.addRow('Total Number of Patterns:', view.numberOfPatternsSpinBox)
        view.setLayout(layout)

        return view


class PatternCropView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Crop', parent)
        self.centerLabel = QLabel('Center [px]:')
        self.centerXSpinBox = QSpinBox()
        self.centerYSpinBox = QSpinBox()
        self.extentLabel = QLabel('Extent [px]:')
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PatternCropView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.centerLabel, 0, 0)
        layout.addWidget(view.centerXSpinBox, 0, 1)
        layout.addWidget(view.centerYSpinBox, 0, 2)
        layout.addWidget(view.extentLabel, 1, 0)
        layout.addWidget(view.extentXSpinBox, 1, 1)
        layout.addWidget(view.extentYSpinBox, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        view.setLayout(layout)

        return view


class PatternTransformView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Transform', parent)
        self.thresholdLabel = QLabel('Threshold:')
        self.thresholdSpinBox = QSpinBox()
        self.axesLabel = QLabel('Axes:')
        self.flipXCheckBox = QCheckBox('Flip X')
        self.flipYCheckBox = QCheckBox('Flip Y')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PatternTransformView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.thresholdLabel, 0, 0)
        layout.addWidget(view.thresholdSpinBox, 0, 1, 1, 2)
        layout.addWidget(view.axesLabel, 1, 0)
        layout.addWidget(view.flipXCheckBox, 1, 1, Qt.AlignHCenter)
        layout.addWidget(view.flipYCheckBox, 1, 2, Qt.AlignHCenter)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        view.setLayout(layout)

        return view


class PatternsView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Patterns', parent)
        self.loadView = PatternLoadView.createInstance()
        self.cropView = PatternCropView.createInstance()
        self.transformView = PatternTransformView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PatternsView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.loadView)
        layout.addWidget(view.cropView)
        layout.addWidget(view.transformView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class DatasetView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Dataset', parent)
        self.treeView = QTreeView()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DatasetView:
        view = cls(parent)

        view.treeView.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        view.setLayout(layout)

        return view


class DataParametersView(QStackedWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.filePage = DataNavigationPage.createInstance( \
                DatasetFileView.createInstance(), 'Unused', 'Load Dataset')
        self.metadataPage = DataNavigationPage.createInstance( \
                MetadataView.createInstance(), 'Reload Dataset', 'Import Metadata')
        self.patternsPage = DataNavigationPage.createInstance( \
                PatternsView.createInstance(), 'Revisit Metadata', 'Load Patterns')
        self.datasetPage = DataNavigationPage.createInstance( \
                DatasetView.createInstance(), 'Reload Patterns', 'Export Patterns')

    def _createNavigationLambda(self, index: int) -> Callable[[], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda: self.setCurrentIndex(index)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DataParametersView:
        view = cls(parent)

        # TODO data size/rate stats live view; assembled size on memory/disk
        pages: list[DataNavigationPage] = [
            view.filePage, view.metadataPage, view.patternsPage, view.datasetPage
        ]

        for index, page in enumerate(pages):
            if index > 0:
                page.backwardButton.clicked.connect(view._createNavigationLambda(index - 1))

            if index < len(pages) - 1:
                page.forwardButton.clicked.connect(view._createNavigationLambda(index + 1))

            view.addWidget(page)

        return view
