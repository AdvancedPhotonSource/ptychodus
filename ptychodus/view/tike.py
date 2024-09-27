from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class TikeParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__("Tike Parameters", parent)
        self.numGpusLineEdit = QLineEdit()
        self.noiseModelComboBox = QComboBox()
        self.numBatchSpinBox = QSpinBox()
        self.batchMethodComboBox = QComboBox()
        self.numIterSpinBox = QSpinBox()
        self.convergenceWindowSpinBox = QSpinBox()
        self.alphaSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.stepLengthSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.logLevelComboBox = QComboBox()

    @classmethod
    def createInstance(cls,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeParametersView:
        view = cls(parent)

        view.numGpusLineEdit.setToolTip(
            "The number of GPUs to use. If the number of GPUs is less than the requested number, "
            "only workers for the available GPUs are allocated.")
        view.noiseModelComboBox.setToolTip("The noise model to use for the cost function.")
        view.numBatchSpinBox.setToolTip("The dataset is divided into this number of groups "
                                        "where each group is processed sequentially.")
        view.batchMethodComboBox.setToolTip("The name of the batch selection method.")
        view.numIterSpinBox.setToolTip("The number of epochs to process before returning.")
        view.convergenceWindowSpinBox.setToolTip(
            "The number of epochs to consider for convergence monitoring. "
            "Set to any value less than 2 to disable.")
        view.alphaSlider.setToolTip("RPIE becomes EPIE when this parameter is 1.")
        view.stepLengthSlider.setToolTip(
            "Scales the inital search directions before the line search.")

        layout = QFormLayout()
        layout.addRow("Number of GPUs:", view.numGpusLineEdit)
        layout.addRow("Noise Model:", view.noiseModelComboBox)
        layout.addRow("Number of Batches:", view.numBatchSpinBox)
        layout.addRow("Batch Method:", view.batchMethodComboBox)
        layout.addRow("Number of Iterations:", view.numIterSpinBox)
        layout.addRow("Convergence Window:", view.convergenceWindowSpinBox)

        if showAlpha:
            layout.addRow("Alpha:", view.alphaSlider)

        if showStepLength:
            layout.addRow("Step Length:", view.stepLengthSlider)

        layout.addRow("Log Level:", view.logLevelComboBox)

        view.setLayout(layout)

        return view


class TikeView(QWidget):

    def __init__(self, showAlpha: bool, showStepLength: bool, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = TikeParametersView.createInstance(showAlpha, showStepLength)
        self.multigridView = TikeMultigridView.createInstance()
        self.positionCorrectionView = TikePositionCorrectionView.createInstance()
        self.probeCorrectionView = TikeProbeCorrectionView.createInstance()
        self.objectCorrectionView = TikeObjectCorrectionView.createInstance()

    @classmethod
    def createInstance(cls,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeView:
        view = cls(showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.multigridView)
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addStretch()
        view.setLayout(layout)

        return view
