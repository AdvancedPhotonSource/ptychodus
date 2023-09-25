from __future__ import annotations
from typing import Generic, Optional, TypeVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QSpinBox, QVBoxLayout, QWidget)

from .widgets import DecimalSlider, LengthWidget, RepositoryTreeView

__all__ = [
    'ObjectEditorDialog',
    'ObjectParametersView',
    'ObjectView',
    'RandomObjectView',
]

T = TypeVar('T', bound=QGroupBox)


class ObjectParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        view.setLayout(layout)

        return view


class RandomObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfSlicesSpinBox = QSpinBox()
        self.extraPaddingXSpinBox = QSpinBox()
        self.extraPaddingYSpinBox = QSpinBox()
        self.amplitudeMeanSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.amplitudeDeviationSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.phaseDeviationSlider = DecimalSlider.createInstance(Qt.Horizontal)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> RandomObjectView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Slices:', view.numberOfSlicesSpinBox)
        layout.addRow('Extra Padding X:', view.extraPaddingXSpinBox)
        layout.addRow('Extra Padding Y:', view.extraPaddingYSpinBox)
        layout.addRow('Amplitude Mean:', view.amplitudeMeanSlider)
        layout.addRow('Amplitude Deviation:', view.amplitudeDeviationSlider)
        layout.addRow('Phase Deviation:', view.phaseDeviationSlider)
        view.setLayout(layout)

        return view


class CompareObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.name1ComboBox = QComboBox()
        self.name2ComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CompareObjectView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Name 1:', view.name1ComboBox)
        layout.addRow('Name 2:', view.name2ComboBox)
        view.setLayout(layout)

        return view


class ObjectEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       title: str,
                       editorView: T,
                       parent: Optional[QWidget] = None) -> ObjectEditorDialog[T]:
        view = cls(editorView, parent)
        view.setWindowTitle(title)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ObjectView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = ObjectParametersView.createInstance()
        self.repositoryView = RepositoryTreeView.createInstance('Repository')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view
