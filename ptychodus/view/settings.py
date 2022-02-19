from PyQt5.QtWidgets import QAbstractButton, QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget


class ImportSettingsDialog(QDialog):
    @staticmethod
    def createCheckBox(text: str) -> QCheckBox:
        widget = QCheckBox(text)
        widget.setChecked(True)
        return widget

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.instructionsLabel = QLabel('Select the settings to overwrite using metadata values:')
        self.detectorPixelSizeCheckBox = self.createCheckBox('Detector Pixel Size')
        self.detectorDistanceCheckBox = self.createCheckBox('Detector Distance')
        self.imageCropCenterCheckBox = self.createCheckBox('Image Crop Center')
        self.imageCropExtentCheckBox = self.createCheckBox('Image Crop Extent')
        self.probeEnergyCheckBox = self.createCheckBox('Probe Energy')
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None):
        view = cls(parent)

        view.setWindowTitle('Import Settings')
        view.buttonBox.addButton(QDialogButtonBox.Apply)
        view.buttonBox.addButton(QDialogButtonBox.Close)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.instructionsLabel)
        layout.addWidget(view.detectorPixelSizeCheckBox)
        layout.addWidget(view.detectorDistanceCheckBox)
        layout.addWidget(view.imageCropCenterCheckBox)
        layout.addWidget(view.imageCropExtentCheckBox)
        layout.addWidget(view.probeEnergyCheckBox)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ApplyRole:
            self.accept()
        else:
            self.reject()
