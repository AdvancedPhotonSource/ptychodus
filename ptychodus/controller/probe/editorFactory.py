from PyQt5.QtWidgets import QButtonGroup, QDialog, QHBoxLayout, QMessageBox, QRadioButton, QWidget

from ...api.observer import Observable, Observer
from ...api.parametric import StringParameter
from ...model.probe import (DiskProbeBuilder, FresnelZonePlateProbeBuilder, MultimodalProbeBuilder,
                            ProbeModeDecayType, ProbeRepositoryItem, RectangularProbeBuilder,
                            SuperGaussianProbeBuilder)
from ..parametric import ParameterDialogBuilder, ParameterViewController


class DecayTypeParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: StringParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._polynomialDecayButton = QRadioButton('Polynomial')
        self._exponentialDecayButton = QRadioButton('Exponential')

        self._buttonGroup = QButtonGroup()
        self._buttonGroup.addButton(self._polynomialDecayButton,
                                    ProbeModeDecayType.POLYNOMIAL.value)
        self._buttonGroup.addButton(self._exponentialDecayButton,
                                    ProbeModeDecayType.EXPONENTIAL.value)
        self._buttonGroup.setExclusive(True)
        self._buttonGroup.idToggled.connect(self._syncViewToModel)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._polynomialDecayButton)
        layout.addWidget(self._exponentialDecayButton)

        self._widget = QWidget()
        self._widget.setLayout(layout)

        self._syncModelToView()
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, toolId: int, checked: bool) -> None:
        if checked:
            decayType = ProbeModeDecayType(toolId)
            self._parameter.setValue(decayType.name)

    def _syncModelToView(self) -> None:
        try:
            decayType = ProbeModeDecayType[self._parameter.getValue().upper()]
        except KeyError:
            decayType = ProbeModeDecayType.POLYNOMIAL

        button = self._buttonGroup.button(decayType.value)
        button.setChecked(True)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class ProbeEditorViewControllerFactory:

    def _appendAdditionalModes(self, dialogBuilder: ParameterDialogBuilder,
                               modesBuilder: MultimodalProbeBuilder) -> None:
        additionalModesGroup = 'Additional Modes'
        dialogBuilder.addSpinBox(
            modesBuilder.numberOfModes,
            'Number of Modes:',
            additionalModesGroup,
        )
        dialogBuilder.addCheckBox(
            modesBuilder.isOrthogonalizeModesEnabled,
            'Orthogonalize Modes:',
            additionalModesGroup,
        )
        dialogBuilder.addViewController(
            DecayTypeParameterViewController(modesBuilder.modeDecayType),
            'Decay Type:',
            additionalModesGroup,
        )
        dialogBuilder.addDecimalSlider(
            modesBuilder.modeDecayRatio,
            'Decay Ratio:',
            additionalModesGroup,
        )

    def createEditorDialog(self, itemName: str, item: ProbeRepositoryItem,
                           parent: QWidget) -> QDialog:
        probeBuilder = item.getBuilder()
        builderName = probeBuilder.getName()
        modesBuilder = item.getAdditionalModesBuilder()
        primaryModeGroup = 'Primary Mode'
        title = f'{itemName} [{builderName}]'

        if isinstance(probeBuilder, DiskProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.diameterInMeters,
                'Diameter:',
                primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(probeBuilder, FresnelZonePlateProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.zonePlateDiameterInMeters,
                'Zone Plate Diameter:',
                primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.outermostZoneWidthInMeters,
                'Outermost Zone Width:',
                primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.centralBeamstopDiameterInMeters,
                'Central Beamstop Diameter:',
                primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.defocusDistanceInMeters,
                'Defocus Distance:',
                primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(probeBuilder, RectangularProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.widthInMeters,
                'Width:',
                primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.heightInMeters,
                'Height:',
                primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(probeBuilder, SuperGaussianProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.annularRadiusInMeters,
                'Annular Radius:',
                primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.fwhmInMeters,
                'Full Width at Half Maximum:',
                primaryModeGroup,
            )
            dialogBuilder.addDecimalLineEdit(
                probeBuilder.orderParameter,
                'Order Parameter:',
                primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'\"{builderName}\" has no editable parameters!',
            QMessageBox.Ok,
            parent,
        )
