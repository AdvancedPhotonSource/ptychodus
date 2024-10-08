from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QTableView,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import StringParameter

from ...model.product.probe import (
    AveragePatternProbeBuilder,
    DiskProbeBuilder,
    FresnelZonePlateProbeBuilder,
    MultimodalProbeBuilder,
    ProbeModeDecayType,
    ProbeRepositoryItem,
    RectangularProbeBuilder,
    SuperGaussianProbeBuilder,
    ZernikeProbeBuilder,
)
from ...view.widgets import GroupBoxWithPresets
from ..parametric import (
    LengthWidgetParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
)
from .zernike import ZernikeTableModel

__all__ = [
    'ProbeEditorViewControllerFactory',
]


class FresnelZonePlateViewController(ParameterViewController):
    def __init__(self, title: str, probeBuilder: FresnelZonePlateProbeBuilder) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets(title)

        for index, presetsLabel in enumerate(probeBuilder.labelsForPresets()):
            action = self._widget.presetsMenu.addAction(presetsLabel)
            action.triggered.connect(lambda _, index=index: probeBuilder.applyPresets(index))

        self._zonePlateDiameterViewController = LengthWidgetParameterViewController(
            probeBuilder.zonePlateDiameterInMeters
        )
        self._outermostZoneWidthInMetersViewController = LengthWidgetParameterViewController(
            probeBuilder.outermostZoneWidthInMeters
        )
        self._centralBeamstopDiameterInMetersViewController = LengthWidgetParameterViewController(
            probeBuilder.centralBeamstopDiameterInMeters
        )
        self._defocusDistanceInMetersViewController = LengthWidgetParameterViewController(
            probeBuilder.defocusDistanceInMeters
        )

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow('Zone Plate Diameter:', self._zonePlateDiameterViewController.getWidget())
        layout.addRow(
            'Outermost Zone Width:',
            self._outermostZoneWidthInMetersViewController.getWidget(),
        )
        layout.addRow(
            'Central Beamstop Diameter:',
            self._centralBeamstopDiameterInMetersViewController.getWidget(),
        )
        layout.addRow('Defocus Distance:', self._defocusDistanceInMetersViewController.getWidget())
        self._widget.contents.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class ZernikeViewController(ParameterViewController, Observer):
    def __init__(self, title: str, probeBuilder: ZernikeProbeBuilder) -> None:
        super().__init__()
        self._widget = QGroupBox(title)
        self._probeBuilder = probeBuilder
        self._orderSpinBox = QSpinBox()
        self._coefficientsTableModel = ZernikeTableModel(probeBuilder)
        self._coefficientsTableView = QTableView()
        self._diameterViewController = LengthWidgetParameterViewController(
            probeBuilder.diameterInMeters
        )

        self._coefficientsTableView.setModel(self._coefficientsTableModel)

        layout = QFormLayout()
        layout.addRow('Diameter:', self._diameterViewController.getWidget())
        layout.addRow('Order:', self._orderSpinBox)
        layout.addRow(self._coefficientsTableView)
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._orderSpinBox.valueChanged.connect(probeBuilder.setOrder)
        probeBuilder.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._orderSpinBox.setRange(1, 100)
        self._orderSpinBox.setValue(self._probeBuilder.getOrder())

        self._coefficientsTableModel.beginResetModel()  # TODO clean up
        self._coefficientsTableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._probeBuilder:
            self._syncModelToView()


class DecayTypeParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: StringParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._polynomialDecayButton = QRadioButton('Polynomial')
        self._exponentialDecayButton = QRadioButton('Exponential')

        self._buttonGroup = QButtonGroup()
        self._buttonGroup.addButton(
            self._polynomialDecayButton, ProbeModeDecayType.POLYNOMIAL.value
        )
        self._buttonGroup.addButton(
            self._exponentialDecayButton, ProbeModeDecayType.EXPONENTIAL.value
        )
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
    def _appendAdditionalModes(
        self,
        dialogBuilder: ParameterViewBuilder,
        modesBuilder: MultimodalProbeBuilder,
    ) -> None:
        additionalModesGroup = 'Additional Modes'
        dialogBuilder.addSpinBox(
            modesBuilder.numberOfModes,
            'Number of Modes:',
            group=additionalModesGroup,
        )
        dialogBuilder.addCheckBox(
            modesBuilder.isOrthogonalizeModesEnabled,
            'Orthogonalize Modes:',
            group=additionalModesGroup,
        )
        dialogBuilder.addViewController(
            DecayTypeParameterViewController(modesBuilder.modeDecayType),
            'Decay Type:',
            group=additionalModesGroup,
        )
        dialogBuilder.addDecimalSlider(
            modesBuilder.modeDecayRatio,
            'Decay Ratio:',
            group=additionalModesGroup,
        )

    def createEditorDialog(
        self, itemName: str, item: ProbeRepositoryItem, parent: QWidget
    ) -> QDialog:
        probeBuilder = item.getBuilder()
        builderName = probeBuilder.getName()
        modesBuilder = item.getAdditionalModesBuilder()
        primaryModeGroup = 'Primary Mode'
        title = f'{itemName} [{builderName}]'

        if isinstance(probeBuilder, AveragePatternProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)
        elif isinstance(probeBuilder, DiskProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.diameterInMeters,
                'Diameter:',
                group=primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.defocusDistanceInMeters,
                'Defocus Distance:',
                group=primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)
        elif isinstance(probeBuilder, FresnelZonePlateProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.addViewControllerToTop(
                FresnelZonePlateViewController(primaryModeGroup, probeBuilder)
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)
        elif isinstance(probeBuilder, RectangularProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.widthInMeters,
                'Width:',
                group=primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.heightInMeters,
                'Height:',
                group=primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.defocusDistanceInMeters,
                'Defocus Distance:',
                group=primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)
        elif isinstance(probeBuilder, SuperGaussianProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.addLengthWidget(
                probeBuilder.annularRadiusInMeters,
                'Annular Radius:',
                group=primaryModeGroup,
            )
            dialogBuilder.addLengthWidget(
                probeBuilder.fwhmInMeters,
                'Full Width at Half Maximum:',
                group=primaryModeGroup,
            )
            dialogBuilder.addDecimalLineEdit(
                probeBuilder.orderParameter,
                'Order Parameter:',
                group=primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)
        elif isinstance(probeBuilder, ZernikeProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.addViewControllerToTop(
                ZernikeViewController(primaryModeGroup, probeBuilder)
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.buildDialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builderName}" has no editable parameters!',
            QMessageBox.Ok,
            parent,
        )
