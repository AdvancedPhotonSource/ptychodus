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

        for label in probeBuilder.labelsForPresets():
            action = self._widget.presetsMenu.addAction(label)
            action.triggered.connect(lambda _, label=label: probeBuilder.applyPresets(label))

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
        layout.addRow('Zone Plate Diameter:', self._zonePlateDiameterViewController.get_widget())
        layout.addRow(
            'Outermost Zone Width:',
            self._outermostZoneWidthInMetersViewController.get_widget(),
        )
        layout.addRow(
            'Central Beamstop Diameter:',
            self._centralBeamstopDiameterInMetersViewController.get_widget(),
        )
        layout.addRow('Defocus Distance:', self._defocusDistanceInMetersViewController.get_widget())
        self._widget.contents.setLayout(layout)

    def get_widget(self) -> QWidget:
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
        layout.addRow('Diameter:', self._diameterViewController.get_widget())
        layout.addRow('Order:', self._orderSpinBox)
        layout.addRow(self._coefficientsTableView)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        self._orderSpinBox.valueChanged.connect(probeBuilder.setOrder)
        probeBuilder.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._orderSpinBox.setRange(1, 100)
        self._orderSpinBox.setValue(self._probeBuilder.getOrder())

        self._coefficientsTableModel.beginResetModel()  # TODO clean up
        self._coefficientsTableModel.endResetModel()

    def _update(self, observable: Observable) -> None:
        if observable is self._probeBuilder:
            self._sync_model_to_view()


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
        self._buttonGroup.idToggled.connect(self._sync_view_to_model)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._polynomialDecayButton)
        layout.addWidget(self._exponentialDecayButton)

        self._widget = QWidget()
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, toolId: int, checked: bool) -> None:
        if checked:
            decayType = ProbeModeDecayType(toolId)
            self._parameter.set_value(decayType.name)

    def _sync_model_to_view(self) -> None:
        try:
            decayType = ProbeModeDecayType[self._parameter.get_value().upper()]
        except KeyError:
            decayType = ProbeModeDecayType.POLYNOMIAL

        button = self._buttonGroup.button(decayType.value)
        button.setChecked(True)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class ProbeEditorViewControllerFactory:
    def _appendAdditionalModes(
        self,
        dialogBuilder: ParameterViewBuilder,
        modesBuilder: MultimodalProbeBuilder,
    ) -> None:
        additionalModesGroup = 'Additional Modes'  # FIXME OPR
        dialogBuilder.add_spin_box(
            modesBuilder.numberOfIncoherentModes,
            'Number of Modes:',
            group=additionalModesGroup,
        )
        dialogBuilder.add_check_box(
            modesBuilder.orthogonalizeIncoherentModes,
            'Orthogonalize Modes:',
            group=additionalModesGroup,
        )
        dialogBuilder.add_view_controller(
            DecayTypeParameterViewController(modesBuilder.incoherentModeDecayType),
            'Decay Type:',
            group=additionalModesGroup,
        )
        dialogBuilder.add_decimal_slider(
            modesBuilder.incoherentModeDecayRatio,
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
            return dialogBuilder.build_dialog(title, parent)
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
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(probeBuilder, FresnelZonePlateProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_view_controller_to_top(
                FresnelZonePlateViewController(primaryModeGroup, probeBuilder)
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build_dialog(title, parent)
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
            return dialogBuilder.build_dialog(title, parent)
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
            dialogBuilder.add_decimal_line_edit(
                probeBuilder.orderParameter,
                'Order Parameter:',
                group=primaryModeGroup,
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(probeBuilder, ZernikeProbeBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_view_controller_to_top(
                ZernikeViewController(primaryModeGroup, probeBuilder)
            )
            self._appendAdditionalModes(dialogBuilder, modesBuilder)
            return dialogBuilder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builderName}" has no editable parameters!',
            QMessageBox.Ok,
            parent,
        )
