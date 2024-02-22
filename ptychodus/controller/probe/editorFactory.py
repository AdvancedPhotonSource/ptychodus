from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.probe import (DiskProbeBuilder, FresnelZonePlateProbeBuilder, ProbeRepositoryItem,
                            SuperGaussianProbeBuilder)
from ..parametric import ParameterDialogBuilder


class ProbeEditorViewControllerFactory:

    def createEditorDialog(self, itemName: str, item: ProbeRepositoryItem,
                           parent: QWidget) -> QDialog:
        probeBuilder = item.getBuilder()
        builderName = probeBuilder.getName()
        title = f'{builderName}: {itemName}'

        if isinstance(probeBuilder, DiskProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget('Diameter', probeBuilder.diameterInMeters)
            return dialogBuilder.build(title, parent)
        elif isinstance(probeBuilder, FresnelZonePlateProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget('Zone Plate Diameter',
                                          probeBuilder.zonePlateDiameterInMeters)
            dialogBuilder.addLengthWidget('Outermost Zone Width',
                                          probeBuilder.outermostZoneWidthInMeters)
            dialogBuilder.addLengthWidget('Central Beamstop Diameter',
                                          probeBuilder.centralBeamstopDiameterInMeters)
            dialogBuilder.addLengthWidget('Defocus Distance', probeBuilder.defocusDistanceInMeters)
            return dialogBuilder.build(title, parent)
        elif isinstance(probeBuilder, SuperGaussianProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addLengthWidget('Annular Radius', probeBuilder.annularRadiusInMeters)
            dialogBuilder.addLengthWidget('Full Width at Half Maximum', probeBuilder.fwhmInMeters)
            dialogBuilder.addDecimalLineEdit('Order Parameter', probeBuilder.orderParameter)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
