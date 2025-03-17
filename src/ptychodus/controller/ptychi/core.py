from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...model.ptychi import (
    PtyChiAutodiffSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiPIESettings,
    PtyChiReconstructorLibrary,
)

from ..reconstructor import ReconstructorViewControllerFactory
from .object import PtyChiObjectViewController
from .opr import PtyChiOPRViewController
from .positions import PtyChiProbePositionsViewController
from .probe import PtyChiProbeViewController
from .reconstructor import PtyChiReconstructorViewController

__all__ = ['PtyChiViewControllerFactory']


class PtyChiViewController(QWidget):
    def __init__(
        self,
        model: PtyChiReconstructorLibrary,
        reconstructorName: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        autodiffSettings: PtyChiAutodiffSettings | None = None
        dmSettings: PtyChiDMSettings | None = None
        lsqmlSettings: PtyChiLSQMLSettings | None = None
        pieSettings: PtyChiPIESettings | None = None

        match reconstructorName:
            case 'Autodiff':
                autodiffSettings = model.autodiffSettings
            case 'DM':
                dmSettings = model.dmSettings
            case 'LSQML':
                lsqmlSettings = model.lsqmlSettings
            case 'PIE' | 'ePIE' | 'rPIE':
                pieSettings = model.pieSettings

        # FIXME verify tooltips
        self._reconstructorViewController = PtyChiReconstructorViewController(
            model.reconstructorSettings,
            autodiffSettings,
            dmSettings,
            lsqmlSettings,
            model.enumerators,
            model.deviceRepository,
        )
        self._objectViewController = PtyChiObjectViewController(
            model.objectSettings,
            dmSettings,
            lsqmlSettings,
            pieSettings,
            model.reconstructorSettings.numEpochs,
            model.enumerators,
        )
        self._probeViewController = PtyChiProbeViewController(
            model.probeSettings,
            lsqmlSettings,
            pieSettings,
            model.reconstructorSettings.numEpochs,
            model.enumerators,
        )
        self._probePositionsViewController = PtyChiProbePositionsViewController(
            model.probePositionSettings, model.reconstructorSettings.numEpochs, model.enumerators
        )
        self._oprViewController = PtyChiOPRViewController(
            model.oprSettings, model.reconstructorSettings.numEpochs, model.enumerators
        )

        layout = QVBoxLayout()
        layout.addWidget(self._reconstructorViewController.getWidget())
        layout.addWidget(self._objectViewController.getWidget())
        layout.addWidget(self._probeViewController.getWidget())
        layout.addWidget(self._probePositionsViewController.getWidget())
        layout.addWidget(self._oprViewController.getWidget())
        layout.addStretch()
        self.setLayout(layout)


class PtyChiViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtyChiReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backend_name(self) -> str:
        return 'pty-chi'

    def create_view_controller(self, reconstructorName: str) -> QWidget:
        return PtyChiViewController(self._model, reconstructorName)
