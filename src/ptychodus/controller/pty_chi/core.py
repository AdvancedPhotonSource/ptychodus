from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...model.pty_chi import PtyChiReconstructorLibrary
from ..reconstructor import ReconstructorViewControllerFactory
from .object import PtyChiObjectViewController
from .opr import PtyChiOPRViewController
from .positions import PtyChiProbePositionsViewController
from .probe import PtyChiProbeViewController
from .reconstructor import PtyChiReconstructorViewController

__all__ = ['PtyChiViewControllerFactory']


class PtyChiViewController(QWidget):
    def __init__(self, model: PtyChiReconstructorLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reconstructorViewController = PtyChiReconstructorViewController(
            model.reconstructorSettings, model.enumerators, model.deviceRepository
        )
        self._objectViewController = PtyChiObjectViewController(
            model.objectSettings, model.enumerators
        )
        self._probeViewController = PtyChiProbeViewController(
            model.probeSettings, model.enumerators
        )
        self._probePositionsViewController = PtyChiProbePositionsViewController(
            model.probePositionSettings, model.enumerators
        )
        self._oprViewController = PtyChiOPRViewController(model.oprSettings, model.enumerators)

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
    def backendName(self) -> str:
        return 'pty-chi'

    def createViewController(self, reconstructorName: str) -> QWidget:
        return PtyChiViewController(self._model)
