from __future__ import annotations

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from ...model.ptychopack import PtychoPackReconstructorLibrary
from ..reconstructor import ReconstructorViewControllerFactory
from .viewControllers import (
    PtychoPackAlgorithm,
    PtychoPackExitWaveCorrectionViewController,
    PtychoPackObjectCorrectionViewController,
    PtychoPackParametersViewController,
    PtychoPackPositionCorrectionViewController,
    PtychoPackProbeCorrectionViewController,
)


class PtychoPackViewController(QWidget):
    def __init__(
        self,
        model: PtychoPackReconstructorLibrary,
        algorithm: PtychoPackAlgorithm,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.parametersViewController = PtychoPackParametersViewController(model.presenter)
        self.objectViewController = PtychoPackObjectCorrectionViewController(
            model.settings, algorithm
        )
        self.probeViewController = PtychoPackProbeCorrectionViewController(
            model.settings, algorithm
        )
        self.positionViewController = PtychoPackPositionCorrectionViewController(model.settings)

        layout = QVBoxLayout()
        layout.addWidget(self.parametersViewController.getWidget())

        if algorithm in (PtychoPackAlgorithm.DM, PtychoPackAlgorithm.RAAR):
            self.exit_waveViewController = PtychoPackExitWaveCorrectionViewController(
                model.settings, algorithm
            )
            layout.addWidget(self.exit_waveViewController.getWidget())

        layout.addWidget(self.objectViewController.getWidget())
        layout.addWidget(self.probeViewController.getWidget())
        layout.addWidget(self.positionViewController.getWidget())
        layout.addStretch()
        self.setLayout(layout)


class PtychoPackViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtychoPackReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[PtychoPackViewController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoPack'

    def createViewController(self, reconstructorName: str) -> QWidget:
        if reconstructorName.casefold() == 'dm':
            viewController = PtychoPackViewController(self._model, PtychoPackAlgorithm.DM)
        elif reconstructorName.casefold() == 'raar':
            viewController = PtychoPackViewController(self._model, PtychoPackAlgorithm.RAAR)
        else:
            viewController = PtychoPackViewController(self._model, PtychoPackAlgorithm.PIE)

        self._controllerList.append(viewController)
        return viewController
