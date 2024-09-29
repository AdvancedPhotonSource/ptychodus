from __future__ import annotations

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from ...model.tike import TikeReconstructorLibrary
from ..reconstructor import ReconstructorViewControllerFactory
from .viewControllers import (
    TikeMultigridViewController,
    TikeObjectCorrectionViewController,
    TikeParametersViewController,
    TikePositionCorrectionViewController,
    TikeProbeCorrectionViewController,
)


class TikeViewController(QWidget):
    def __init__(
        self, model: TikeReconstructorLibrary, showAlpha: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._parametersViewController = TikeParametersViewController(
            model.settings, showAlpha=showAlpha
        )
        self._multigridViewController = TikeMultigridViewController(model.multigridSettings)
        self._objectCorrectionViewController = TikeObjectCorrectionViewController(
            model.objectCorrectionSettings
        )
        self._probeCorrectionViewController = TikeProbeCorrectionViewController(
            model.probeCorrectionSettings
        )
        self._positionCorrectionViewController = TikePositionCorrectionViewController(
            model.positionCorrectionSettings
        )

        layout = QVBoxLayout()
        layout.addWidget(self._parametersViewController.getWidget())
        layout.addWidget(self._multigridViewController.getWidget())
        layout.addWidget(self._positionCorrectionViewController.getWidget())
        layout.addWidget(self._probeCorrectionViewController.getWidget())
        layout.addWidget(self._objectCorrectionViewController.getWidget())
        layout.addStretch()
        self.setLayout(layout)


class TikeViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: TikeReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[TikeViewController] = list()

    @property
    def backendName(self) -> str:
        return 'Tike'

    def createViewController(self, reconstructorName: str) -> QWidget:
        viewController = None

        if reconstructorName == 'rpie':
            viewController = TikeViewController(self._model, showAlpha=True)
        else:
            viewController = TikeViewController(self._model, showAlpha=False)

        self._controllerList.append(viewController)

        return viewController
