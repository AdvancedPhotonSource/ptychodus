from PyQt5.QtWidgets import QVBoxLayout, QWidget

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
        layout.addWidget(self._parametersViewController.get_widget())
        layout.addWidget(self._multigridViewController.get_widget())
        layout.addWidget(self._positionCorrectionViewController.get_widget())
        layout.addWidget(self._probeCorrectionViewController.get_widget())
        layout.addWidget(self._objectCorrectionViewController.get_widget())
        layout.addStretch()
        self.setLayout(layout)


class TikeViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: TikeReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backend_name(self) -> str:
        return 'Tike'

    def create_view_controller(self, reconstructorName: str) -> QWidget:
        if reconstructorName == 'rpie':
            viewController = TikeViewController(self._model, showAlpha=True)
        else:
            viewController = TikeViewController(self._model, showAlpha=False)

        return viewController
