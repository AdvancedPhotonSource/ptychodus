from __future__ import annotations

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from ...model.tike import TikeReconstructorLibrary


class TikeView(QWidget):
    def __init__(self, showAlpha: bool, showStepLength: bool, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.parametersView = TikeParametersView.createInstance(showAlpha, showStepLength)
        self.multigridView = TikeMultigridView.createInstance()
        self.positionCorrectionView = TikePositionCorrectionView.createInstance()
        self.probeCorrectionView = TikeProbeCorrectionView.createInstance()
        self.objectCorrectionView = TikeObjectCorrectionView.createInstance()

    @classmethod
    def createInstance(
        cls, showAlpha: bool, showStepLength: bool, parent: QWidget | None = None
    ) -> TikeView:
        view = cls(showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.multigridView)
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class TikeController:
    def __init__(self, model: TikeReconstructorLibrary, view: TikeView) -> None:
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

        self._parametersController = TikeParametersController.createInstance(
            model.parametersPresenter, view.parametersView
        )
