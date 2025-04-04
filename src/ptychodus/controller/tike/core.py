from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...model.tike import TikeReconstructorLibrary
from ..reconstructor import ReconstructorViewControllerFactory
from .view_controllers import (
    TikeMultigridViewController,
    TikeObjectCorrectionViewController,
    TikeParametersViewController,
    TikePositionCorrectionViewController,
    TikeProbeCorrectionViewController,
)


class TikeViewController(QWidget):
    def __init__(
        self, model: TikeReconstructorLibrary, show_alpha: bool, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._parameters_view_controller = TikeParametersViewController(
            model.settings, show_alpha=show_alpha
        )
        self._multigrid_view_controller = TikeMultigridViewController(model.multigrid_settings)
        self._object_correction_view_controller = TikeObjectCorrectionViewController(
            model.object_correction_settings
        )
        self._probe_correction_view_controller = TikeProbeCorrectionViewController(
            model.probe_correction_settings
        )
        self._position_correction_view_controller = TikePositionCorrectionViewController(
            model.position_correction_settings
        )

        layout = QVBoxLayout()
        layout.addWidget(self._parameters_view_controller.get_widget())
        layout.addWidget(self._multigrid_view_controller.get_widget())
        layout.addWidget(self._position_correction_view_controller.get_widget())
        layout.addWidget(self._probe_correction_view_controller.get_widget())
        layout.addWidget(self._object_correction_view_controller.get_widget())
        layout.addStretch()
        self.setLayout(layout)


class TikeViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: TikeReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backend_name(self) -> str:
        return 'Tike'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        if reconstructor_name == 'rpie':
            view_controller = TikeViewController(self._model, show_alpha=True)
        else:
            view_controller = TikeViewController(self._model, show_alpha=False)

        return view_controller
