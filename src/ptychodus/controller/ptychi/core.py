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
        reconstructor_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        autodiff_settings: PtyChiAutodiffSettings | None = None
        dm_settings: PtyChiDMSettings | None = None
        lsqml_settings: PtyChiLSQMLSettings | None = None
        pie_settings: PtyChiPIESettings | None = None

        match reconstructor_name:
            case 'Autodiff':
                autodiff_settings = model.autodiff_settings
            case 'DM':
                dm_settings = model.dm_settings
            case 'LSQML':
                lsqml_settings = model.lsqml_settings
            case 'PIE' | 'ePIE' | 'rPIE':
                pie_settings = model.pie_settings

        self._reconstructor_view_controller = PtyChiReconstructorViewController(
            model.settings,
            autodiff_settings,
            dm_settings,
            lsqml_settings,
            model.enumerators,
            model.device_repository,
        )
        self._object_view_controller = PtyChiObjectViewController(
            model.object_settings,
            dm_settings,
            lsqml_settings,
            pie_settings,
            model.settings.num_epochs,
            model.enumerators,
        )
        self._probe_view_controller = PtyChiProbeViewController(
            model.probe_settings,
            dm_settings,
            lsqml_settings,
            pie_settings,
            model.settings.num_epochs,
            model.enumerators,
        )
        self._probe_positions_view_controller = PtyChiProbePositionsViewController(
            model.probe_position_settings,
            model.settings.num_epochs,
            model.enumerators,
        )
        self._opr_view_controller = PtyChiOPRViewController(
            model.opr_settings, model.settings.num_epochs, model.enumerators
        )

        layout = QVBoxLayout()
        layout.addWidget(self._reconstructor_view_controller.get_widget())
        layout.addWidget(self._object_view_controller.get_widget())
        layout.addWidget(self._probe_view_controller.get_widget())
        layout.addWidget(self._probe_positions_view_controller.get_widget())
        layout.addWidget(self._opr_view_controller.get_widget())
        layout.addStretch()
        self.setLayout(layout)


class PtyChiViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtyChiReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backend_name(self) -> str:
        return 'pty-chi'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        return PtyChiViewController(self._model, reconstructor_name)
