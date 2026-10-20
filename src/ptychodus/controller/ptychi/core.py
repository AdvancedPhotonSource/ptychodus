from PyQt5.QtWidgets import QVBoxLayout, QWidget

from ...model.processing import ProcessingParameterStore

from ._settings import (
    PtyChiAutodiffSettings,
    PtyChiBHSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiRAARSettings,
    PtyChiSettings,
    PtyChiSettingsSource,
)
from ..processing import ReconstructorViewControllerFactory
from .object import PtyChiObjectViewController
from .opr import PtyChiOPRViewController
from .probe_positions import PtyChiProbePositionsViewController
from .probe import PtyChiProbeViewController
from .reconstructor import PtyChiReconstructorViewController

__all__ = ['PtyChiViewControllerFactory']


class PtyChiViewController(QWidget):
    """pty-chi's hand-laid-out panel, bound to the settings for one algorithm.

    Only the algorithm's own group is passed down; the five shared groups go
    to every panel. Which algorithm-specific group exists is decided here
    rather than read from the source, because each sub-panel shows different
    rows for each and has to be told which it is looking at.
    """

    def __init__(
        self,
        source: PtyChiSettingsSource,
        reconstructor_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        autodiff_settings: PtyChiAutodiffSettings | None = None
        bh_settings: PtyChiBHSettings | None = None
        dm_settings: PtyChiDMSettings | None = None
        lsqml_settings: PtyChiLSQMLSettings | None = None
        pie_settings: PtyChiPIESettings | None = None
        raar_settings: PtyChiRAARSettings | None = None

        match reconstructor_name.lower():
            case 'autodiff':
                autodiff_settings = PtyChiAutodiffSettings(source)
            case 'bh':
                bh_settings = PtyChiBHSettings(source)
            case 'dm':
                dm_settings = PtyChiDMSettings(source)
            case 'lsqml':
                lsqml_settings = PtyChiLSQMLSettings(source)
            case 'pie' | 'epie' | 'rpie':
                pie_settings = PtyChiPIESettings(source)
            case 'raar':
                raar_settings = PtyChiRAARSettings(source)
            case _:
                raise ValueError(f'Unknown reconstructor name: {reconstructor_name}')

        settings = PtyChiSettings(source)
        object_settings = PtyChiObjectSettings(source)
        probe_settings = PtyChiProbeSettings(source)
        probe_position_settings = PtyChiProbePositionSettings(source)
        opr_settings = PtyChiOPRSettings(source)

        self._reconstructor_view_controller = PtyChiReconstructorViewController(
            settings,
            autodiff_settings,
            bh_settings,
            dm_settings,
            lsqml_settings,
            raar_settings,
            source,
        )
        self._object_view_controller = PtyChiObjectViewController(
            object_settings,
            dm_settings,
            lsqml_settings,
            pie_settings,
            raar_settings,
            settings.num_epochs,
            source,
        )
        self._probe_view_controller = PtyChiProbeViewController(
            probe_settings,
            dm_settings,
            lsqml_settings,
            pie_settings,
            raar_settings,
            settings.num_epochs,
            source,
        )
        self._probe_positions_view_controller = PtyChiProbePositionsViewController(
            probe_position_settings,
            settings.num_epochs,
            source,
        )
        self._opr_view_controller = PtyChiOPRViewController(
            opr_settings, settings.num_epochs, source
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
    def __init__(self, parameter_store: ProcessingParameterStore) -> None:
        super().__init__()
        self._parameter_store = parameter_store

    @property
    def name(self) -> str:
        return 'pty-chi'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        return PtyChiViewController(
            PtyChiSettingsSource(self._parameter_store, self.name, reconstructor_name),
            reconstructor_name,
        )
