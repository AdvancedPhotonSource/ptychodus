from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .agent import AgentChatController, AgentController
from .automation import AutomationController
from .data import FileDialogFactory
from .image import ImageController
from .memory import MemoryController
from .object import ObjectController
from .diffraction import DiffractionController
from .probe import ProbeController
from .product import ProductController
from .ptychi import PtyChiViewControllerFactory
from .ptychonn import PtychoNNViewControllerFactory
from .ptychopinn import PtychoPINNViewControllerFactory
from .reconstructor import ReconstructorController
from .scan import ScanController
from .settings import SettingsController
from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:
    def __init__(
        self, model: ModelCore, view: ViewCore, *, is_developer_mode_enabled: bool = False
    ) -> None:
        self.view = view
        self._status_bar = view.statusBar()

        if self._status_bar is None:
            raise ValueError('QStatusBar is None!')

        self._memory_controller = MemoryController(model.memory_presenter, view.memory_widget)
        self._file_dialog_factory = FileDialogFactory()
        self._ptychi_view_controller_factory = PtyChiViewControllerFactory(
            model.ptychi_reconstructor_library
        )
        self._ptychonn_view_controller_factory = PtychoNNViewControllerFactory(
            model.ptychonn_reconstructor_library
        )
        self._ptychopinn_view_controller_factory = PtychoPINNViewControllerFactory(
            model.ptychopinn_reconstructor_library, self._file_dialog_factory
        )
        self._tike_view_controller_factory = TikeViewControllerFactory(
            model.tike_reconstructor_library
        )
        self._settings_controller = SettingsController(
            model.settings_registry,
            view.settings_view,
            view.settings_table_view,
            self._file_dialog_factory,
        )
        self._patterns_image_controller = ImageController(
            model.pattern_visualization_engine,
            view.patterns_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._patterns_controller = DiffractionController(
            model.diffraction.detector_settings,
            model.diffraction.diffraction_settings,
            model.diffraction.pattern_sizer,
            model.diffraction.diffraction_api,
            model.diffraction.dataset,
            model.metadata_presenter,
            view.patterns_view,
            self._patterns_image_controller,
            self._file_dialog_factory,
        )
        self._product_controller = ProductController.create_instance(
            model.diffraction.dataset,
            model.product.product_repository,
            model.product.product_api,
            view.product_view,
            self._file_dialog_factory,
        )
        self._scan_controller = ScanController(
            model.product.scan_repository,
            model.product.scan_api,
            view.scan_view,
            view.scan_plot_view,
            self._file_dialog_factory,
            is_developer_mode_enabled=is_developer_mode_enabled,
        )
        self._probe_image_controller = ImageController(
            model.probe_visualization_engine,
            view.probe_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._probe_controller = ProbeController(
            model.product.probe_repository,
            model.product.probe_api,
            self._probe_image_controller,
            model.analysis.probe_propagator,
            model.analysis.probe_propagator_visualization_engine,
            model.analysis.stxm_simulator,
            model.analysis.stxm_visualization_engine,
            model.analysis.exposure_analyzer,
            model.analysis.exposure_visualization_engine,
            model.fluorescence_core.enhancer,
            model.fluorescence_core.visualization_engine,
            view.probe_view,
            self._file_dialog_factory,
            is_developer_mode_enabled=is_developer_mode_enabled,
        )
        self._object_image_controller = ImageController(
            model.object_visualization_engine,
            view.object_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._object_controller = ObjectController(
            model.product.object_repository,
            model.product.object_api,
            self._object_image_controller,
            model.analysis.fourier_ring_correlator,
            model.analysis.fourier_analyzer,
            model.analysis.fourier_real_space_visualization_engine,
            model.analysis.fourier_reciprocal_space_visualization_engine,
            model.analysis.xmcd_analyzer,
            model.analysis.xmcd_visualization_engine,
            view.object_view,
            self._file_dialog_factory,
            is_developer_mode_enabled=is_developer_mode_enabled,
        )
        self._reconstructor_controller = ReconstructorController(
            model.reconstructor.presenter,
            model.product.product_repository,
            view.reconstructor_view,
            view.reconstructor_plot_view,
            self._product_controller.table_model,
            self._file_dialog_factory,
            [
                self._ptychi_view_controller_factory,
                self._ptychopinn_view_controller_factory,
                self._ptychonn_view_controller_factory,
                self._tike_view_controller_factory,
            ],
        )
        self._workflow_controller = WorkflowController(
            model.workflow.parameters_presenter,
            model.workflow.authorization_presenter,
            model.workflow.status_presenter,
            model.workflow.execution_presenter,
            view.workflow_parameters_view,
            view.workflow_table_view,
            self._product_controller.table_model,
        )
        self._automation_controller = AutomationController.create_instance(
            model.automation,
            model.automation.presenter,
            model.automation.processing_presenter,
            view.automation_view,
            self._file_dialog_factory,
        )
        self._agent_controller = AgentController(
            model.agent.settings, model.agent.presenter, view.agent_view
        )
        self._agent_chat_controller = AgentChatController(
            model.agent.chat_history, model.agent.presenter, view.agent_chat_view
        )

        self._refresh_data_timer = QTimer()
        self._refresh_data_timer.timeout.connect(model.refresh_active_dataset)
        self._refresh_data_timer.start(1000)  # TODO make configurable

        view.workflow_action.setVisible(model.workflow.is_supported)

        self._swap_central_widgets(view.patterns_action)
        view.patterns_action.setChecked(True)
        view.navigation_action_group.triggered.connect(
            lambda action: self._swap_central_widgets(action)
        )

        view.agent_action.setVisible(is_developer_mode_enabled)
        view.scan_view.button_box.analyze_button.setEnabled(is_developer_mode_enabled)

    def show_main_window(self, window_title: str) -> None:
        self.view.setWindowTitle(window_title)
        self.view.show()

    def _swap_central_widgets(self, action: QAction | None) -> None:
        if action is None:
            raise ValueError('QAction is None!')

        index = action.data()
        self.view.left_panel.setCurrentIndex(index)
        self.view.right_panel.setCurrentIndex(index)
