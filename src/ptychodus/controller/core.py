from __future__ import annotations
from typing import Final

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .agent import AgentChatController, AgentController
from .automation import AutomationController
from .data import FileDialogFactory
from .diffraction import DiffractionController
from .globus import GlobusController
from .image import ImageController
from .memory import MemoryController
from .object import ObjectController
from .probe import ProbeController
from .product import ProductController
from .ptychi import PtyChiViewControllerFactory
from .ptychonn import PtychoNNViewControllerFactory
from .ptychopinn import PtychoPINNViewControllerFactory
from .processing import ProcessingController
from .probe_positions import ProbePositionsController
from .settings import SettingsController


class ControllerCore:
    ONE_HOUR_S: Final[int] = 3600

    def __init__(
        self, model: ModelCore, view: ViewCore, *, is_developer_mode_enabled: bool = False
    ) -> None:
        self.model = model
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
            model.diffraction_core.detector_settings,
            model.diffraction_core.diffraction_settings,
            model.diffraction_core.pattern_sizer,
            model.diffraction_core.bad_pixels_provider,
            model.diffraction_core.diffraction_api,
            model.diffraction_core.dataset,
            model.metadata_presenter,
            model.product_core.product_repository,
            model.analysis_core.diffraction_simulator,
            view.patterns_view,
            self._patterns_image_controller,
            self._file_dialog_factory,
        )
        self._product_controller = ProductController.create_instance(
            model.diffraction_core.diffraction_api,
            model.product_core.product_repository,
            model.product_core.product_api,
            view.product_view,
            self._file_dialog_factory,
        )
        self._probe_positions_controller = ProbePositionsController(
            model.product_core.probe_positions_repository,
            model.product_core.probe_positions_api,
            view.probe_positions_view,
            view.probe_positions_plot_view,
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
            model.product_core.probe_repository,
            model.product_core.probe_api,
            self._probe_image_controller,
            model.analysis_core.probe_propagator,
            model.analysis_core.probe_propagator_visualization_engine,
            model.analysis_core.exposure_analyzer,
            model.analysis_core.exposure_visualization_engine,
            model.fluorescence_core.enhancer,
            model.fluorescence_core.visualization_engine,
            view.probe_view,
            self._file_dialog_factory,
        )
        self._object_image_controller = ImageController(
            model.object_visualization_engine,
            view.object_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._object_controller = ObjectController(
            model.product_core.object_repository,
            model.product_core.object_api,
            self._object_image_controller,
            model.analysis_core.fourier_ring_correlator,
            model.analysis_core.fourier_analyzer,
            model.analysis_core.fourier_real_space_visualization_engine,
            model.analysis_core.fourier_reciprocal_space_visualization_engine,
            model.analysis_core.xmcd_analyzer,
            model.analysis_core.xmcd_visualization_engine,
            view.object_view,
            self._file_dialog_factory,
        )
        self._processing_controller = ProcessingController(
            model.processing_core.algorithm_parameter,
            model.processing_core.processing_api,
            model.product_core.product_repository,
            model.globus_core,
            view.processing_view,
            view.processing_status_view,
            self._file_dialog_factory,
            [
                self._ptychi_view_controller_factory,
                self._ptychopinn_view_controller_factory,
                self._ptychonn_view_controller_factory,
            ],
        )
        self._globus_controller = GlobusController(
            model.globus_core.settings,
            model.globus_core.authorizer,
            model.globus_core.status_repository,
            view.globus_view,
            view.globus_status_view,
            self._file_dialog_factory,
        )
        self._automation_controller = AutomationController(
            model.automation_core.settings,
            model.automation_core.repository,
            model.automation_core.presenter,
            view.automation_view,
            self._file_dialog_factory,
        )
        self._agent_controller = AgentController(
            model.agent_core.settings, model.agent_core.presenter, view.agent_view
        )
        self._agent_chat_controller = AgentChatController(
            model.agent_core.chat_history, model.agent_core.presenter, view.agent_chat_view
        )

        self._one_second_counter = 0
        self._run_tasks_timer = QTimer()
        self._run_tasks_timer.timeout.connect(self._run_tasks)
        self._run_tasks_timer.start(1000)

        view.globus_action.setVisible(model.globus_core.is_supported)

        self._swap_central_widgets(view.patterns_action)
        view.patterns_action.setChecked(True)
        view.navigation_action_group.triggered.connect(
            lambda action: self._swap_central_widgets(action)
        )

        view.agent_action.setVisible(is_developer_mode_enabled)
        view.probe_positions_view.button_box.analyze_button.setEnabled(is_developer_mode_enabled)

    def show_main_window(self, window_title: str) -> None:
        self.view.setWindowTitle(window_title)
        self.view.show()

    def _swap_central_widgets(self, action: QAction | None) -> None:
        if action is None:
            raise ValueError('QAction is None!')

        index = action.data()
        self.view.left_panel.setCurrentIndex(index)
        self.view.right_panel.setCurrentIndex(index)

    def _run_tasks(self) -> None:
        self.model.run_tasks()
        self._memory_controller.run_tasks(self._one_second_counter)
        self._globus_controller.run_tasks(self._one_second_counter)
        self._one_second_counter += 1

        # counter value is not intended to be precise; protect from overflow
        if self._one_second_counter > ControllerCore.ONE_HOUR_S:
            self._one_second_counter -= ControllerCore.ONE_HOUR_S
