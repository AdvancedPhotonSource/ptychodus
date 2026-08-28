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
from .fluorescence import FluorescenceController
from .fluorescence.enhance_dialog import FluorescenceEnhanceDialogController
from .genesis import GenesisController
from .globus import GlobusController
from .image import ImageController
from .memory import MemoryController
from .object import ObjectController
from .probe import ProbeController
from .probe_positions import ProbePositionsController
from .helpers import create_brush_for_editable_cell
from .processing import ProcessingController
from .product import ProductController
from .product.core import ProductRepositoryTableModel
from .product.visualization import ProductVisualizationController
from .task_status import TaskStatusController
from .ptycho_fm import PtychoFMViewControllerFactory
from .ptychi import PtyChiViewControllerFactory
from .ptychonn import PtychoNNViewControllerFactory
from .ptychopinn import PtychoPINNViewControllerFactory
from .ptychopinn_torch import PtychoPINNTorchViewControllerFactory
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
        self._ptychopinn_torch_view_controller_factory = PtychoPINNTorchViewControllerFactory(
            model.ptychopinn_torch_reconstructor_library,
            self._file_dialog_factory,
        )
        self._ptycho_fm_view_controller_factory = PtychoFMViewControllerFactory(
            model.ptycho_fm_reconstructor_library,
            self._file_dialog_factory,
        )
        # Shared product-repository table model. Constructing it here (before
        # any consumer) lets SettingsController, DiffractionController,
        # ProductController, and ProcessingController all bind to the same
        # instance — one observer registration on the repository serves every
        # widget that shows product rows.
        self._product_table_model = ProductRepositoryTableModel(
            model.product_core.product_repository,
            model.diffraction_core.repository,
            create_brush_for_editable_cell(view.product_view.table_view),
        )
        self._settings_controller = SettingsController(
            model.settings_registry,
            model.product_core.product_repository,
            self._product_table_model,
            view.settings_view,
            view.settings_table_view,
            self._file_dialog_factory,
        )
        self._diffraction_image_controller = ImageController(
            model.pattern_visualization_engine,
            view.diffraction_image_view.image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._diffraction_controller = DiffractionController(
            model.diffraction_core.detector_settings,
            model.diffraction_core.diffraction_settings,
            model.product_core.settings,
            model.diffraction_core.diffraction_api,
            model.diffraction_core.repository,
            model.diffraction_core.task_monitor,
            model.product_core.product_repository,
            self._product_table_model,
            model.analysis_core.diffraction_simulator,
            model.analysis_core.diffraction_simulator_settings,
            view.diffraction_view,
            view.diffraction_image_view.status_view,
            self._diffraction_image_controller,
            self._file_dialog_factory,
        )
        self._product_controller = ProductController.create_instance(
            model.product_core.product_repository,
            model.product_core.product_api,
            model.diffraction_core.repository,
            self._product_table_model,
            view.product_view,
            self._file_dialog_factory,
        )
        self._product_status_controller = TaskStatusController(
            model.product_core.task_monitor,
            view.product_right_view.status_view,
        )
        if is_developer_mode_enabled:
            self._product_visualization_controller = ProductVisualizationController(
                model.analysis_core.residual_analyzer,
                model.analysis_core.residual_real_space_visualization_engine,
                model.analysis_core.residual_reciprocal_space_visualization_engine,
                self._product_controller,
                view.product_visualization_view,
                self._status_bar,
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
            model.analysis_core.probe_propagator_settings,
            model.analysis_core.probe_propagator_visualization_engine,
            model.analysis_core.illumination_mapper,
            model.analysis_core.illumination_visualization_engine,
            view.probe_view,
            self._file_dialog_factory,
        )
        self._object_image_controller = ImageController(
            model.object_visualization_engine,
            view.object_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._fluorescence_image_controller = ImageController(
            model.fluorescence_core.visualization_engine,
            view.fluorescence_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._fluorescence_enhance_dialog_controller = FluorescenceEnhanceDialogController(
            model.fluorescence_core,
            has_ptychozoon=model.fluorescence_core.ptychozoon_enhancer is not None,
        )
        self._fluorescence_controller = FluorescenceController(
            model.fluorescence_core.repository,
            model.fluorescence_core.fluorescence_api,
            model.product_core.product_repository,
            self._product_table_model,
            view.fluorescence_view,
            self._fluorescence_image_controller,
            self._fluorescence_enhance_dialog_controller,
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
            model.analysis_core.xmcd_structural_visualization_engine,
            model.analysis_core.xmcd_magnetic_visualization_engine,
            view.object_view,
            self._file_dialog_factory,
        )
        self._processing_controller = ProcessingController(
            model.processing_core.algorithm_parameter,
            model.processing_core.processing_api,
            model.product_core.product_repository,
            self._product_table_model,
            model.globus_core,
            model.genesis_core,
            view.processing_view,
            view.processing_status_view,
            self._file_dialog_factory,
            [
                self._ptychi_view_controller_factory,
                self._ptychopinn_torch_view_controller_factory,
                self._ptychopinn_view_controller_factory,
                self._ptychonn_view_controller_factory,
                self._ptycho_fm_view_controller_factory,
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
        self._genesis_controller = GenesisController(
            model.genesis_core.settings,
            model.genesis_core.presenter,
            model.genesis_core.status_repository,
            view.genesis_view,
            view.genesis_status_view,
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
            model.agent_core.settings, model.agent_core.catalog, view.agent_view
        )
        self._agent_chat_controller = AgentChatController(
            model.agent_core.repository, model.agent_core.terminal, view.agent_chat_view
        )

        self._one_second_counter = 0
        self._run_tasks_timer = QTimer()
        self._run_tasks_timer.timeout.connect(self._run_tasks)
        self._run_tasks_timer.start(1000)

        self._action_permitted: dict[QAction, bool] = {
            view.globus_action: model.globus_core.is_supported,
            view.genesis_action: model.genesis_core.is_supported,
        }

        self._swap_central_widgets(view.diffraction_action, animated=False)
        view.diffraction_action.setChecked(True)
        view.navigation.action_group.triggered.connect(
            lambda action: self._swap_central_widgets(action)
        )

        view.agent_action.setVisible(is_developer_mode_enabled)
        view.probe_positions_view.button_box.analyze_button.setEnabled(is_developer_mode_enabled)

    def show_main_window(self, window_title: str) -> None:
        self.view.setWindowTitle(window_title)
        self.view.show()

    def _swap_central_widgets(self, action: QAction | None, *, animated: bool = True) -> None:
        if action is None:
            raise ValueError('QAction is None!')

        self.view.navigation.set_current_index(action.data())
        self._update_subview_visibility(action, animated=animated)

    def _update_subview_visibility(self, action: QAction, *, animated: bool = True) -> None:
        for group in self.view.navigation.subview_groups:
            expanded = action is group.parent_action or action in group.child_actions
            for child in group.child_actions:
                allowed = self._action_permitted.get(child, True)
                group.container.set_child_button_visible(child, allowed)
            group.container.set_expanded(expanded, animated=animated)
            group.top_separator.setVisible(expanded)
            group.bottom_separator.setVisible(expanded)

    def _run_tasks(self) -> None:
        self.model.run_tasks()
        self._memory_controller.run_tasks(self._one_second_counter)
        self._globus_controller.run_tasks(self._one_second_counter)
        self._one_second_counter += 1

        # counter value is not intended to be precise; protect from overflow
        if self._one_second_counter > ControllerCore.ONE_HOUR_S:
            self._one_second_counter -= ControllerCore.ONE_HOUR_S
