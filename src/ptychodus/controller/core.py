from __future__ import annotations
from typing import Final

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .agent import AgentChatController, AgentController
from .automation import AutomationController
from .data import FileDialogFactory
from .diffraction.core import DiffractionController
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
from .processing.core import ReconstructorViewControllerFactory
from .processing.generated import GeneratedViewControllerFactory
from .product import ProductController
from .product.core import ProductRepositoryTableModel
from .product.visualization import ProductVisualizationController
from .rendering import VisualizationEngine
from .task_status import TaskStatusController
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
        # Every backend's panel is rendered from what it declares, so adding
        # one needs no wiring here.
        self._view_controller_factories: list[ReconstructorViewControllerFactory] = [
            GeneratedViewControllerFactory(name, model.processing_core.parameter_store)
            for name in model.processing_core.library_names()
        ]
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
            VisualizationEngine(is_complex=False),
            view.diffraction_image_view.image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        # Its own engine, not the pattern one above: autoscaling the wizard's mean-pattern
        # color range must not disturb whatever range the user set on the main pane.
        summary_visualization_engine = VisualizationEngine(is_complex=False)
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
            model.diffraction_core.summary_service,
            summary_visualization_engine,
            self._status_bar,
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
                VisualizationEngine(is_complex=False),
                VisualizationEngine(is_complex=False),
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
            VisualizationEngine(is_complex=True),
            view.probe_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        # Complex so the propagation dialog can render a single incoherent mode's
        # wavefield; Real up front because the dialog opens on the mode-summed
        # intensity, for which the is_complex default of 'Complex' paints a flat hue.
        probe_propagator_visualization_engine = VisualizationEngine(is_complex=True)
        probe_propagator_visualization_engine.set_renderer('Real')
        # Complex so the dialog can render the composed mode and the deviation from its
        # across-scan mean as wavefields: OPR moves amplitude and phase together, and the
        # default 'Complex' renderer shows both at once.
        opr_mode_visualization_engine = VisualizationEngine(is_complex=True)
        self._probe_controller = ProbeController(
            model.product_core.probe_repository,
            model.product_core.probe_api,
            self._probe_image_controller,
            model.analysis_core.probe_propagator,
            model.analysis_core.probe_propagator_settings,
            probe_propagator_visualization_engine,
            model.analysis_core.illumination_mapper,
            VisualizationEngine(is_complex=False),
            model.analysis_core.probe_overlap_analyzer,
            VisualizationEngine(is_complex=False),
            model.analysis_core.opr_mode_analyzer,
            opr_mode_visualization_engine,
            view.probe_view,
            self._file_dialog_factory,
        )
        self._object_image_controller = ImageController(
            VisualizationEngine(is_complex=True),
            view.object_image_view,
            self._status_bar,
            self._file_dialog_factory,
        )
        self._fluorescence_image_controller = ImageController(
            VisualizationEngine(is_complex=False),
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
            VisualizationEngine(is_complex=True),
            VisualizationEngine(is_complex=True),
            model.analysis_core.xmcd_analyzer,
            VisualizationEngine(is_complex=True),
            VisualizationEngine(is_complex=True),
            view.object_view,
            self._file_dialog_factory,
        )
        self._processing_controller = ProcessingController(
            model.processing_core.library_parameter,
            model.processing_core.algorithm_parameter,
            model.processing_core.processing_api,
            model.product_core.product_repository,
            self._product_table_model,
            model.globus_core,
            model.genesis_core,
            view.processing_view,
            view.processing_status_view,
            self._file_dialog_factory,
            self._view_controller_factories,
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

        action_permitted: dict[QAction, bool] = {
            view.globus_action: model.globus_core.is_supported,
            view.genesis_action: model.genesis_core.is_supported,
        }

        # Support is fixed for the life of the process, so this runs once rather than
        # on every navigation.
        for group in view.navigation.subview_groups:
            for child in group.child_actions:
                group.container.set_child_button_visible(child, action_permitted.get(child, True))

        self._swap_central_widgets(view.diffraction_action)
        view.diffraction_action.setChecked(True)
        view.navigation.action_group.triggered.connect(
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

        self.view.navigation.set_current_index(action.data())

    def _run_tasks(self) -> None:
        self.model.run_tasks()
        self._memory_controller.run_tasks(self._one_second_counter)
        self._globus_controller.run_tasks(self._one_second_counter)
        self._one_second_counter += 1

        # counter value is not intended to be precise; protect from overflow
        if self._one_second_counter > ControllerCore.ONE_HOUR_S:
            self._one_second_counter -= ControllerCore.ONE_HOUR_S
