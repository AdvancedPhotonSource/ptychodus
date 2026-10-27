from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstruct import (
    ReconstructorLibrary,
    SubprocessReconstructor,
    TrainableReconstructor,
)

from ...model.genesis import GenesisCore
from ...model.globus import GlobusCore
from ...model.parameters import PluginChooserParameter
from ...model.processing import ProcessingAPI, ProcessingAlgorithmParameter
from ...model.product import ProductRepository
from ...view.processing import ProcessingActionsView, ProcessingStatusView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import connect_triggered_signal
from ..parameters import ComboBoxParameterViewController, ParameterViewController
from ..product.core import ProductRepositoryTableModel
from .parameters import (
    ProcessingStatusController,
    ProductParameterViewController,
    ComputeParameterViewController,
)
from .train import TrainDialogController

logger = logging.getLogger(__name__)


def _centered_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class _AlgorithmComboViewController(ParameterViewController, Observer):
    """A combo box whose items change with the library that offers them.

    ``ComboBoxParameterViewController`` fixes its items at construction,
    which an algorithm list cannot be: selecting another backend replaces
    every choice. Kept here rather than beside that one because the shared
    module is tested and this is panel wiring.
    """

    def __init__(self, parameter: ProcessingAlgorithmParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QComboBox()
        self._repopulate()
        # textActivated, not currentTextChanged: it fires only when a person
        # picks something, so rebuilding the items below cannot write back.
        self._widget.textActivated.connect(parameter.set_value)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _repopulate(self) -> None:
        choices = list(self._parameter.choices())
        current = self._parameter.get_value()

        self._widget.blockSignals(True)

        if [self._widget.itemText(row) for row in range(self._widget.count())] != choices:
            self._widget.clear()
            self._widget.addItems(choices)

        self._widget.setCurrentText(current)
        self._widget.setEnabled(bool(choices))
        self._widget.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._repopulate()


class ReconstructorViewControllerFactory(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        pass


class ProcessingController(Observer):
    def __init__(
        self,
        library_parameter: PluginChooserParameter[ReconstructorLibrary],
        algorithm_parameter: ProcessingAlgorithmParameter,
        processing_api: ProcessingAPI,
        product_repository: ProductRepository,
        product_table_model: ProductRepositoryTableModel,
        globus: GlobusCore,
        genesis: GenesisCore,
        view: QWidget,
        status_view: ProcessingStatusView,
        file_dialog_factory: FileDialogFactory,
        view_controller_factories: Iterable[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._library_parameter = library_parameter
        self._algorithm_parameter = algorithm_parameter
        self._processing_api = processing_api
        self._product_repository = product_repository
        self._globus = globus
        self._genesis = genesis
        self._view = view
        self._status_view = status_view
        self._file_dialog_factory = file_dialog_factory
        self._train_dialog_controller = TrainDialogController(view, file_dialog_factory)

        self._stacked_widget = QStackedWidget()
        self._page: dict[tuple[str, str], int] = {}
        self._empty_page: dict[str, int] = {}
        self._populate_stacked_widget(view_controller_factories)
        stacked_widget_layout = self._stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._stacked_widget)

        self._library_view_controller = ComboBoxParameterViewController(
            library_parameter, library_parameter.choices()
        )
        self._algorithm_view_controller = _AlgorithmComboViewController(algorithm_parameter)
        self._status_controller = ProcessingStatusController(
            product_repository, processing_api.task_monitor, status_view
        )
        self._product_view_controller = ProductParameterViewController(
            product_repository, product_table_model, self._status_controller
        )
        self._compute_view_controller = ComputeParameterViewController(
            globus_supported=globus.is_supported,
            genesis_supported=genesis.is_supported,
        )
        self._actions_view = ProcessingActionsView()

        layout = QFormLayout()
        layout.addRow('Library:', self._library_view_controller.get_widget())
        layout.addRow('Algorithm:', self._algorithm_view_controller.get_widget())
        layout.addRow('Product:', self._product_view_controller.get_widget())

        if globus.is_supported or genesis.is_supported:
            layout.addRow('Compute:', self._compute_view_controller.get_widget())

        layout.addRow('Action:', self._actions_view)
        layout.addRow(self._scroll_area)
        self._view.setLayout(layout)

        self._actions_view.reconstruct_button.clicked.connect(self._reconstruct)
        self._actions_view.train_button.clicked.connect(self._train)

        self._reconstruct_split_action = self._actions_view.actions_menu.addAction(
            'Reconstruct Split'
        )
        connect_triggered_signal(self._reconstruct_split_action, self._reconstruct_split)
        self._load_model_action = self._actions_view.actions_menu.addAction('Load Model...')
        connect_triggered_signal(self._load_model_action, self._load_model)
        self._export_training_data_action = self._actions_view.actions_menu.addAction(
            'Export Training Data...'
        )
        connect_triggered_signal(self._export_training_data_action, self._export_training_data)

        combo = self._product_view_controller.get_widget()
        combo.currentIndexChanged.connect(self._sync_action_buttons_to_selection)
        combo_model = combo.model()
        combo_model.dataChanged.connect(lambda *_: self._sync_action_buttons_to_selection())
        combo_model.rowsInserted.connect(lambda *_: self._sync_action_buttons_to_selection())
        combo_model.rowsRemoved.connect(lambda *_: self._sync_action_buttons_to_selection())
        self._sync_action_buttons_to_selection()

        self._sync_model_to_view()
        library_parameter.add_observer(self)
        algorithm_parameter.add_observer(self)

    def _sync_action_buttons_to_selection(self) -> None:
        index = self._product_view_controller.get_widget().currentIndex()
        can_act = False

        if index >= 0:
            try:
                item = self._product_repository[index]
            except IndexError:
                pass
            else:
                can_act = (
                    not item.is_pending()
                    and not item.is_failed()
                    and item.get_dataset() is not None
                )

        self._actions_view.reconstruct_button.setEnabled(can_act)
        self._actions_view.train_button.setEnabled(can_act)
        self._export_training_data_action.setEnabled(can_act)

    def _populate_stacked_widget(
        self, view_controller_factories: Iterable[ReconstructorViewControllerFactory]
    ) -> None:
        """Build one page per algorithm, remembered by the pair that selects it.

        By the pair rather than by position: the combo boxes no longer
        enumerate pages one-for-one, and a page found by counting would
        follow whichever list happened to be built the same way.
        """
        vcf_map = {vcf.name.casefold(): vcf for vcf in view_controller_factories}

        for library in self._processing_api.available_libraries():
            algorithms = list(self._processing_api.available_algorithms(library))

            if not algorithms:
                # The backend is registered but its packages are not
                # installed. It is still offered, so it needs somewhere to
                # land rather than leaving the previous backend's panel up.
                self._empty_page[library.casefold()] = self._stacked_widget.addWidget(
                    _centered_label(f'{library} is not installed.')
                )
                continue

            for algorithm in algorithms:
                try:
                    vcf = vcf_map[library.casefold()]
                except KeyError:
                    logger.warning(f'{library} not found!')
                    widget: QWidget = _centered_label(f'{library} not found!')
                else:
                    widget = vcf.create_view_controller(algorithm)

                self._page[library.casefold(), algorithm.casefold()] = (
                    self._stacked_widget.addWidget(widget)
                )

    def _has_dataset_or_warn(self, product_index: int, action: str) -> bool:
        try:
            item = self._product_repository[product_index]
        except IndexError:
            logger.warning(f'Cannot {action}: no product at index {product_index}.')
            return False

        if item.get_dataset() is None:
            logger.warning(
                f'Cannot {action}: product "{item.get_name()}" '
                'has no associated diffraction dataset.'
            )
            return False

        return True

    def _reconstruct(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        if not self._has_dataset_or_warn(input_product_index, 'reconstruct'):
            return

        if self._compute_view_controller.is_globus_button_checked():
            try:
                self._globus.executor.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Remote (Globus)', exc)
        elif self._compute_view_controller.is_genesis_button_checked():
            try:
                self._genesis.executor.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Remote (Genesis)', exc)
        else:  # local
            try:
                output_product_index = self._processing_api.reconstruct(
                    product_index=input_product_index
                )
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Local', exc)
            else:
                self._product_view_controller.get_widget().setCurrentIndex(output_product_index)

    def _train(self) -> None:
        product_index = self._product_view_controller.get_widget().currentIndex()

        if product_index < 0:
            return

        if not self._has_dataset_or_warn(product_index, 'train'):
            return

        if self._compute_view_controller.is_globus_button_checked():
            try:
                self._globus.executor.train(product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Remote (Globus)', exc)
        elif self._compute_view_controller.is_genesis_button_checked():
            try:
                self._genesis.executor.train(product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Remote (Genesis)', exc)
        else:  # local
            directories = self._train_dialog_controller.get_directories()

            if directories is None:
                return

            data_path, output_path = directories

            try:
                self._processing_api.train(
                    data_path,
                    output_path,
                    product_index=product_index,
                )
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Local', exc)
            else:
                self._sync_model_to_view()

    def _reconstruct_split(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        if not self._has_dataset_or_warn(input_product_index, 'reconstruct split'):
            return

        if (
            self._compute_view_controller.is_globus_button_checked()
            or self._compute_view_controller.is_genesis_button_checked()
        ):
            logger.warning('Cannot reconstruct split: only local compute splits a scan.')
            return

        try:
            output_product_index_odd, _ = self._processing_api.reconstruct_split(
                product_index=input_product_index
            )
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Reconstruct Split', exc)
        else:
            self._product_view_controller.get_widget().setCurrentIndex(output_product_index_odd)

    def _load_model(self) -> None:
        name_filter = self._processing_api.get_model_file_filter()
        file_path, _ = self._file_dialog_factory.get_open_file_path(
            self._view, 'Load Model', name_filters=[name_filter], selected_name_filter=name_filter
        )

        if not file_path:
            return

        try:
            self._processing_api.load_model(file_path)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Load Model', exc)
        else:
            self._sync_model_to_view()

    def _export_training_data(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        name_filters = self._processing_api.get_training_data_file_filters()

        if not name_filters:
            return

        default_file_type = self._processing_api.get_default_training_data_file_type()
        selected_name_filter = next(
            (name for name, file_type in name_filters.items() if file_type == default_file_type),
            next(iter(name_filters)),
        )
        file_path, selected_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Export Training Data',
            name_filters=list(name_filters),
            selected_name_filter=selected_name_filter,
        )

        if not file_path:
            return

        if not self._has_dataset_or_warn(input_product_index, 'export training data'):
            return

        try:
            self._processing_api.export_training_data(
                file_path,
                product_index=input_product_index,
                file_type=name_filters.get(selected_filter),
            )
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Export Training Data', exc)

    def _sync_model_to_view(self) -> None:
        # Asked of the model rather than read off a combo box, so that a
        # library change part-way through settling its algorithm selects
        # nothing rather than the wrong thing.
        library, algorithm = self._processing_api.current_selection()
        index = self._page.get((library.casefold(), algorithm.casefold()))

        if index is None:
            index = self._empty_page.get(library.casefold())

        if index is not None:
            self._stacked_widget.setCurrentIndex(index)

        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        is_trainable = isinstance(reconstructor, TrainableReconstructor)

        if isinstance(reconstructor, SubprocessReconstructor):
            is_trainable = reconstructor.is_trainable

        if is_trainable:
            assert isinstance(reconstructor, TrainableReconstructor)
            is_model_loaded = reconstructor.is_model_loaded()
            self._actions_view.reconstruct_button.setText('Infer')
            self._actions_view.reconstruct_button.setEnabled(is_model_loaded)
        else:
            self._actions_view.reconstruct_button.setText('Reconstruct')
            self._actions_view.reconstruct_button.setEnabled(True)

        self._actions_view.train_button.setVisible(is_trainable)
        self._load_model_action.setVisible(is_trainable)

    def _update(self, observable: Observable) -> None:
        if observable in (self._library_parameter, self._algorithm_parameter):
            self._sync_model_to_view()
