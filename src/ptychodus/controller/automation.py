from __future__ import annotations
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QTimer
from PyQt5.QtGui import QFont

from ptychodus.api.observer import Observable, Observer

from ..model.automation import (
    AutomationCore,
    AutomationDatasetState,
    AutomationPresenter,
    AutomationProcessingPresenter,
)
from ..view.automation import (
    AutomationView,
    AutomationProcessingView,
    AutomationWatchdogView,
)
from .data import FileDialogFactory


class AutomationProcessingController(Observer):
    def __init__(
        self,
        presenter: AutomationPresenter,
        view: AutomationProcessingView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._file_dialog_factory = file_dialog_factory

        presenter.add_observer(self)

        for strategy in presenter.get_strategies():
            view.strategy_combo_box.addItem(strategy)

        view.strategy_combo_box.textActivated.connect(presenter.set_strategy)
        view.directory_line_edit.editingFinished.connect(self._sync_directory_to_model)
        view.directory_browse_button.clicked.connect(self._browse_directory)
        view.interval_spin_box.valueChanged.connect(presenter.set_processing_interval_s)

        self._sync_model_to_view()

    def _sync_directory_to_model(self) -> None:
        data_dir = Path(self._view.directory_line_edit.text())
        self._presenter.set_data_directory(data_dir)

    def _browse_directory(self) -> None:
        dir_path = self._file_dialog_factory.get_existing_directory_path(
            self._view, 'Choose Data Directory'
        )

        if dir_path:
            self._presenter.set_data_directory(dir_path)

    def _sync_model_to_view(self) -> None:
        self._view.strategy_combo_box.blockSignals(True)
        self._view.strategy_combo_box.setCurrentText(self._presenter.get_strategy())
        self._view.strategy_combo_box.blockSignals(False)

        data_dir = self._presenter.get_data_directory()

        if data_dir:
            self._view.directory_line_edit.setText(str(data_dir))
        else:
            self._view.directory_line_edit.clear()

        interval_limits_s = self._presenter.get_processing_interval_limits_s()

        self._view.interval_spin_box.blockSignals(True)
        self._view.interval_spin_box.setRange(interval_limits_s.lower, interval_limits_s.upper)
        self._view.interval_spin_box.setValue(self._presenter.get_processing_interval_s())
        self._view.interval_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()


class AutomationWatchdogController(Observer):
    def __init__(self, presenter: AutomationPresenter, view: AutomationWatchdogView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        presenter.add_observer(self)

        view.delay_spin_box.valueChanged.connect(presenter.set_watchdog_delay_s)
        view.use_polling_observer_check_box.toggled.connect(
            presenter.set_watchdog_polling_observer_enabled
        )

        self._sync_model_to_view()

    def _sync_model_to_view(self) -> None:
        delay_limits_s = self._presenter.get_watchdog_delay_limits_s()

        self._view.delay_spin_box.blockSignals(True)
        self._view.delay_spin_box.setRange(delay_limits_s.lower, delay_limits_s.upper)
        self._view.delay_spin_box.setValue(self._presenter.get_watchdog_delay_s())
        self._view.delay_spin_box.blockSignals(False)

        self._view.use_polling_observer_check_box.setChecked(
            self._presenter.is_watchdog_polling_observer_enabled()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()


class AutomationProcessingListModel(QAbstractListModel):
    def __init__(
        self, presenter: AutomationProcessingPresenter, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return self._presenter.get_dataset_label(index.row())
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                state = self._presenter.get_dataset_state(index.row())

                if state == AutomationDatasetState.WAITING:
                    font.setItalic(True)
                elif state == AutomationDatasetState.PROCESSING:
                    font.setBold(True)

                return font

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return self._presenter.get_num_datasets()


class AutomationController(Observer):
    def __init__(
        self,
        core: AutomationCore,
        presenter: AutomationPresenter,
        processing_presenter: AutomationProcessingPresenter,
        view: AutomationView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._core = core
        self._presenter = presenter
        self._processing_controller = AutomationProcessingController(
            presenter, view.processing_view, file_dialog_factory
        )
        self._watchdog_controller = AutomationWatchdogController(presenter, view.watchdog_view)
        self._processing_presenter = processing_presenter
        self._list_model = AutomationProcessingListModel(processing_presenter)
        self._view = view
        self._execute_waiting_tasks_timer = QTimer()
        self._automation_timer = QTimer()

    @classmethod
    def create_instance(
        cls,
        core: AutomationCore,
        presenter: AutomationPresenter,
        processing_presenter: AutomationProcessingPresenter,
        view: AutomationView,
        file_dialog_factory: FileDialogFactory,
    ) -> AutomationController:
        controller = cls(core, presenter, processing_presenter, view, file_dialog_factory)
        processing_presenter.add_observer(controller)

        view.processing_list_view.setModel(controller._list_model)

        view.load_button.clicked.connect(presenter.load_existing_datasets_to_repository)
        view.watch_button.setCheckable(True)
        view.watch_button.toggled.connect(presenter.set_watchdog_enabled)
        view.process_button.setCheckable(True)
        view.process_button.toggled.connect(processing_presenter.set_processing_enabled)
        view.clear_button.clicked.connect(presenter.clear_dataset_repository)

        controller._sync_model_to_view()

        controller._execute_waiting_tasks_timer.timeout.connect(core.execute_waiting_tasks)
        controller._execute_waiting_tasks_timer.start(60 * 1000)  # TODO customize (in milliseconds)

        controller._automation_timer.timeout.connect(core.refresh_dataset_repository)
        controller._automation_timer.start(10 * 1000)  # TODO customize (in milliseconds)

        return controller

    def _sync_model_to_view(self) -> None:
        self._view.process_button.setChecked(self._processing_presenter.is_processing_enabled())
        self._list_model.beginResetModel()
        self._list_model.endResetModel()

        self._view.watch_button.setChecked(self._presenter.is_watchdog_enabled())
        self._view.process_button.setChecked(self._processing_presenter.is_processing_enabled())

    def _update(self, observable: Observable) -> None:
        if observable is self._processing_presenter:
            self._sync_model_to_view()
