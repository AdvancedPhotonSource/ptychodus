from __future__ import annotations
from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer, SequenceObserver

from ..model.automation import (
    AutomationDataset,
    AutomationDatasetRepository,
    AutomationDatasetState,
    AutomationPresenter,
    AutomationSettings,
)
from .parametric import ParameterViewBuilder, ParameterViewController
from .data import FileDialogFactory


class AutomationDatasetListModel(QAbstractListModel):
    def __init__(
        self, repository: AutomationDatasetRepository, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._repository = repository

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            dataset = self._repository[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole:
                    return dataset.label
                case Qt.ItemDataRole.FontRole:
                    font = QFont()

                    match dataset.state:
                        case AutomationDatasetState.WAITING:
                            font.setItalic(True)
                        case AutomationDatasetState.RUNNING:
                            font.setBold(True)

                    return font

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._repository)


class AutomationDatasetListViewController(
    ParameterViewController, SequenceObserver[AutomationDataset]
):
    def __init__(self, repository: AutomationDatasetRepository) -> None:
        super().__init__()
        self._model = AutomationDatasetListModel(repository)
        self._widget = QListView()
        self._widget.setModel(self._model)

        repository.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def handle_item_inserted(self, index: int, item: AutomationDataset) -> None:
        self._model.beginInsertRows(QModelIndex(), index, index)
        self._model.endInsertRows()

    def handle_item_changed(self, index: int, item: AutomationDataset) -> None:
        top_left = self._model.index(index, 0)
        bottom_right = self._model.index(index, self._model.columnCount())
        self._model.dataChanged.emit(top_left, bottom_right)

    def handle_item_removed(self, index: int, item: AutomationDataset) -> None:
        self._model.beginRemoveRows(QModelIndex(), index, index)
        self._model.endRemoveRows()


class AutomationButtonsViewController(ParameterViewController, Observer):
    def __init__(
        self, repository: AutomationDatasetRepository, presenter: AutomationPresenter
    ) -> None:
        super().__init__()
        self._repository = repository
        self._presenter = presenter
        self._load_button = QPushButton('Load')
        self._watch_button = QPushButton('Watch')
        self._clear_button = QPushButton('Clear')
        self._widget = QWidget()

        self._load_button.clicked.connect(presenter.load_existing_datasets_to_repository)
        self._watch_button.setCheckable(True)
        self._watch_button.toggled.connect(presenter.set_watcher_enabled)
        self._clear_button.clicked.connect(repository.clear)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._load_button)
        layout.addWidget(self._watch_button)
        layout.addWidget(self._clear_button)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        presenter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._watch_button.setChecked(self._presenter.is_watcher_enabled())

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()


class AutomationController:
    def __init__(
        self,
        settings: AutomationSettings,
        repository: AutomationDatasetRepository,
        presenter: AutomationPresenter,
        view: QWidget,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._presenter = presenter
        self._view = view
        self._file_dialog_factory = file_dialog_factory

        view_builder = ParameterViewBuilder(file_dialog_factory)

        processing_group = 'Processing'
        view_builder.add_combo_box(
            settings.workflow, presenter.available_workflows(), 'Workflow:', group=processing_group
        )
        view_builder.add_directory_chooser(
            settings.data_directory, 'Data Directory:', group=processing_group
        )

        watchdog_group = 'Watchdog'
        view_builder.add_spin_box(settings.watchdog_delay_s, 'Delay [sec]:', group=watchdog_group)
        view_builder.add_check_box(
            settings.use_watchdog_polling_observer, 'Use Polling Observer', group=watchdog_group
        )

        list_view_controller = AutomationDatasetListViewController(repository)
        view_builder.add_view_controller_to_bottom(list_view_controller)

        buttons_view_controller = AutomationButtonsViewController(repository, presenter)
        view_builder.add_view_controller_to_bottom(buttons_view_controller)

        contents = view_builder.build_widget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(contents)
        view.setLayout(layout)
