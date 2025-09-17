from collections.abc import Sequence
from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileSystemModel,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QTableView,
    QWidget,
    QWizardPage,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import PathParameter

from ....model.diffraction import DiffractionAPI, DiffractionSettings
from ....view.diffraction import OpenDatasetWizardPage
from ....view.widgets import ExceptionDialog
from ...data import FileDialogFactory
from ...helpers import connect_current_changed_signal

logger = logging.getLogger(__name__)


class OpenDatasetWizardBreadcrumbsViewController(Observer):
    def __init__(self, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._file_dialog_factory = file_dialog_factory
        self._widget = QWidget()
        self._path_list: list[Path] = []
        self._button_group = QButtonGroup()
        self._button_group.idClicked.connect(self._handle_id_clicked)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        file_dialog_factory.add_observer(self)

    def _handle_id_clicked(self, button_id: int) -> None:
        path = self._path_list[button_id]
        self._file_dialog_factory.set_open_working_directory(path)

    def _sync_model_to_view(self) -> None:
        path = self._file_dialog_factory.get_open_working_directory().resolve()

        for button_id, existing_path in enumerate(self._path_list):
            if path == existing_path:
                button = self._button_group.button(button_id)
                button.setChecked(True)
                return

        layout = self._widget.layout()

        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()

                if widget is None:
                    continue
                elif isinstance(widget, QPushButton):
                    self._button_group.removeButton(widget)
                    widget.deleteLater()

        self._path_list.clear()
        button_list: list[QPushButton] = []

        while True:
            if path.name:
                button = QPushButton(path.name)
                button.setCheckable(True)
                button_list.append(button)
                self._path_list.append(path)
                path = path.parent
            else:
                button = QPushButton(path.anchor)
                button.setCheckable(True)
                button_list.append(button)
                self._path_list.append(Path(path.anchor))
                break

        for button_id, button in reversed(list(enumerate(button_list))):
            self._button_group.addButton(button, button_id)
            layout.addWidget(button)

        if isinstance(layout, QHBoxLayout):
            layout.addStretch()

        button_list[0].setChecked(True)
        self._widget.setLayout(layout)
        self._widget.update()

    def get_widget(self) -> QWidget:
        return self._widget

    def _update(self, observable: Observable) -> None:
        if observable is self._file_dialog_factory:
            self._sync_model_to_view()


class OpenDatasetWizardLocationViewController(Observer):
    def __init__(self, file_path: PathParameter, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._file_path = file_path
        self._file_dialog_factory = file_dialog_factory

        self._widget = QLineEdit()
        self._widget.editingFinished.connect(self._handle_editing_finished)

        self._sync_model_to_view()
        file_path.add_observer(self)
        file_dialog_factory.add_observer(self)

    def _handle_editing_finished(self) -> None:
        text = self._widget.text()
        path = Path(text)

        if not path.is_absolute():
            path = self._file_dialog_factory.get_open_working_directory() / text

        path = path.resolve()

        self._file_dialog_factory.set_open_working_directory(path)
        self._file_path.set_value(path)

    def _sync_model_to_view(self) -> None:
        file_path = self._file_path.get_value()

        if file_path.is_file():
            self._widget.setText(file_path.name)
        else:
            self._widget.clear()

    def get_widget(self) -> QWidget:
        return self._widget

    def _update(self, observable: Observable) -> None:
        if observable in (self._file_path, self._file_dialog_factory):
            self._sync_model_to_view()


class OpenDatasetWizardFilePathViewController(Observer):
    def __init__(self, file_path: PathParameter, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._file_path = file_path
        self._file_dialog_factory = file_dialog_factory

        self._model = QFileSystemModel()
        self._model.setNameFilterDisables(False)

        self._proxy_model = QSortFilterProxyModel()
        self._proxy_model.setSourceModel(self._model)

        self._widget = QTableView()
        self._widget.setModel(self._proxy_model)
        self._widget.setSortingEnabled(True)
        self._widget.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._widget.verticalHeader().hide()
        self._widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._widget.doubleClicked.connect(self._handle_table_double_clicked)

        connect_current_changed_signal(self._widget, self._handle_current_changed)

        self._sync_model_to_view()
        file_path.add_observer(self)
        file_dialog_factory.add_observer(self)

    def set_name_filters(self, name_filters: Sequence[str]) -> None:
        logger.debug(f'Name Filters: {name_filters}')
        self._model.setNameFilters(name_filters)

    def _handle_table_double_clicked(self, proxy_index: QModelIndex) -> None:
        index = self._proxy_model.mapToSource(proxy_index)
        file_info = self._model.fileInfo(index)

        if file_info.isDir():
            directory = Path(file_info.canonicalFilePath())
            self._file_dialog_factory.set_open_working_directory(directory)

    def _handle_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._proxy_model.mapToSource(current)
        path = Path(self._model.filePath(index))
        self._file_path.set_value(path)

    def _sync_model_to_view(self) -> None:
        file_path = self._file_path.get_value()
        root_path = self._file_dialog_factory.get_open_working_directory()

        index = self._model.setRootPath(str(root_path))
        proxy_index = self._proxy_model.mapFromSource(index)
        self._widget.setRootIndex(proxy_index)

        if file_path.is_relative_to(root_path):
            index = self._model.index(str(file_path))
            proxy_index = self._proxy_model.mapFromSource(index)
            self._widget.setCurrentIndex(proxy_index)

    def get_widget(self) -> QWidget:
        return self._widget

    def _update(self, observable: Observable) -> None:
        if observable is self._file_path:
            self._sync_model_to_view()
        elif observable is self._file_dialog_factory:
            self._sync_model_to_view()


class OpenDatasetWizardFileTypeViewController(Observable, Observer):
    def __init__(self, api: DiffractionAPI) -> None:
        super().__init__()
        self._file_reader_chooser = api.get_file_reader_chooser()
        self._file_reader_chooser.add_observer(self)
        self._combo_box = QComboBox()

        for plugin in self._file_reader_chooser:
            self._combo_box.addItem(plugin.display_name)

        self._sync_model_to_view()
        self._combo_box.textActivated.connect(self._handle_text_activated)

    def get_name_filters(self) -> Sequence[str]:
        text = self._combo_box.currentText()
        z = re.search(r'\((.+)\)', text)
        return z.group(1).split() if z else []

    def _handle_text_activated(self, text: str) -> None:
        self._file_reader_chooser.set_current_plugin(text)

    def _sync_model_to_view(self) -> None:
        self._combo_box.setCurrentText(self._file_reader_chooser.get_current_plugin().display_name)

    def get_widget(self) -> QWidget:
        return self._combo_box

    def _update(self, observable: Observable) -> None:
        if observable is self._file_reader_chooser:
            self._sync_model_to_view()
            self.notify_observers()


class OpenDatasetWizardFilesViewController(Observer):
    def __init__(
        self,
        settings: DiffractionSettings,
        api: DiffractionAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._api = api
        self._file_dialog_factory = file_dialog_factory

        self._breadcrumbs_view_controller = OpenDatasetWizardBreadcrumbsViewController(
            file_dialog_factory
        )
        self._location_view_controller = OpenDatasetWizardLocationViewController(
            settings.file_path, file_dialog_factory
        )
        self._file_path_view_controller = OpenDatasetWizardFilePathViewController(
            settings.file_path, file_dialog_factory
        )
        self._file_type_view_controller = OpenDatasetWizardFileTypeViewController(api)
        self._file_type_view_controller.add_observer(self)

        layout = QFormLayout()
        layout.addRow(self._breadcrumbs_view_controller.get_widget())
        layout.addRow('Location:', self._location_view_controller.get_widget())
        layout.addRow(self._file_path_view_controller.get_widget())
        layout.addRow('File Type:', self._file_type_view_controller.get_widget())

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Choose Dataset File(s)')
        self._page.setLayout(layout)

        self._sync_model_to_view()
        settings.file_path.add_observer(self)

    def open_dataset(self) -> None:
        file_reader_chooser = self._api.get_file_reader_chooser()
        file_type = file_reader_chooser.get_current_plugin().simple_name
        file_path = self._settings.file_path.get_value()

        try:
            self._api.open_patterns(file_path, file_type=file_type)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Open Dataset', err)

    def get_widget(self) -> QWizardPage:
        return self._page

    def _check_if_complete(self) -> None:
        file_path = self._settings.file_path.get_value()
        self._page._set_complete(file_path.is_file())

    def restart(self) -> None:
        self._check_if_complete()

    def _handle_file_type_changed(self) -> None:
        name_filters = self._file_type_view_controller.get_name_filters()
        self._file_path_view_controller.set_name_filters(name_filters)

    def _sync_model_to_view(self) -> None:
        file_path = self._settings.file_path.get_value()

        if file_path.exists():
            self._file_dialog_factory.set_open_working_directory(file_path)

        self._handle_file_type_changed()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings.file_path:
            self._check_if_complete()
        elif observable is self._file_type_view_controller:
            self._handle_file_type_changed()
