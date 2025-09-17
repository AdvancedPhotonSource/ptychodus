from typing import Any

from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
)
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.diffraction import AssembledDiffractionDataset
from ...model.product import ProductRepositoryItem
from ...view.product import ProductEditorDialog
from ..helpers import create_brush_for_editable_cell


class ProductPropertyTableModel(QAbstractTableModel):
    def __init__(
        self,
        product_item: ProductRepositoryItem,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._product_item = product_item
        self._editable_item_brush = editable_item_brush
        self._header = ['Property', 'Value']
        self._properties = [
            'Probe Wavelength [nm]',
            'Probe Wavenumber [1/nm]',
            'Probe Angular Wavenumber [rad/nm]',
            'Probe Photon Flux [ph/s]',
            'Probe Power [W]',
            'Object Plane Pixel Width [nm]',
            'Object Plane Pixel Height [nm]',
            'Exposure Time [s]',
            'Mass Attenuation [m\u00b2/kg]',
            'Tomography Angle [deg]',
            'Fresnel Number',
            'Detector Numerical Aperture',
            'Depth of Field [nm]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.row() in (7, 8, 9):
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case 0:
                        return self._properties[index.row()]
                    case 1:
                        metadata_item = self._product_item.get_metadata_item()
                        geometry = self._product_item.get_geometry()

                        match index.row():
                            case 0:
                                return f'{geometry.probe_wavelength_m * 1e9:.4g}'
                            case 1:
                                return f'{geometry.probe_wavelengths_per_m * 1e-9:.4g}'
                            case 2:
                                return f'{geometry.probe_radians_per_m * 1e-9:.4g}'
                            case 3:
                                return f'{geometry.probe_photons_per_s:.4g}'
                            case 4:
                                return f'{geometry.probe_power_W:.4g}'
                            case 5:
                                return f'{geometry.object_plane_pixel_width_m * 1e9:.4g}'
                            case 6:
                                return f'{geometry.object_plane_pixel_height_m * 1e9:.4g}'
                            case 7:
                                return f'{metadata_item.exposure_time_s.get_value():.4g}'
                            case 8:
                                return f'{metadata_item.mass_attenuation_m2_kg.get_value():.4g}'
                            case 9:
                                return f'{metadata_item.tomography_angle_deg.get_value():.4g}'
                            case 10:
                                try:
                                    return f'{geometry.fresnel_number:.4g}'
                                except ZeroDivisionError:
                                    return 'inf'
                            case 11:
                                try:
                                    return f'{geometry.detector_numerical_aperture:.4g}'
                                except ZeroDivisionError:
                                    return 'inf'
                            case 12:
                                try:
                                    return f'{geometry.depth_of_field_m * 1e9:.4g}'
                                except ZeroDivisionError:
                                    return 'inf'
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            metadata_item = self._product_item.get_metadata_item()

            match index.row():
                case 7:
                    try:
                        exposure_time_s = float(value)
                    except ValueError:
                        return False

                    metadata_item.exposure_time_s.set_value(exposure_time_s)
                    return True
                case 8:
                    try:
                        mass_attenuation_m2_kg = float(value)
                    except ValueError:
                        return False

                    metadata_item.mass_attenuation_m2_kg.set_value(mass_attenuation_m2_kg)
                    return True
                case 9:
                    try:
                        tomography_angle_deg = float(value)
                    except ValueError:
                        return False

                    metadata_item.tomography_angle_deg.set_value(tomography_angle_deg)
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


class ProductEditorViewController(Observer):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        product: ProductRepositoryItem,
        table_model: ProductPropertyTableModel,
        dialog: ProductEditorDialog,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._product = product
        self._table_model = table_model
        self._dialog = dialog

    @classmethod
    def edit_product(
        cls, dataset: AssembledDiffractionDataset, product: ProductRepositoryItem, parent: QWidget
    ) -> None:
        dialog = ProductEditorDialog(parent)
        dialog.setWindowTitle(f'Edit Product: {product.get_name()}')

        editable_item_brush = create_brush_for_editable_cell(dialog.table_view)
        table_model = ProductPropertyTableModel(product, editable_item_brush)

        table_proxy_model = QSortFilterProxyModel()
        table_proxy_model.setSourceModel(table_model)

        dialog.table_view.setModel(table_proxy_model)
        dialog.table_view.setSortingEnabled(True)
        vertical_header = dialog.table_view.verticalHeader()

        if vertical_header is not None:
            vertical_header.hide()

        dialog.table_view.resizeColumnsToContents()
        dialog.table_view.resizeRowsToContents()

        view_controller = cls(dataset, product, table_model, dialog)
        product.add_observer(view_controller)
        dialog.text_edit.textChanged.connect(view_controller._sync_view_to_model)

        view_controller._sync_model_to_view()

        dialog.actions_view.estimate_probe_photon_count_button.clicked.connect(
            view_controller._estimate_probe_photon_count
        )
        dialog.finished.connect(view_controller._finish)
        dialog.open()
        dialog.adjustSize()

    def _sync_view_to_model(self) -> None:
        metadata = self._product.get_metadata_item()
        metadata.comments.set_value(self._dialog.text_edit.toPlainText())

    def _sync_model_to_view(self) -> None:
        self._table_model.beginResetModel()
        self._table_model.endResetModel()

        metadata = self._product.get_metadata_item()
        self._dialog.text_edit.setPlainText(metadata.comments.get_value())

    def _estimate_probe_photon_count(self) -> None:
        metadata = self._product.get_metadata_item()
        metadata.probe_photon_count.set_value(self._dataset.get_maximum_pattern_counts())

        self._table_model.beginResetModel()
        self._table_model.endResetModel()

    def _finish(self, result: int) -> None:
        self._product.remove_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._product:
            self._sync_model_to_view()
