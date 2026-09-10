from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PyQt5.QtWidgets import QHeaderView, QWizardPage

from ptychodus.api.constants import LengthUnit
from ptychodus.api.diffraction import BeamCenter, CropRegion, DiffractionMetadata
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.parameters import RealParameter

from ....model.diffraction import DetectorSettings, DiffractionSettings
from ....model.product import ProductSettings
from ....view.diffraction import OpenDatasetWizardMetadataPage
from ...parameters import LengthParameterViewController


@dataclass(frozen=True)
class _MetadataRow:
    name: str
    is_present: Callable[[DiffractionMetadata], bool]
    format_value: Callable[[DiffractionMetadata], str]
    apply: Callable[[DiffractionMetadata], None]


class MetadataTableModel(QAbstractTableModel):
    _HEADER = ('Metadata', 'Sync', 'Value')

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rows: list[_MetadataRow] = []
        self._metadata = DiffractionMetadata.create_null()
        self._checked_rows: set[int] = set()

    def set_rows(self, rows: Sequence[_MetadataRow], metadata: DiffractionMetadata) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self._metadata = metadata
        self._checked_rows = set(range(len(self._rows)))
        self.endResetModel()

    def checked_rows(self) -> list[_MetadataRow]:
        return [self._rows[i] for i in sorted(self._checked_rows)]

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._HEADER[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = self._rows[index.row()]
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return row.name
            elif column == 2:
                return row.format_value(self._metadata)
        elif role == Qt.ItemDataRole.CheckStateRole and column == 1:
            return (
                Qt.CheckState.Checked
                if index.row() in self._checked_rows
                else Qt.CheckState.Unchecked
            )

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() == 1:
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(  # noqa: N802
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if index.isValid() and index.column() == 1 and role == Qt.ItemDataRole.CheckStateRole:
            if value == Qt.CheckState.Checked:
                self._checked_rows.add(index.row())
            else:
                self._checked_rows.discard(index.row())
            self.dataChanged.emit(index, index)
            return True
        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._HEADER)


def _format_detector_distance(metadata: DiffractionMetadata) -> str:
    distance_m = metadata.detector_distance_m
    return '—' if distance_m is None else f'{distance_m:.3f} m'


def _format_beam_center(metadata: DiffractionMetadata) -> str:
    beam_center = metadata.beam_center
    if beam_center is not None:
        return f'({beam_center.x_px}, {beam_center.y_px}) px'
    extent = metadata.detector_extent
    if extent is not None:
        return f'({int(extent.width_px) // 2}, {int(extent.height_px) // 2}) px'
    return '—'


def _format_crop_extent(metadata: DiffractionMetadata) -> str:
    extent = metadata.detector_extent
    if extent is None:
        return '—'
    return f'{int(extent.width_px)} × {int(extent.height_px)} px'


def _format_probe_energy(metadata: DiffractionMetadata) -> str:
    energy_eV = metadata.probe_energy_eV  # noqa: N806
    return '—' if energy_eV is None else f'{energy_eV:.3f} eV'


def _format_probe_photon_count(metadata: DiffractionMetadata) -> str:
    count = metadata.probe_photon_count
    return '—' if count is None else f'{count:g}'


def _format_exposure_time(metadata: DiffractionMetadata) -> str:
    exposure_time_s = metadata.exposure_time_s
    return '—' if exposure_time_s is None else f'{exposure_time_s * 1e3:.3f} ms'


class OpenDatasetWizardMetadataViewController:
    def __init__(
        self,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
        product_settings: ProductSettings,
        get_metadata: Callable[[], DiffractionMetadata],
    ) -> None:
        self._detector_settings = detector_settings
        self._diffraction_settings = diffraction_settings
        self._product_settings = product_settings
        self._get_metadata = get_metadata
        self._page = OpenDatasetWizardMetadataPage()
        self._table_model = MetadataTableModel()
        self._all_rows: tuple[_MetadataRow, ...] = self._build_rows()

        # Scratch parameters isolate wizard edits from DetectorSettings until the user
        # accepts (settings are written in import_metadata). Bound only to the two
        # LengthParameterViewController widgets; nothing else observes them.
        self._pixel_width_param = RealParameter(
            detector_settings.pixel_width_m.get_value(), parent=None, minimum=0.0
        )
        self._pixel_height_param = RealParameter(
            detector_settings.pixel_height_m.get_value(), parent=None, minimum=0.0
        )
        self._pixel_width_view_controller = LengthParameterViewController(
            self._pixel_width_param, default_unit=LengthUnit.MICROMETER
        )
        self._pixel_height_view_controller = LengthParameterViewController(
            self._pixel_height_param, default_unit=LengthUnit.MICROMETER
        )
        self._page.pixel_size_form.addRow('Width:', self._pixel_width_view_controller.get_widget())
        self._page.pixel_size_form.addRow(
            'Height:', self._pixel_height_view_controller.get_widget()
        )

        self._page.table_view.setModel(self._table_model)

        horizontal_header = self._page.table_view.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            horizontal_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            horizontal_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        vertical_header = self._page.table_view.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)

        self.refresh()

    def _build_rows(self) -> tuple[_MetadataRow, ...]:
        return (
            _MetadataRow(
                name='Detector Distance',
                is_present=lambda m: m.detector_distance_m is not None,
                format_value=_format_detector_distance,
                apply=self._apply_detector_distance,
            ),
            _MetadataRow(
                name='Beam Center',
                is_present=lambda m: m.beam_center is not None or m.detector_extent is not None,
                format_value=_format_beam_center,
                apply=self._apply_beam_center,
            ),
            _MetadataRow(
                name='Pattern Crop Extent',
                is_present=lambda m: m.detector_extent is not None,
                format_value=_format_crop_extent,
                apply=self._apply_crop_extent,
            ),
            _MetadataRow(
                name='Probe Energy',
                is_present=lambda m: m.probe_energy_eV is not None,
                format_value=_format_probe_energy,
                apply=self._apply_probe_energy,
            ),
            _MetadataRow(
                name='Probe Photon Count',
                is_present=lambda m: m.probe_photon_count is not None,
                format_value=_format_probe_photon_count,
                apply=self._apply_probe_photon_count,
            ),
            _MetadataRow(
                name='Exposure Time',
                is_present=lambda m: m.exposure_time_s is not None,
                format_value=_format_exposure_time,
                apply=self._apply_exposure_time,
            ),
        )

    def import_metadata(self) -> None:
        metadata = self._get_metadata()

        for row in self._table_model.checked_rows():
            row.apply(metadata)

        if self._page.sync_pixel_size_check_box.isChecked():
            self._detector_settings.pixel_width_m.set_value(self._pixel_width_param.get_value())
            self._detector_settings.pixel_height_m.set_value(self._pixel_height_param.get_value())

    def refresh(self) -> None:
        metadata = self._get_metadata()
        visible = [row for row in self._all_rows if row.is_present(metadata)]
        self._table_model.set_rows(visible, metadata)

        # Seed the pixel-size widgets: metadata wins when the reader supplied a value,
        # otherwise fall back to current DetectorSettings. The sync checkbox reflects
        # the same choice so an untouched wizard behaves like the old table row (write
        # metadata into settings on accept; leave settings alone when metadata is
        # absent).
        pixel_geometry = metadata.detector_pixel_geometry
        if pixel_geometry is not None:
            self._pixel_width_param.set_value(pixel_geometry.width_m)
            self._pixel_height_param.set_value(pixel_geometry.height_m)
            self._page.sync_pixel_size_check_box.setChecked(True)
        else:
            self._pixel_width_param.set_value(self._detector_settings.pixel_width_m.get_value())
            self._pixel_height_param.set_value(self._detector_settings.pixel_height_m.get_value())
            self._page.sync_pixel_size_check_box.setChecked(False)

    def get_widget(self) -> QWizardPage:
        return self._page

    # --- Detector distance ---

    def _apply_detector_distance(self, metadata: DiffractionMetadata) -> None:
        distance_m = metadata.detector_distance_m
        if distance_m:
            self._product_settings.detector_distance_m.set_value(distance_m)

    # --- Beam center ---

    def _apply_beam_center(self, metadata: DiffractionMetadata) -> None:
        beam_center = metadata.beam_center
        if beam_center is not None:
            self._diffraction_settings.beam_center_x_px.set_value(beam_center.x_px)
            self._diffraction_settings.beam_center_y_px.set_value(beam_center.y_px)
        elif metadata.detector_extent is not None:
            self._diffraction_settings.beam_center_x_px.set_value(
                int(metadata.detector_extent.width_px) // 2
            )
            self._diffraction_settings.beam_center_y_px.set_value(
                int(metadata.detector_extent.height_px) // 2
            )

    # --- Crop extent ---

    def _apply_crop_extent(self, metadata: DiffractionMetadata) -> None:
        extent = metadata.detector_extent
        if extent is None:
            return

        center = BeamCenter(
            x_px=self._diffraction_settings.beam_center_x_px.get_value(),
            y_px=self._diffraction_settings.beam_center_y_px.get_value(),
        )
        detector_extent = ImageExtent(
            width_px=int(extent.width_px),
            height_px=int(extent.height_px),
        )
        region = CropRegion.from_largest_pow2(center, detector_extent)
        self._diffraction_settings.crop_width_px.set_value(region.width_px)
        self._diffraction_settings.crop_height_px.set_value(region.height_px)

    # --- Probe energy ---

    def _apply_probe_energy(self, metadata: DiffractionMetadata) -> None:
        energy_eV = metadata.probe_energy_eV  # noqa: N806
        if energy_eV:
            self._product_settings.probe_energy_eV.set_value(energy_eV)

    # --- Probe photon count ---

    def _apply_probe_photon_count(self, metadata: DiffractionMetadata) -> None:
        count = metadata.probe_photon_count
        if count:
            self._product_settings.probe_photon_count.set_value(count)

    # --- Exposure time ---

    def _apply_exposure_time(self, metadata: DiffractionMetadata) -> None:
        exposure_time_s = metadata.exposure_time_s
        if exposure_time_s:
            self._product_settings.exposure_time_s.set_value(exposure_time_s)
