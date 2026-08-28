from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable

from ....model.diffraction import DetectorSettings, DiffractionAPI, DiffractionSettings
from ....view.diffraction import OpenDatasetWizardPage

from ...data import FileDialogFactory
from ..detector_extent import DetectorExtentSource
from ...parameters import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)


def _crop_size_limits(det_size_px: int) -> Interval[int]:
    return Interval[int](1, det_size_px)


def _crop_center_limits(det_size_px: int) -> Interval[int]:
    return Interval[int](1, det_size_px)


def _effective_crop_size(det_size_px: int, requested: int, *, crop_enabled: bool) -> int:
    """The crop dimension actually used by the pipeline: clamped to detector when cropping is on;
    otherwise the full detector width/height."""
    return _crop_size_limits(det_size_px).clamp(requested) if crop_enabled else det_size_px


def _bin_size_limits(effective_crop_px: int) -> Interval[int]:
    return Interval[int](1, effective_crop_px)


def _set_spin_box(box: QSpinBox, limits: Interval[int], value: int) -> None:
    box.blockSignals(True)
    box.setRange(limits.lower, limits.upper)
    box.setValue(value)
    box.blockSignals(False)


class StorageViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self, settings: DiffractionSettings, file_dialog_factory: FileDialogFactory
    ) -> None:
        super().__init__(settings.memmap_enabled, 'Memory Map Diffraction Data')
        self._view_controller = PathParameterViewController.create_directory_chooser(
            settings.scratch_directory, file_dialog_factory
        )

        layout = QFormLayout()
        layout.addRow('Scratch Directory:', self._view_controller.get_widget())
        self.get_widget().setLayout(layout)


class BadPixelsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        detector_settings: DetectorSettings,
        api: DiffractionAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__(detector_settings.bad_pixels_enabled, 'Apply Bad Pixels Mask')
        self._file_reader_parameter = api.get_bad_pixels_file_reader_parameter()
        self._file_path_view_controller = PathParameterViewController.create_file_opener(
            detector_settings.bad_pixels_file_path,
            file_dialog_factory,
            caption='Open Bad Pixels File',
            name_filters=list(self._file_reader_parameter.choices()),
            selected_name_filter=self._file_reader_parameter.get_value(),
            on_filter_selected=self._file_reader_parameter.set_value,
        )

        layout = QFormLayout()
        layout.addRow('File:', self._file_path_view_controller.get_widget())
        self.get_widget().setLayout(layout)

        self._file_reader_parameter.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._file_reader_parameter:
            self._file_path_view_controller.set_selected_name_filter(
                self._file_reader_parameter.get_value()
            )
        else:
            super()._update(observable)


class CropViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        extent_source: DetectorExtentSource,
    ) -> None:
        super().__init__(diffraction_settings.crop_enabled, 'Crop')
        self._diffraction_settings = diffraction_settings
        self._extent_source = extent_source

        self._center_x_spin_box = QSpinBox()
        self._center_y_spin_box = QSpinBox()
        self._width_spin_box = QSpinBox()
        self._height_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Center:'), 0, 0)
        layout.addWidget(self._center_x_spin_box, 0, 1)
        layout.addWidget(self._center_y_spin_box, 0, 2)
        layout.addWidget(QLabel('Extent:'), 1, 0)
        layout.addWidget(self._width_spin_box, 1, 1)
        layout.addWidget(self._height_spin_box, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)

        self._observed = (
            diffraction_settings.crop_center_x_px,
            diffraction_settings.crop_center_y_px,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_height_px,
        )

        self._sync_model_to_view()

        self._center_x_spin_box.valueChanged.connect(
            diffraction_settings.crop_center_x_px.set_value
        )
        self._center_y_spin_box.valueChanged.connect(
            diffraction_settings.crop_center_y_px.set_value
        )
        self._width_spin_box.valueChanged.connect(diffraction_settings.crop_width_px.set_value)
        self._height_spin_box.valueChanged.connect(diffraction_settings.crop_height_px.set_value)

        for parameter in self._observed:
            parameter.add_observer(self)
        extent_source.add_observer(self)

    def _sync_model_to_view(self) -> None:
        extent = self._extent_source.get_extent()
        spin_boxes = (
            self._center_x_spin_box,
            self._center_y_spin_box,
            self._width_spin_box,
            self._height_spin_box,
        )

        if extent is None:
            for box in spin_boxes:
                box.setEnabled(False)
            return

        for box in spin_boxes:
            box.setEnabled(True)

        det_w = extent.width_px
        det_h = extent.height_px

        _set_spin_box(
            self._center_x_spin_box,
            _crop_center_limits(det_w),
            _crop_center_limits(det_w).clamp(
                self._diffraction_settings.crop_center_x_px.get_value()
            ),
        )
        _set_spin_box(
            self._center_y_spin_box,
            _crop_center_limits(det_h),
            _crop_center_limits(det_h).clamp(
                self._diffraction_settings.crop_center_y_px.get_value()
            ),
        )
        _set_spin_box(
            self._width_spin_box,
            _crop_size_limits(det_w),
            _crop_size_limits(det_w).clamp(self._diffraction_settings.crop_width_px.get_value()),
        )
        _set_spin_box(
            self._height_spin_box,
            _crop_size_limits(det_h),
            _crop_size_limits(det_h).clamp(self._diffraction_settings.crop_height_px.get_value()),
        )

    def _update(self, observable: Observable) -> None:
        if observable in self._observed or observable is self._extent_source:
            self._sync_model_to_view()
        else:
            super()._update(observable)


class BinningViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        extent_source: DetectorExtentSource,
    ) -> None:
        super().__init__(diffraction_settings.binning_enabled, 'Bin Pixels')
        self._diffraction_settings = diffraction_settings
        self._extent_source = extent_source

        self._bin_size_x_spin_box = QSpinBox()
        self._bin_size_y_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Bin Size:'), 0, 0)
        layout.addWidget(self._bin_size_x_spin_box, 0, 1)
        layout.addWidget(self._bin_size_y_spin_box, 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)

        # Also observe crop_enabled + crop extents because the effective bin-size upper bound
        # depends on the crop settings.
        self._observed = (
            diffraction_settings.bin_size_x,
            diffraction_settings.bin_size_y,
            diffraction_settings.crop_enabled,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_height_px,
        )

        self._sync_model_to_view()

        self._bin_size_x_spin_box.valueChanged.connect(diffraction_settings.bin_size_x.set_value)
        self._bin_size_y_spin_box.valueChanged.connect(diffraction_settings.bin_size_y.set_value)

        for parameter in self._observed:
            parameter.add_observer(self)
        extent_source.add_observer(self)

    def _sync_model_to_view(self) -> None:
        extent = self._extent_source.get_extent()
        if extent is None:
            self._bin_size_x_spin_box.setEnabled(False)
            self._bin_size_y_spin_box.setEnabled(False)
            return

        self._bin_size_x_spin_box.setEnabled(True)
        self._bin_size_y_spin_box.setEnabled(True)

        det_w = extent.width_px
        det_h = extent.height_px
        crop_enabled = self._diffraction_settings.crop_enabled.get_value()
        binning_enabled = self._diffraction_settings.binning_enabled.get_value()

        effective_w = _effective_crop_size(
            det_w,
            self._diffraction_settings.crop_width_px.get_value(),
            crop_enabled=crop_enabled,
        )
        effective_h = _effective_crop_size(
            det_h,
            self._diffraction_settings.crop_height_px.get_value(),
            crop_enabled=crop_enabled,
        )

        bin_x_limits = _bin_size_limits(effective_w)
        bin_y_limits = _bin_size_limits(effective_h)

        bin_x_value = (
            bin_x_limits.clamp(self._diffraction_settings.bin_size_x.get_value())
            if binning_enabled
            else 1
        )
        bin_y_value = (
            bin_y_limits.clamp(self._diffraction_settings.bin_size_y.get_value())
            if binning_enabled
            else 1
        )

        _set_spin_box(self._bin_size_x_spin_box, bin_x_limits, bin_x_value)
        _set_spin_box(self._bin_size_y_spin_box, bin_y_limits, bin_y_value)

    def _update(self, observable: Observable) -> None:
        if observable in self._observed or observable is self._extent_source:
            self._sync_model_to_view()
        else:
            super()._update(observable)


class UpsampleViewController(CheckableGroupBoxParameterViewController):
    """UpsampleStep — FFT zero-pad upsampling by an isotropic integer factor."""

    def __init__(self, diffraction_settings: DiffractionSettings) -> None:
        super().__init__(diffraction_settings.upsample_enabled, 'Upsample')
        self._factor_view_controller = SpinBoxParameterViewController(
            diffraction_settings.upsample_factor
        )

        layout = QGridLayout()
        layout.addWidget(QLabel('Factor:'), 0, 0)
        layout.addWidget(self._factor_view_controller.get_widget(), 0, 1)
        layout.setColumnStretch(1, 1)
        self.get_widget().setLayout(layout)


class PaddingViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, diffraction_settings: DiffractionSettings) -> None:
        super().__init__(diffraction_settings.padding_enabled, 'Pad')
        self._pad_x_view_controller = SpinBoxParameterViewController(diffraction_settings.pad_x)
        self._pad_y_view_controller = SpinBoxParameterViewController(diffraction_settings.pad_y)

        layout = QGridLayout()
        layout.addWidget(QLabel('Padding:'), 0, 0)
        layout.addWidget(self._pad_x_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._pad_y_view_controller.get_widget(), 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)


class ValueFilterViewController:
    """FilterValuesStep — zero pattern values outside [lower_bound, upper_bound)."""

    def __init__(self, settings: DiffractionSettings) -> None:
        self._lower_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.value_lower_bound_enabled, 'Value Lower Bound:'
        )
        self._lower_bound_view_controller = SpinBoxParameterViewController(
            settings.value_lower_bound
        )
        self._upper_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.value_upper_bound_enabled, 'Value Upper Bound:'
        )
        self._upper_bound_view_controller = SpinBoxParameterViewController(
            settings.value_upper_bound
        )

        layout = QGridLayout()
        layout.addWidget(self._lower_bound_enabled_view_controller.get_widget(), 0, 0)
        layout.addWidget(self._lower_bound_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._upper_bound_enabled_view_controller.get_widget(), 1, 0)
        layout.addWidget(self._upper_bound_view_controller.get_widget(), 1, 1)
        layout.setColumnStretch(1, 1)

        self._widget = QGroupBox('Value Filter')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class TotalCountsFilterViewController:
    """Drop patterns whose good-pixel total counts fall outside [lower_bound, upper_bound]
    (inclusive). Runs after the prep pipeline, so counts reflect the same patterns the
    reconstructor sees (i.e. after any pixel-value zeroing, crop, binning, and padding).
    """

    def __init__(self, settings: DiffractionSettings) -> None:
        self._lower_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.total_counts_lower_bound_enabled, 'Total Counts Lower Bound:'
        )
        self._lower_bound_view_controller = SpinBoxParameterViewController(
            settings.total_counts_lower_bound
        )
        self._upper_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.total_counts_upper_bound_enabled, 'Total Counts Upper Bound:'
        )
        self._upper_bound_view_controller = SpinBoxParameterViewController(
            settings.total_counts_upper_bound
        )

        layout = QGridLayout()
        layout.addWidget(self._lower_bound_enabled_view_controller.get_widget(), 0, 0)
        layout.addWidget(self._lower_bound_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._upper_bound_enabled_view_controller.get_widget(), 1, 0)
        layout.addWidget(self._upper_bound_view_controller.get_widget(), 1, 1)
        layout.setColumnStretch(1, 1)

        self._widget = QGroupBox('Total Counts Filter')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class TransformViewController:
    """HorizontalFlipStep + VerticalFlipStep + TransposeStep."""

    def __init__(self, settings: DiffractionSettings) -> None:
        self._hflip_view_controller = CheckBoxParameterViewController(
            settings.hflip, 'Flip Horizontal'
        )
        self._vflip_view_controller = CheckBoxParameterViewController(
            settings.vflip, 'Flip Vertical'
        )
        self._transpose_view_controller = CheckBoxParameterViewController(
            settings.transpose, 'Transpose'
        )

        layout = QGridLayout()
        layout.addWidget(self._hflip_view_controller.get_widget(), 0, 0)
        layout.addWidget(self._vflip_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._transpose_view_controller.get_widget(), 0, 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self._widget = QGroupBox('Transform')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class OpenDatasetWizardProcessingViewController(ParameterViewController):
    """Processing wizard page. Groups are laid out top-to-bottom in the
    DiffractionPrepPipeline execution order (see api/preprocess/diffraction.py):
    filter → crop → binning → upsample → padding → transform (hflip → vflip → transpose).
    Storage (memory map) and Bad Pixels are not part of the pipeline but are
    retained here as load-time concerns; the horizontal separator between them
    and Value Filter marks that boundary visually. Total Counts Filter is
    grouped next to Value Filter for user clarity but runs after the pipeline
    completes — it drops whole patterns, which no pipeline step is allowed to
    do.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        detector_settings: DetectorSettings,
        extent_source: DetectorExtentSource,
        api: DiffractionAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._storage_view_controller = StorageViewController(
            diffraction_settings, file_dialog_factory
        )
        self._bad_pixels_view_controller = BadPixelsViewController(
            detector_settings, api, file_dialog_factory
        )
        self._value_filter_view_controller = ValueFilterViewController(diffraction_settings)
        self._total_counts_filter_view_controller = TotalCountsFilterViewController(
            diffraction_settings
        )
        self._crop_view_controller = CropViewController(diffraction_settings, extent_source)
        self._binning_view_controller = BinningViewController(diffraction_settings, extent_source)
        self._upsample_view_controller = UpsampleViewController(diffraction_settings)
        self._padding_view_controller = PaddingViewController(diffraction_settings)
        self._transform_view_controller = TransformViewController(diffraction_settings)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        layout = QVBoxLayout()
        layout.addWidget(self._storage_view_controller.get_widget())
        layout.addWidget(self._bad_pixels_view_controller.get_widget())
        layout.addWidget(separator)
        layout.addWidget(self._value_filter_view_controller.get_widget())
        layout.addWidget(self._total_counts_filter_view_controller.get_widget())
        layout.addWidget(self._crop_view_controller.get_widget())
        layout.addWidget(self._binning_view_controller.get_widget())
        layout.addWidget(self._upsample_view_controller.get_widget())
        layout.addWidget(self._padding_view_controller.get_widget())
        layout.addWidget(self._transform_view_controller.get_widget())
        layout.addStretch()

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Processing')
        self._page._set_complete(True)
        self._page.setLayout(layout)

    def get_widget(self) -> QWizardPage:
        return self._page
