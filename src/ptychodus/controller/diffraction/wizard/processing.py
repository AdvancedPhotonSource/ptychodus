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

from ....model.diffraction import DiffractionSettings
from ....view.diffraction import OpenDatasetWizardPage

from ...data import FileDialogFactory
from ..detector_extent import DetectorExtentSource
from ...parametric import (
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


class FlipViewController:
    """HorizontalFlipStep + VerticalFlipStep."""

    def __init__(self, settings: DiffractionSettings) -> None:
        self._hflip_view_controller = CheckBoxParameterViewController(
            settings.hflip, 'Flip Horizontal'
        )
        self._vflip_view_controller = CheckBoxParameterViewController(
            settings.vflip, 'Flip Vertical'
        )

        layout = QGridLayout()
        layout.addWidget(self._hflip_view_controller.get_widget(), 0, 0)
        layout.addWidget(self._vflip_view_controller.get_widget(), 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self._widget = QGroupBox('Flip')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class TransposeViewController:
    """TransposeStep — swap the last two axes."""

    def __init__(self, settings: DiffractionSettings) -> None:
        self._transpose_view_controller = CheckBoxParameterViewController(
            settings.transpose, 'Transpose'
        )

        layout = QGridLayout()
        layout.addWidget(self._transpose_view_controller.get_widget(), 0, 0)

        self._widget = QGroupBox('Transpose')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class OpenDatasetWizardProcessingViewController(ParameterViewController):
    """Processing wizard page. Groups are laid out top-to-bottom in the
    DiffractionPrepPipeline execution order (see api/diffraction_prep.py):
    filter → crop → binning → padding → hflip → vflip → transpose. Storage
    (memory map) is not part of the pipeline but is retained here as a
    load-time concern; the horizontal separator between it and Value Filter
    marks that boundary visually.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        extent_source: DetectorExtentSource,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._storage_view_controller = StorageViewController(
            diffraction_settings, file_dialog_factory
        )
        self._value_filter_view_controller = ValueFilterViewController(diffraction_settings)
        self._crop_view_controller = CropViewController(diffraction_settings, extent_source)
        self._binning_view_controller = BinningViewController(diffraction_settings, extent_source)
        self._padding_view_controller = PaddingViewController(diffraction_settings)
        self._flip_view_controller = FlipViewController(diffraction_settings)
        self._transpose_view_controller = TransposeViewController(diffraction_settings)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        layout = QVBoxLayout()
        layout.addWidget(self._storage_view_controller.get_widget())
        layout.addWidget(separator)
        layout.addWidget(self._value_filter_view_controller.get_widget())
        layout.addWidget(self._crop_view_controller.get_widget())
        layout.addWidget(self._binning_view_controller.get_widget())
        layout.addWidget(self._padding_view_controller.get_widget())
        layout.addWidget(self._flip_view_controller.get_widget())
        layout.addWidget(self._transpose_view_controller.get_widget())
        layout.addStretch()

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Processing')
        self._page._set_complete(True)
        self._page.setLayout(layout)

    def get_widget(self) -> QWizardPage:
        return self._page
