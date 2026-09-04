from collections.abc import Callable
import logging

import numpy

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.preprocess.diffraction import estimate_beam_center
from ptychodus.api.preprocess.noise import RobustStatistics, compute_robust_statistics

from ....model.diffraction import (
    DetectorSettings,
    DiffractionAPI,
    DiffractionSettings,
    DiffractionSummaryService,
)
from ....model.visualization import VisualizationEngine
from ....view.diffraction import OpenDatasetWizardPage
from ....view.widgets import ExceptionDialog

from ...data import FileDialogFactory
from ..detector_extent import DetectorExtentSource
from ...parameters import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)
from .summary import SummaryPanelViewController

logger = logging.getLogger(__name__)


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
        summary_service: DiffractionSummaryService,
    ) -> None:
        super().__init__(diffraction_settings.crop_enabled, 'Crop')
        self._diffraction_settings = diffraction_settings
        self._extent_source = extent_source
        self._summary_service = summary_service

        self._center_x_spin_box = QSpinBox()
        self._center_y_spin_box = QSpinBox()
        self._width_spin_box = QSpinBox()
        self._height_spin_box = QSpinBox()
        self._estimate_button = QPushButton('Estimate Beam Center')
        self._estimate_button.clicked.connect(self._on_estimate_clicked)
        self._estimate_button.setEnabled(False)

        layout = QGridLayout()
        layout.addWidget(QLabel('Center:'), 0, 0)
        layout.addWidget(self._center_x_spin_box, 0, 1)
        layout.addWidget(self._center_y_spin_box, 0, 2)
        layout.addWidget(QLabel('Extent:'), 1, 0)
        layout.addWidget(self._width_spin_box, 1, 1)
        layout.addWidget(self._height_spin_box, 1, 2)
        layout.addWidget(self._estimate_button, 2, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)

        self._observed = (
            diffraction_settings.beam_center_x_px,
            diffraction_settings.beam_center_y_px,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_height_px,
        )

        self._sync_model_to_view()
        self._sync_estimate_button_state()

        self._center_x_spin_box.valueChanged.connect(
            diffraction_settings.beam_center_x_px.set_value
        )
        self._center_y_spin_box.valueChanged.connect(
            diffraction_settings.beam_center_y_px.set_value
        )
        self._width_spin_box.valueChanged.connect(diffraction_settings.crop_width_px.set_value)
        self._height_spin_box.valueChanged.connect(diffraction_settings.crop_height_px.set_value)

        for parameter in self._observed:
            parameter.add_observer(self)
        extent_source.add_observer(self)
        summary_service.task_monitor.add_observer(self)

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
                self._diffraction_settings.beam_center_x_px.get_value()
            ),
        )
        _set_spin_box(
            self._center_y_spin_box,
            _crop_center_limits(det_h),
            _crop_center_limits(det_h).clamp(
                self._diffraction_settings.beam_center_y_px.get_value()
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

    def _sync_estimate_button_state(self) -> None:
        monitor = self._summary_service.task_monitor
        has_summary = self._summary_service.get_last_summary() is not None
        self._estimate_button.setEnabled(has_summary and not monitor.is_processing)

    def _on_estimate_clicked(self) -> None:
        summary = self._summary_service.get_last_summary()
        if summary is None:
            return
        try:
            center = estimate_beam_center(summary.mean_pattern)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Estimate Beam Center', exc)
            return
        self._diffraction_settings.beam_center_x_px.set_value(int(center.x_px))
        self._diffraction_settings.beam_center_y_px.set_value(int(center.y_px))

    def _update(self, observable: Observable) -> None:
        if observable is self._summary_service.task_monitor:
            self._sync_estimate_button_state()
        elif observable in self._observed or observable is self._extent_source:
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


class RobustStatisticsDialog(QDialog):
    """Modal helper for computing and applying robust bounds on total_counts.

    Freshly computes ``compute_robust_statistics(summary.total_counts)`` each
    time :meth:`exec_` is called so the median / MAD reflect whichever summary
    the service is currently holding. Apply Bounds writes to the total-counts
    settings and stays open so the user can tweak ``k`` and re-apply; Close
    dismisses.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        summary_service: DiffractionSummaryService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Total-Counts Robust Statistics')
        self.setModal(True)
        self._diffraction_settings = diffraction_settings
        self._summary_service = summary_service
        self._stats: RobustStatistics | None = None

        self._median_label = QLabel('—')
        self._mad_label = QLabel('—')
        self._k_spin_box = QDoubleSpinBox()
        self._k_spin_box.setRange(0.5, 20.0)
        self._k_spin_box.setSingleStep(0.5)
        self._k_spin_box.setDecimals(2)
        self._k_spin_box.setValue(3.0)

        self._apply_button = QPushButton('Apply Bounds')
        self._apply_button.clicked.connect(self._on_apply_clicked)
        self._close_button = QPushButton('Close')
        self._close_button.clicked.connect(self.accept)

        form_layout = QFormLayout()
        form_layout.addRow('Median:', self._median_label)
        form_layout.addRow('MAD:', self._mad_label)
        form_layout.addRow('k:', self._k_spin_box)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self._apply_button)
        button_row.addWidget(self._close_button)

        layout = QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def exec_(self) -> int:
        self._recompute_stats()
        return super().exec_()

    def _recompute_stats(self) -> None:
        summary = self._summary_service.get_last_summary()
        if summary is None or summary.total_counts.size == 0:
            self._stats = None
            self._median_label.setText('—')
            self._mad_label.setText('—')
            self._apply_button.setEnabled(False)
            return

        counts = numpy.asarray(summary.total_counts, dtype=numpy.float64)
        try:
            stats = compute_robust_statistics(counts)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Compute Robust Statistics', exc)
            self._stats = None
            self._apply_button.setEnabled(False)
            return

        self._stats = stats
        self._median_label.setText(f'{stats.median:.6g}')
        self._mad_label.setText(f'{stats.median_absolute_deviation:.6g}')
        self._apply_button.setEnabled(True)

    def _on_apply_clicked(self) -> None:
        stats = self._stats
        if stats is None:
            return

        bounds = stats.get_bounds(k=float(self._k_spin_box.value()), require_positive=True)
        lower_int = max(0, int(numpy.floor(bounds.lower)))
        upper_int = max(lower_int + 1, int(numpy.ceil(bounds.upper)))

        self._diffraction_settings.total_counts_lower_bound.set_value(lower_int)
        self._diffraction_settings.total_counts_upper_bound.set_value(upper_int)
        self._diffraction_settings.total_counts_lower_bound_enabled.set_value(True)
        self._diffraction_settings.total_counts_upper_bound_enabled.set_value(True)


class TotalCountsFilterViewController(Observer):
    """Drop patterns whose good-pixel total counts fall outside [lower_bound, upper_bound]
    (inclusive). Runs after the prep pipeline, so counts reflect the same patterns the
    reconstructor sees (i.e. after any pixel-value zeroing, crop, binning, and padding).
    """

    def __init__(
        self,
        settings: DiffractionSettings,
        summary_service: DiffractionSummaryService,
    ) -> None:
        super().__init__()
        self._summary_service = summary_service
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
        self._robust_stats_button = QPushButton('Robust Statistics…')
        self._robust_stats_button.clicked.connect(self._on_robust_stats_clicked)
        self._robust_stats_button.setEnabled(False)

        layout = QGridLayout()
        layout.addWidget(self._lower_bound_enabled_view_controller.get_widget(), 0, 0)
        layout.addWidget(self._lower_bound_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._upper_bound_enabled_view_controller.get_widget(), 1, 0)
        layout.addWidget(self._upper_bound_view_controller.get_widget(), 1, 1)
        layout.addWidget(self._robust_stats_button, 2, 0, 1, 2)
        layout.setColumnStretch(1, 1)

        self._widget = QGroupBox('Total Counts Filter')
        self._widget.setLayout(layout)

        self._dialog = RobustStatisticsDialog(settings, summary_service, parent=self._widget)

        summary_service.task_monitor.add_observer(self)
        self._sync_button_state()

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_button_state(self) -> None:
        monitor = self._summary_service.task_monitor
        has_summary = self._summary_service.get_last_summary() is not None
        self._robust_stats_button.setEnabled(has_summary and not monitor.is_processing)

    def _on_robust_stats_clicked(self) -> None:
        self._dialog.exec_()

    def _update(self, observable: Observable) -> None:
        if observable is self._summary_service.task_monitor:
            self._sync_button_state()


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
    """Processing wizard page. Split horizontally: preprocess-pipeline groups on
    the left (in :class:`DiffractionPrepPipeline` execution order), summary
    viewer + Crop + Total Counts Filter on the right so the user can tune
    those two groups against a rendered mean_pattern and a plot of per-pattern
    total counts.

    Storage (memory map) and Bad Pixels are load-time concerns rather than
    pipeline steps; the horizontal separator marks that boundary visually.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        detector_settings: DetectorSettings,
        extent_source: DetectorExtentSource,
        api: DiffractionAPI,
        summary_service: DiffractionSummaryService,
        summary_visualization_engine: VisualizationEngine,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
        get_pending_dataset_index: Callable[[], int],
    ) -> None:
        self._storage_view_controller = StorageViewController(
            diffraction_settings, file_dialog_factory
        )
        self._bad_pixels_view_controller = BadPixelsViewController(
            detector_settings, api, file_dialog_factory
        )
        self._value_filter_view_controller = ValueFilterViewController(diffraction_settings)
        self._total_counts_filter_view_controller = TotalCountsFilterViewController(
            diffraction_settings, summary_service
        )
        self._crop_view_controller = CropViewController(
            diffraction_settings, extent_source, summary_service
        )
        self._binning_view_controller = BinningViewController(diffraction_settings, extent_source)
        self._upsample_view_controller = UpsampleViewController(diffraction_settings)
        self._padding_view_controller = PaddingViewController(diffraction_settings)
        self._transform_view_controller = TransformViewController(diffraction_settings)
        self._summary_view_controller = SummaryPanelViewController(
            diffraction_settings,
            summary_service,
            summary_visualization_engine,
            status_bar,
            file_dialog_factory,
            get_pending_dataset_index,
        )

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        # Left pane — full preprocess-pipeline column, in DiffractionPrepPipeline order.
        # No trailing addStretch(): the layout fills the scroll viewport top-to-bottom.
        left_layout = QVBoxLayout()
        left_layout.addWidget(self._storage_view_controller.get_widget())
        left_layout.addWidget(self._bad_pixels_view_controller.get_widget())
        left_layout.addWidget(separator)
        left_layout.addWidget(self._value_filter_view_controller.get_widget())
        left_layout.addWidget(self._total_counts_filter_view_controller.get_widget())
        left_layout.addWidget(self._crop_view_controller.get_widget())
        left_layout.addWidget(self._binning_view_controller.get_widget())
        left_layout.addWidget(self._upsample_view_controller.get_widget())
        left_layout.addWidget(self._padding_view_controller.get_widget())
        left_layout.addWidget(self._transform_view_controller.get_widget())

        left_content = QWidget()
        left_content.setLayout(left_layout)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)
        # No horizontal scroll bar; horizontal size follows the content's
        # preferred width so groups render at their natural width and the
        # summary pane absorbs any extra window width.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Right pane — summary panel placed directly in the splitter. Its own
        # QSplitter(Vertical) governs the image/plot/controls proportions, so a
        # QScrollArea here would steal that stretch.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self._summary_view_controller.get_widget())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(splitter)

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Processing')
        self._page._set_complete(True)
        self._page.setLayout(outer_layout)

    def get_widget(self) -> QWizardPage:
        return self._page
