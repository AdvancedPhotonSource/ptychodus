from typing import Final

from PyQt5.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizardPage,
)

from ptychodus.api.observer import Observable

from ....model.diffraction import DiffractionSettings, PatternSizer
from ....view.diffraction import OpenDatasetWizardPage

from ...data import FileDialogFactory
from ...parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)


class PatternLoadViewController(ParameterViewController):
    def __init__(self, settings: DiffractionSettings) -> None:
        super().__init__()
        self._view_controller = SpinBoxParameterViewController(
            settings.num_data_threads,
        )
        self._widget = QGroupBox('Load')

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', self._view_controller.get_widget())
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class PatternMemoryMapViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self, settings: DiffractionSettings, file_dialog_factory: FileDialogFactory
    ) -> None:
        super().__init__(settings.is_memmap_enabled, 'Memory Map Diffraction Data')
        self._view_controller = PathParameterViewController.create_directory_chooser(
            settings.scratch_directory, file_dialog_factory
        )

        layout = QFormLayout()
        layout.addRow('Scratch Directory:', self._view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PatternCropViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.is_crop_enabled, 'Crop')
        self._settings = settings
        self._sizer = sizer

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

        self._sync_model_to_view()

        self._center_x_spin_box.valueChanged.connect(settings.crop_center_x_px.set_value)
        self._center_y_spin_box.valueChanged.connect(settings.crop_center_y_px.set_value)
        self._width_spin_box.valueChanged.connect(settings.crop_width_px.set_value)
        self._height_spin_box.valueChanged.connect(settings.crop_height_px.set_value)

        sizer.add_observer(self)

    def _sync_model_to_view(self) -> None:
        center_x = self._sizer.axis_x.get_crop_center()
        center_y = self._sizer.axis_y.get_crop_center()
        width = self._sizer.axis_x.get_crop_size()
        height = self._sizer.axis_y.get_crop_size()

        center_x_limits = self._sizer.axis_x.get_crop_center_limits()
        center_y_limits = self._sizer.axis_y.get_crop_center_limits()
        width_limits = self._sizer.axis_x.get_crop_size_limits()
        height_limits = self._sizer.axis_y.get_crop_size_limits()

        self._center_x_spin_box.blockSignals(True)
        self._center_x_spin_box.setRange(center_x_limits.lower, center_x_limits.upper)
        self._center_x_spin_box.setValue(center_x)
        self._center_x_spin_box.blockSignals(False)

        self._center_y_spin_box.blockSignals(True)
        self._center_y_spin_box.setRange(center_y_limits.lower, center_y_limits.upper)
        self._center_y_spin_box.setValue(center_y)
        self._center_y_spin_box.blockSignals(False)

        self._width_spin_box.blockSignals(True)
        self._width_spin_box.setRange(width_limits.lower, width_limits.upper)
        self._width_spin_box.setValue(width)
        self._width_spin_box.blockSignals(False)

        self._height_spin_box.blockSignals(True)
        self._height_spin_box.setRange(height_limits.lower, height_limits.upper)
        self._height_spin_box.setValue(height)
        self._height_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super()._update(observable)


class PatternBinningViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.is_binning_enabled, 'Bin Pixels')
        self._settings = settings
        self._sizer = sizer

        self._bin_size_x_spin_box = QSpinBox()
        self._bin_size_y_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Bin Size:'), 0, 0)
        layout.addWidget(self._bin_size_x_spin_box, 0, 1)
        layout.addWidget(self._bin_size_y_spin_box, 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)

        self._sync_model_to_view()

        self._bin_size_x_spin_box.valueChanged.connect(settings.bin_size_x.set_value)
        self._bin_size_y_spin_box.valueChanged.connect(settings.bin_size_y.set_value)

        sizer.add_observer(self)

    def _sync_model_to_view(self) -> None:
        bin_size_x = self._sizer.axis_x.get_bin_size()
        bin_size_y = self._sizer.axis_y.get_bin_size()

        bin_size_x_limits = self._sizer.axis_x.get_bin_size_limits()
        bin_size_y_limits = self._sizer.axis_y.get_bin_size_limits()

        self._bin_size_x_spin_box.blockSignals(True)
        self._bin_size_x_spin_box.setRange(bin_size_x_limits.lower, bin_size_x_limits.upper)
        self._bin_size_x_spin_box.setValue(bin_size_x)
        self._bin_size_x_spin_box.blockSignals(False)

        self._bin_size_y_spin_box.blockSignals(True)
        self._bin_size_y_spin_box.setRange(bin_size_y_limits.lower, bin_size_y_limits.upper)
        self._bin_size_y_spin_box.setValue(bin_size_y)
        self._bin_size_y_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super()._update(observable)


class PatternPaddingViewController(CheckableGroupBoxParameterViewController):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.is_padding_enabled, 'Pad')
        self._settings = settings
        self._sizer = sizer

        self._pad_x_spin_box = QSpinBox()
        self._pad_y_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Padding:'), 0, 0)
        layout.addWidget(self._pad_x_spin_box, 0, 1)
        layout.addWidget(self._pad_y_spin_box, 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.get_widget().setLayout(layout)

        self._sync_model_to_view()

        self._pad_x_spin_box.valueChanged.connect(settings.pad_x.set_value)
        self._pad_y_spin_box.valueChanged.connect(settings.pad_y.set_value)

        sizer.add_observer(self)

    def _sync_model_to_view(self) -> None:
        pad_x = self._sizer.axis_x.get_pad_size()
        pad_y = self._sizer.axis_y.get_pad_size()

        self._pad_x_spin_box.blockSignals(True)
        self._pad_x_spin_box.setRange(0, self.MAX_INT)
        self._pad_x_spin_box.setValue(pad_x)
        self._pad_x_spin_box.blockSignals(False)

        self._pad_y_spin_box.blockSignals(True)
        self._pad_y_spin_box.setRange(0, self.MAX_INT)
        self._pad_y_spin_box.setValue(pad_y)
        self._pad_y_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super()._update(observable)


class PatternTransformViewController:
    def __init__(
        self, settings: DiffractionSettings, file_dialog_factory: FileDialogFactory
    ) -> None:
        self._hflip_view_controller = CheckBoxParameterViewController(
            settings.hflip, 'Flip Horizontal'
        )
        self._vflip_view_controller = CheckBoxParameterViewController(
            settings.vflip, 'Flip Vertical'
        )
        self._transpose_view_controller = CheckBoxParameterViewController(
            settings.transpose, 'Transpose'
        )
        self._lower_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.is_value_lower_bound_enabled, 'Value Lower Bound:'
        )
        self._lower_bound_view_controller = SpinBoxParameterViewController(
            settings.value_lower_bound
        )
        self._upper_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.is_value_upper_bound_enabled, 'Value upper Bound:'
        )
        self._upper_bound_view_controller = SpinBoxParameterViewController(
            settings.value_upper_bound
        )

        layout = QGridLayout()
        layout.addWidget(QLabel('Axes:'), 0, 0)
        layout.addWidget(self._hflip_view_controller.get_widget(), 0, 1)
        layout.addWidget(self._vflip_view_controller.get_widget(), 0, 2)
        layout.addWidget(self._transpose_view_controller.get_widget(), 0, 3)
        layout.addWidget(self._lower_bound_enabled_view_controller.get_widget(), 1, 0)
        layout.addWidget(self._lower_bound_view_controller.get_widget(), 1, 1, 1, 3)
        layout.addWidget(self._upper_bound_enabled_view_controller.get_widget(), 2, 0)
        layout.addWidget(self._upper_bound_view_controller.get_widget(), 2, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        self._widget = QGroupBox('Transform')
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class OpenDatasetWizardPatternsViewController(ParameterViewController):
    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._load_view_controller = PatternLoadViewController(settings)
        self._memory_map_view_controller = PatternMemoryMapViewController(
            settings, file_dialog_factory
        )
        self._crop_view_controller = PatternCropViewController(settings, sizer)
        self._binning_view_controller = PatternBinningViewController(settings, sizer)
        self._padding_view_controller = PatternPaddingViewController(settings, sizer)
        self._transform_view_controller = PatternTransformViewController(
            settings, file_dialog_factory
        )

        layout = QVBoxLayout()
        layout.addWidget(self._load_view_controller.get_widget())
        layout.addWidget(self._memory_map_view_controller.get_widget())
        layout.addWidget(self._crop_view_controller.get_widget())
        layout.addWidget(self._binning_view_controller.get_widget())
        layout.addWidget(self._padding_view_controller.get_widget())
        layout.addWidget(self._transform_view_controller.get_widget())
        layout.addStretch()

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Pattern Processing')
        self._page._set_complete(True)
        self._page.setLayout(layout)

    def get_widget(self) -> QWizardPage:
        return self._page
