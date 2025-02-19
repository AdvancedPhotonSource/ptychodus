from typing import Final

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
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

from ....model.patterns import PatternSettings, PatternSizer
from ....view.patterns import OpenDatasetWizardPage

from ...data import FileDialogFactory
from ...parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)


class PatternLoadViewController(ParameterViewController):
    def __init__(self, settings: PatternSettings) -> None:
        super().__init__()
        self._view_controller = SpinBoxParameterViewController(
            settings.numberOfDataThreads,
        )
        self._widget = QGroupBox('Load')

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', self._view_controller.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PatternMemoryMapViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: PatternSettings, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__(settings.memmapEnabled, 'Memory Map Diffraction Data')
        self._view_controller = PathParameterViewController.createDirectoryChooser(
            settings.scratchDirectory, file_dialog_factory
        )

        layout = QFormLayout()
        layout.addRow('Scratch Directory:', self._view_controller.getWidget())
        self.getWidget().setLayout(layout)


class PatternCropViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.cropEnabled, 'Crop')
        self._settings = settings
        self._sizer = sizer

        self._center_x_spin_box = QSpinBox()
        self._center_y_spin_box = QSpinBox()
        self._width_spin_box = QSpinBox()
        self._height_spin_box = QSpinBox()
        self._flip_x_check_box = QCheckBox('Flip X')
        self._flip_y_check_box = QCheckBox('Flip Y')

        layout = QGridLayout()
        layout.addWidget(QLabel('Center:'), 0, 0)
        layout.addWidget(self._center_x_spin_box, 0, 1)
        layout.addWidget(self._center_y_spin_box, 0, 2)
        layout.addWidget(QLabel('Extent:'), 1, 0)
        layout.addWidget(self._width_spin_box, 1, 1)
        layout.addWidget(self._height_spin_box, 1, 2)
        layout.addWidget(QLabel('Axes:'), 2, 0)
        layout.addWidget(self._flip_x_check_box, 2, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._flip_y_check_box, 2, 2, Qt.AlignmentFlag.AlignHCenter)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._center_x_spin_box.valueChanged.connect(settings.cropCenterXInPixels.setValue)
        self._center_y_spin_box.valueChanged.connect(settings.cropCenterYInPixels.setValue)
        self._width_spin_box.valueChanged.connect(settings.cropWidthInPixels.setValue)
        self._height_spin_box.valueChanged.connect(settings.cropHeightInPixels.setValue)
        self._flip_x_check_box.toggled.connect(settings.flipXEnabled.setValue)
        self._flip_y_check_box.toggled.connect(settings.flipYEnabled.setValue)

        sizer.addObserver(self)

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

        self._flip_x_check_box.setChecked(self._settings.flipXEnabled.getValue())
        self._flip_y_check_box.setChecked(self._settings.flipYEnabled.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternBinningViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.binningEnabled, 'Bin Pixels')
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
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._bin_size_x_spin_box.valueChanged.connect(settings.binSizeX.setValue)
        self._bin_size_y_spin_box.valueChanged.connect(settings.binSizeY.setValue)

        sizer.addObserver(self)

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

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternPaddingViewController(CheckableGroupBoxParameterViewController):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.paddingEnabled, 'Pad')
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
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._pad_x_spin_box.valueChanged.connect(settings.padX.setValue)
        self._pad_y_spin_box.valueChanged.connect(settings.padY.setValue)

        sizer.addObserver(self)

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

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternTransformViewController:
    def __init__(self, settings: PatternSettings) -> None:
        self._lower_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.valueLowerBoundEnabled, 'Value Lower Bound:'
        )
        self._lower_bound_view_controller = SpinBoxParameterViewController(settings.valueLowerBound)
        self._upper_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.valueUpperBoundEnabled, 'Value upper Bound:'
        )
        self._upper_bound_view_controller = SpinBoxParameterViewController(settings.valueUpperBound)

        layout = QGridLayout()
        layout.addWidget(self._lower_bound_enabled_view_controller.getWidget(), 0, 0)
        layout.addWidget(self._lower_bound_view_controller.getWidget(), 0, 1, 1, 2)
        layout.addWidget(self._upper_bound_view_controller.getWidget(), 1, 0)
        layout.addWidget(self._upper_bound_view_controller.getWidget(), 1, 1, 1, 2)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        self._widget = QGroupBox('Transform')
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class OpenDatasetWizardPatternsViewController(ParameterViewController):
    def __init__(
        self, settings: PatternSettings, sizer: PatternSizer, file_dialog_factory: FileDialogFactory
    ) -> None:
        self._loadViewController = PatternLoadViewController(settings)
        self._memoryMapViewController = PatternMemoryMapViewController(
            settings, file_dialog_factory
        )
        self._cropViewController = PatternCropViewController(settings, sizer)
        self._binningViewController = PatternBinningViewController(settings, sizer)
        self._paddingViewController = PatternPaddingViewController(settings, sizer)
        self._transformViewController = PatternTransformViewController(settings)

        layout = QVBoxLayout()
        layout.addWidget(self._loadViewController.getWidget())
        layout.addWidget(self._memoryMapViewController.getWidget())
        layout.addWidget(self._cropViewController.getWidget())
        layout.addWidget(self._binningViewController.getWidget())
        layout.addWidget(self._paddingViewController.getWidget())
        layout.addWidget(self._transformViewController.getWidget())
        layout.addStretch()

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Pattern Processing')
        self._page._setComplete(True)  # FIXME why???
        self._page.setLayout(layout)

    def getWidget(self) -> QWizardPage:
        return self._page
