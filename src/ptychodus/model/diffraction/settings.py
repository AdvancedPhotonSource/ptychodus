from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry, get_ptychodus_dir


class DetectorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Detector')
        self._group.add_observer(self)

        self.pixel_width_m = self._group.create_real_parameter(
            'PixelWidthInMeters', 75e-6, minimum=0.0
        )
        self.pixel_height_m = self._group.create_real_parameter(
            'PixelHeightInMeters', 75e-6, minimum=0.0
        )

        self.bad_pixels_enabled = self._group.create_boolean_parameter('BadPixelsEnabled', False)
        self.bad_pixels_file_type = self._group.create_string_parameter(
            'BadPixelsFileType', 'NPY_Bad_Pixels'
        )
        self.bad_pixels_file_path = self._group.create_path_parameter(
            'BadPixelsFilePath', Path('/path/to/bad_pixels.npy')
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class DiffractionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Diffraction')
        self._group.add_observer(self)

        self.file_type = self._group.create_string_parameter('FileType', 'NPY')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/data.npy'))
        self.memmap_enabled = self._group.create_boolean_parameter('MemmapEnabled', False)
        self.scratch_directory = self._group.create_path_parameter(
            'ScratchDirectory', get_ptychodus_dir()
        )

        self.crop_enabled = self._group.create_boolean_parameter('CropEnabled', True)
        self.crop_center_x_px = self._group.create_integer_parameter(
            'CropCenterXInPixels', 32, minimum=0
        )
        self.crop_center_y_px = self._group.create_integer_parameter(
            'CropCenterYInPixels', 32, minimum=0
        )
        self.crop_width_px = self._group.create_integer_parameter(
            'CropWidthInPixels', 64, minimum=1
        )
        self.crop_height_px = self._group.create_integer_parameter(
            'CropHeightInPixels', 64, minimum=1
        )

        self.binning_enabled = self._group.create_boolean_parameter('BinningEnabled', False)
        self.bin_size_x = self._group.create_integer_parameter('BinSizeX', 1, minimum=1)
        self.bin_size_y = self._group.create_integer_parameter('BinSizeY', 1, minimum=1)

        self.upsample_enabled = self._group.create_boolean_parameter('UpsampleEnabled', False)
        self.upsample_factor = self._group.create_integer_parameter('UpsampleFactor', 1, minimum=1)

        self.padding_enabled = self._group.create_boolean_parameter('PaddingEnabled', False)
        self.pad_x = self._group.create_integer_parameter('PadX', 0, minimum=0)
        self.pad_y = self._group.create_integer_parameter('PadY', 0, minimum=0)

        self.hflip = self._group.create_boolean_parameter('FlipHorizontal', False)
        self.vflip = self._group.create_boolean_parameter('FlipVertical', False)
        self.transpose = self._group.create_boolean_parameter('Transpose', False)

        self.value_lower_bound_enabled = self._group.create_boolean_parameter(
            'ValueLowerBoundEnabled', False
        )
        self.value_lower_bound = self._group.create_integer_parameter(
            'ValueLowerBound', 0, minimum=0
        )
        self.value_upper_bound_enabled = self._group.create_boolean_parameter(
            'ValueUpperBoundEnabled', False
        )
        self.value_upper_bound = self._group.create_integer_parameter(
            'ValueUpperBound', 65535, minimum=0
        )

        self.total_counts_lower_bound_enabled = self._group.create_boolean_parameter(
            'TotalCountsLowerBoundEnabled', False
        )
        self.total_counts_lower_bound = self._group.create_integer_parameter(
            'TotalCountsLowerBound', 0, minimum=0
        )
        self.total_counts_upper_bound_enabled = self._group.create_boolean_parameter(
            'TotalCountsUpperBoundEnabled', False
        )
        self.total_counts_upper_bound = self._group.create_integer_parameter(
            'TotalCountsUpperBound', 1_000_000_000, minimum=0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
