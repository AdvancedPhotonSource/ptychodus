from ptychodus.api.geometry import ImageExtent, Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter, RealParameter
from ptychodus.api.diffraction import CropCenter

from .processor import (
    DiffractionPatternBinning,
    DiffractionPatternCrop,
    DiffractionPatternFilterValues,
    DiffractionPatternPadding,
    DiffractionPatternProcessor,
)
from .settings import DetectorSettings, DiffractionSettings


class PatternAxisSizer(Observable, Observer):
    def __init__(
        self,
        detector_size: IntegerParameter,
        detector_pixel_size_m: RealParameter,
        crop_enabled: BooleanParameter,
        crop_size: IntegerParameter,
        crop_center: IntegerParameter,
        binning_enabled: BooleanParameter,
        bin_size: IntegerParameter,
        padding_enabled: BooleanParameter,
        pad_size: IntegerParameter,
    ) -> None:
        super().__init__()
        self._detector_size = detector_size
        self._detector_pixel_size_m = detector_pixel_size_m
        self._crop_enabled = crop_enabled
        self._crop_size = crop_size
        self._crop_center = crop_center
        self._binning_enabled = binning_enabled
        self._bin_size = bin_size
        self._padding_enabled = padding_enabled
        self._pad_size = pad_size

        detector_size.add_observer(self)
        detector_pixel_size_m.add_observer(self)
        crop_enabled.add_observer(self)
        crop_size.add_observer(self)
        crop_center.add_observer(self)
        binning_enabled.add_observer(self)
        bin_size.add_observer(self)
        padding_enabled.add_observer(self)
        pad_size.add_observer(self)

    def get_detector_size(self) -> int:
        return self._detector_size.get_value()

    def get_crop_size_limits(self) -> Interval[int]:
        return Interval[int](1, self.get_detector_size())

    def get_crop_size(self) -> int:
        if self._crop_enabled.get_value():
            limits = self.get_crop_size_limits()
            return limits.clamp(self._crop_size.get_value())

        return self.get_detector_size()

    def get_crop_center_limits(self) -> Interval[int]:
        xmin = (self.get_crop_size() + 1) // 2
        xmax = self.get_detector_size() - 1 - xmin
        return Interval[int](xmin, xmax)

    def get_crop_center(self) -> int:
        limits = self.get_crop_center_limits()
        return limits.clamp(self._crop_center.get_value())

    def get_bin_size_limits(self) -> Interval[int]:
        return Interval[int](1, self.get_crop_size())

    def get_bin_size(self) -> int:
        if self._binning_enabled.get_value():
            limits = self.get_bin_size_limits()
            return limits.clamp(self._bin_size.get_value())

        return 1

    def validate_bin_size(self) -> None:
        crop_size = self.get_crop_size()
        bin_size = self.get_bin_size()

        if crop_size % bin_size != 0:
            raise ValueError(f'Invalid binning size! ({crop_size=}, {bin_size=})')

    def get_pad_size(self) -> int:
        if self._padding_enabled.get_value():
            return self._pad_size.get_value()

        return 0

    def get_processed_size(self) -> int:
        return self.get_crop_size() // self.get_bin_size() + self.get_pad_size()

    def get_processed_pixel_size_m(self) -> float:
        return self.get_bin_size() * self._detector_pixel_size_m.get_value()

    def get_processed_size_m(self) -> float:
        return self.get_processed_size() * self.get_processed_pixel_size_m()

    def _update(self, observable: Observable) -> None:
        if observable in (
            self._detector_size,
            self._detector_pixel_size_m,
            self._crop_enabled,
            self._crop_size,
            self._crop_center,
            self._binning_enabled,
            self._bin_size,
            self._padding_enabled,
            self._pad_size,
        ):
            self.notify_observers()


class PatternSizer(Observable, Observer):
    def __init__(
        self, detector_settings: DetectorSettings, diffraction_settings: DiffractionSettings
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings
        self.axis_x = PatternAxisSizer(
            detector_settings.width_px,
            detector_settings.pixel_width_m,
            diffraction_settings.is_crop_enabled,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_center_x_px,
            diffraction_settings.is_binning_enabled,
            diffraction_settings.bin_size_x,
            diffraction_settings.is_padding_enabled,
            diffraction_settings.pad_x,
        )
        self.axis_y = PatternAxisSizer(
            detector_settings.height_px,
            detector_settings.pixel_height_m,
            diffraction_settings.is_crop_enabled,
            diffraction_settings.crop_height_px,
            diffraction_settings.crop_center_y_px,
            diffraction_settings.is_binning_enabled,
            diffraction_settings.bin_size_y,
            diffraction_settings.is_padding_enabled,
            diffraction_settings.pad_y,
        )

        self.axis_x.add_observer(self)
        self.axis_y.add_observer(self)

    def get_detector_extent(self) -> ImageExtent:
        return ImageExtent(
            width_px=self.axis_x.get_detector_size(),
            height_px=self.axis_y.get_detector_size(),
        )

    def get_processed_width_m(self) -> float:
        return self.axis_x.get_processed_size_m()

    def get_processed_height_m(self) -> float:
        return self.axis_y.get_processed_size_m()

    def get_processed_image_extent(self) -> ImageExtent:
        return ImageExtent(
            width_px=self.axis_x.get_processed_size(),
            height_px=self.axis_y.get_processed_size(),
        )

    def get_processed_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.axis_x.get_processed_pixel_size_m(),
            height_m=self.axis_y.get_processed_pixel_size_m(),
        )

    def get_processor(self) -> DiffractionPatternProcessor:
        value_lower_bound: int | None = None
        value_upper_bound: int | None = None
        crop: DiffractionPatternCrop | None = None
        binning: DiffractionPatternBinning | None = None
        padding: DiffractionPatternPadding | None = None

        if self._diffraction_settings.is_value_upper_bound_enabled.get_value():
            value_lower_bound = self._diffraction_settings.value_lower_bound.get_value()

        if self._diffraction_settings.is_value_upper_bound_enabled.get_value():
            value_upper_bound = self._diffraction_settings.value_upper_bound.get_value()

        filter_values = DiffractionPatternFilterValues(
            lower_bound=value_lower_bound,
            upper_bound=value_upper_bound,
        )

        if self._diffraction_settings.is_crop_enabled.get_value():
            crop = DiffractionPatternCrop(
                center=CropCenter(
                    self.axis_x.get_crop_center(),
                    self.axis_y.get_crop_center(),
                ),
                extent=ImageExtent(
                    self.axis_x.get_crop_size(),
                    self.axis_y.get_crop_size(),
                ),
            )

        if self._diffraction_settings.is_binning_enabled.get_value():
            self.axis_x.validate_bin_size()
            self.axis_y.validate_bin_size()
            binning = DiffractionPatternBinning(
                bin_size_x=self.axis_x.get_bin_size(),
                bin_size_y=self.axis_y.get_bin_size(),
            )

        if self._diffraction_settings.is_padding_enabled.get_value():
            padding = DiffractionPatternPadding(
                pad_x=self.axis_x.get_pad_size(),
                pad_y=self.axis_y.get_pad_size(),
            )

        return DiffractionPatternProcessor(
            filter_values=filter_values,
            crop=crop,
            binning=binning,
            padding=padding,
            hflip=self._diffraction_settings.hflip.get_value(),
            vflip=self._diffraction_settings.vflip.get_value(),
            transpose=self._diffraction_settings.transpose.get_value(),
        )

    def _update(self, observable: Observable) -> None:
        if observable in (self.axis_x, self.axis_y):
            self.notify_observers()
