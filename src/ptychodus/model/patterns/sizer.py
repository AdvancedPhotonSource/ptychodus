from ptychodus.api.geometry import ImageExtent, Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter, RealParameter
from ptychodus.api.patterns import CropCenter

from .processor import (
    DiffractionPatternBinning,
    DiffractionPatternCrop,
    DiffractionPatternFilterValues,
    DiffractionPatternPadding,
    DiffractionPatternProcessor,
)
from .settings import DetectorSettings, PatternSettings


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

        detector_size.addObserver(self)
        detector_pixel_size_m.addObserver(self)
        crop_enabled.addObserver(self)
        crop_size.addObserver(self)
        crop_center.addObserver(self)
        binning_enabled.addObserver(self)
        bin_size.addObserver(self)
        padding_enabled.addObserver(self)
        pad_size.addObserver(self)

    def get_detector_size(self) -> int:
        return self._detector_size.getValue()

    def get_crop_size_limits(self) -> Interval[int]:
        return Interval[int](1, self.get_detector_size())

    def get_crop_size(self) -> int:
        if self._crop_enabled.getValue():
            limits = self.get_crop_size_limits()
            return limits.clamp(self._crop_size.getValue())

        return self.get_detector_size()

    def get_crop_center_limits(self) -> Interval[int]:
        xmin = (self.get_crop_size() + 1) // 2
        xmax = self.get_detector_size() - 1 - xmin
        return Interval[int](xmin, xmax)

    def get_crop_center(self) -> int:
        limits = self.get_crop_center_limits()
        return limits.clamp(self._crop_center.getValue())

    def get_bin_size_limits(self) -> Interval[int]:
        return Interval[int](1, self.get_crop_size())

    def get_bin_size(self) -> int:
        if self._binning_enabled.getValue():
            limits = self.get_bin_size_limits()
            return limits.clamp(self._bin_size.getValue())

        return 1

    def validate_bin_size(self) -> None:
        crop_size = self.get_crop_size()
        bin_size = self.get_bin_size()

        if crop_size % bin_size != 0:
            raise ValueError(f'Invalid binning size! ({crop_size=}, {bin_size=})')

    def get_pad_size(self) -> int:
        if self._padding_enabled.getValue():
            return self._pad_size.getValue()

        return 0

    def get_processed_size(self) -> int:
        return self.get_crop_size() // self.get_bin_size() + self.get_pad_size()

    def get_processed_pixel_size_m(self) -> float:
        return self.get_bin_size() * self._detector_pixel_size_m.getValue()

    def get_processed_size_m(self) -> float:
        return self.get_processed_size() * self.get_processed_pixel_size_m()

    def update(self, observable: Observable) -> None:
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
            self.notifyObservers()


class PatternSizer(Observable, Observer):
    def __init__(
        self, detector_settings: DetectorSettings, pattern_settings: PatternSettings
    ) -> None:
        super().__init__()
        self._pattern_settings = pattern_settings
        self.axis_x = PatternAxisSizer(
            detector_settings.widthInPixels,
            detector_settings.pixelWidthInMeters,
            pattern_settings.cropEnabled,
            pattern_settings.cropWidthInPixels,
            pattern_settings.cropCenterXInPixels,
            pattern_settings.binningEnabled,
            pattern_settings.binSizeX,
            pattern_settings.paddingEnabled,
            pattern_settings.padX,
        )
        self.axis_y = PatternAxisSizer(
            detector_settings.heightInPixels,
            detector_settings.pixelHeightInMeters,
            pattern_settings.cropEnabled,
            pattern_settings.cropHeightInPixels,
            pattern_settings.cropCenterYInPixels,
            pattern_settings.binningEnabled,
            pattern_settings.binSizeY,
            pattern_settings.paddingEnabled,
            pattern_settings.padY,
        )

        self.axis_x.addObserver(self)
        self.axis_y.addObserver(self)

    def get_processed_width_m(self) -> float:
        return self.axis_x.get_processed_size_m()

    def get_processed_height_m(self) -> float:
        return self.axis_y.get_processed_size_m()

    def get_processed_image_extent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self.axis_x.get_processed_size(),
            heightInPixels=self.axis_y.get_processed_size(),
        )

    def get_processed_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.axis_x.get_processed_pixel_size_m(),
            heightInMeters=self.axis_y.get_processed_pixel_size_m(),
        )

    def get_processor(self) -> DiffractionPatternProcessor:
        value_lower_bound: int | None = None
        value_upper_bound: int | None = None
        crop: DiffractionPatternCrop | None = None
        binning: DiffractionPatternBinning | None = None
        padding: DiffractionPatternPadding | None = None

        if self._pattern_settings.valueUpperBoundEnabled.getValue():
            value_lower_bound = self._pattern_settings.valueLowerBound.getValue()

        if self._pattern_settings.valueUpperBoundEnabled.getValue():
            value_upper_bound = self._pattern_settings.valueUpperBound.getValue()

        filter_values = DiffractionPatternFilterValues(
            lower_bound=value_lower_bound,
            upper_bound=value_upper_bound,
        )

        if self._pattern_settings.cropEnabled.getValue():
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

        if self._pattern_settings.binningEnabled.getValue():
            self.axis_x.validate_bin_size()
            self.axis_y.validate_bin_size()
            binning = DiffractionPatternBinning(
                bin_size_x=self.axis_x.get_bin_size(),
                bin_size_y=self.axis_y.get_bin_size(),
            )

        if self._pattern_settings.paddingEnabled.getValue():
            padding = DiffractionPatternPadding(
                pad_x=self.axis_x.get_pad_size(),
                pad_y=self.axis_y.get_pad_size(),
            )

        return DiffractionPatternProcessor(
            filter_values=filter_values,
            crop=crop,
            binning=binning,
            padding=padding,
            flip_x=self._pattern_settings.flipXEnabled.getValue(),
            flip_y=self._pattern_settings.flipYEnabled.getValue(),
        )

    def update(self, observable: Observable) -> None:
        if observable in (self.axis_x, self.axis_y):
            self.notifyObservers()
