from ptychodus.api.geometry import ImageExtent, Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.patterns import BooleanArrayType, CropCenter

from .processor import (
    DiffractionPatternBinning,
    DiffractionPatternCrop,
    DiffractionPatternPadding,
    DiffractionPatternProcessor,
)
from .settings import DetectorSettings, PatternSettings


class PatternSizer(Observable, Observer):
    def __init__(
        self, detector_settings: DetectorSettings, pattern_settings: PatternSettings
    ) -> None:
        super().__init__()
        self._detector_settings = detector_settings
        self._pattern_settings = pattern_settings

        pattern_settings.addObserver(self)
        detector_settings.addObserver(self)

    @property
    def _is_crop_enabled(self) -> bool:
        return self._pattern_settings.cropEnabled.getValue()

    @property
    def _is_binning_enabled(self) -> bool:
        # FIXME enable binning
        return self._pattern_settings.binningEnabled.getValue()

    @property
    def _is_padding_enabled(self) -> bool:
        # FIXME enable padding
        return self._pattern_settings.paddingEnabled.getValue()

    def getCropWidthLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector_settings.widthInPixels.getValue())

    def getWidthInPixels(self) -> int:
        if self._is_crop_enabled:
            limits = self.getCropWidthLimitsInPixels()
            return limits.clamp(self._pattern_settings.cropWidthInPixels.getValue())

        return self._detector_settings.widthInPixels.getValue()

    def getCropCenterXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector_settings.widthInPixels.getValue())

    def getCropCenterXInPixels(self) -> int:
        limitsInPixels = self.getCropCenterXLimitsInPixels()
        return (
            limitsInPixels.clamp(self._pattern_settings.cropCenterXInPixels.getValue())
            if self._is_crop_enabled
            else limitsInPixels.midrange
        )

    def getDetectorPixelWidthInMeters(self) -> float:
        width_m = self._detector_settings.pixelWidthInMeters.getValue()

        if self._is_binning_enabled:
            width_m *= self._pattern_settings.binSizeX.getValue()

        return width_m

    def getWidthInMeters(self) -> float:
        return self.getWidthInPixels() * self.getDetectorPixelWidthInMeters()

    def getCropHeightLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector_settings.heightInPixels.getValue())

    def getHeightInPixels(self) -> int:
        if self._is_crop_enabled:
            limits = self.getCropHeightLimitsInPixels()
            return limits.clamp(self._pattern_settings.cropHeightInPixels.getValue())

        return self._detector_settings.heightInPixels.getValue()

    def getCropCenterYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector_settings.heightInPixels.getValue())

    def getCropCenterYInPixels(self) -> int:
        limitsInPixels = self.getCropCenterYLimitsInPixels()
        return (
            limitsInPixels.clamp(self._pattern_settings.cropCenterYInPixels.getValue())
            if self._is_crop_enabled
            else limitsInPixels.midrange
        )

    def getDetectorPixelHeightInMeters(self) -> float:
        height_m = self._detector_settings.pixelHeightInMeters.getValue()

        if self._is_binning_enabled:
            height_m *= self._pattern_settings.binSizeY.getValue()

        return height_m

    def getHeightInMeters(self) -> float:
        return self.getHeightInPixels() * self.getDetectorPixelHeightInMeters()

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self.getWidthInPixels(),
            heightInPixels=self.getHeightInPixels(),
        )

    def getDetectorPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.getDetectorPixelWidthInMeters(),
            heightInMeters=self.getDetectorPixelHeightInMeters(),
        )

    @staticmethod
    def _get_safe_position(x: int, w: int, W: int) -> int:
        xmin = (w + 1) // 2
        xmax = W - 1 - xmin
        return min(max(xmin, x), xmax)

    def _get_crop(self) -> DiffractionPatternCrop | None:
        if self._is_crop_enabled:
            width_px = self.getWidthInPixels()
            height_px = self.getHeightInPixels()

            position_x_px = self._get_safe_position(
                self.getCropCenterXInPixels(),
                width_px,
                self._detector_settings.widthInPixels.getValue(),
            )
            position_y_px = self._get_safe_position(
                self.getCropCenterYInPixels(),
                height_px,
                self._detector_settings.heightInPixels.getValue(),
            )

            return DiffractionPatternCrop(
                center=CropCenter(position_x_px, position_y_px),
                extent=ImageExtent(width_px, height_px),
            )

        return None

    def _get_binning(self) -> DiffractionPatternBinning | None:
        if self._is_binning_enabled:
            width_px = self.getWidthInPixels()
            bin_size_x = self._pattern_settings.binSizeX.getValue()
            binned_width_px = width_px // bin_size_x

            if binned_width_px * bin_size_x != width_px:
                raise ValueError(f'Invalid binning size! ({bin_size_x=}, {width_px=})')

            height_px = self.getHeightInPixels()
            bin_size_y = self._pattern_settings.binSizeY.getValue()
            binned_height_px = height_px // bin_size_y

            if binned_height_px * bin_size_y != height_px:
                raise ValueError(f'Invalid binning size! ({bin_size_y=}, {height_px=})')

            return DiffractionPatternBinning(
                bin_size_x=self._pattern_settings.binSizeX.getValue(),
                bin_size_y=self._pattern_settings.binSizeY.getValue(),
            )

        return None

    def _get_padding(self) -> DiffractionPatternPadding | None:
        if self._is_padding_enabled:
            return DiffractionPatternPadding(
                pad_x=self._pattern_settings.padX.getValue(),
                pad_y=self._pattern_settings.padY.getValue(),
            )

        return None

    def get_processor(self, bad_pixels: BooleanArrayType) -> DiffractionPatternProcessor:
        return DiffractionPatternProcessor(
            bad_pixels=bad_pixels,
            crop=self._get_crop(),
            binning=self._get_binning(),
            padding=self._get_padding(),
            flip_x=self._pattern_settings.flipXEnabled.getValue(),
            flip_y=self._pattern_settings.flipYEnabled.getValue(),
        )

    def update(self, observable: Observable) -> None:
        if observable is self._pattern_settings:
            self.notifyObservers()
        elif observable is self._detector_settings:
            self.notifyObservers()
