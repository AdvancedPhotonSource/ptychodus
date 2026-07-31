from ptychodus.api.diffraction import CropCenter
from ptychodus.api.diffraction_prep import (
    BinningStep,
    CropStep,
    DiffractionPrepPipeline,
    DiffractionPrepStepUnion,
    FilterValuesStep,
    HorizontalFlipStep,
    PaddingStep,
    TransposeStep,
    VerticalFlipStep,
)
from ptychodus.api.geometry import ImageExtent, Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter

from .settings import DetectorSettings, DiffractionSettings


class PatternAxisSizer(Observable, Observer):
    def __init__(
        self,
        detector_size: IntegerParameter,
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
        self._crop_enabled = crop_enabled
        self._crop_size = crop_size
        self._crop_center = crop_center
        self._binning_enabled = binning_enabled
        self._bin_size = bin_size
        self._padding_enabled = padding_enabled
        self._pad_size = pad_size

        detector_size.add_observer(self)
        crop_enabled.add_observer(self)
        crop_size.add_observer(self)
        crop_center.add_observer(self)
        binning_enabled.add_observer(self)
        bin_size.add_observer(self)
        padding_enabled.add_observer(self)
        pad_size.add_observer(self)

    def get_crop_size(self) -> int:
        det_size = self._detector_size.get_value()
        if self._crop_enabled.get_value():
            return Interval[int](1, det_size).clamp(self._crop_size.get_value())
        return det_size

    def get_safe_crop_center(self) -> int:
        """Crop center clamped so the configured crop window fits inside the detector.

        CropStep slices ``[center - radius, center + radius)`` with
        ``radius = crop_size // 2``, so valid centers satisfy
        ``radius <= center <= det_size - radius``.
        """
        radius = self.get_crop_size() // 2
        det_size = self._detector_size.get_value()
        return Interval[int](radius, det_size - radius).clamp(self._crop_center.get_value())

    def get_bin_size(self) -> int:
        if self._binning_enabled.get_value():
            return Interval[int](1, self.get_crop_size()).clamp(self._bin_size.get_value())
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

    def _update(self, observable: Observable) -> None:
        self.notify_observers()


class PatternSizer(Observable, Observer):
    def __init__(
        self, detector_settings: DetectorSettings, diffraction_settings: DiffractionSettings
    ) -> None:
        super().__init__()
        self._detector_settings = detector_settings
        self._diffraction_settings = diffraction_settings
        self._axis_x = PatternAxisSizer(
            detector_settings.width_px,
            diffraction_settings.crop_enabled,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_center_x_px,
            diffraction_settings.binning_enabled,
            diffraction_settings.bin_size_x,
            diffraction_settings.padding_enabled,
            diffraction_settings.pad_x,
        )
        self._axis_y = PatternAxisSizer(
            detector_settings.height_px,
            diffraction_settings.crop_enabled,
            diffraction_settings.crop_height_px,
            diffraction_settings.crop_center_y_px,
            diffraction_settings.binning_enabled,
            diffraction_settings.bin_size_y,
            diffraction_settings.padding_enabled,
            diffraction_settings.pad_y,
        )

        self._axis_x.add_observer(self)
        self._axis_y.add_observer(self)

        # Whole-image parameters that don't decompose per axis. Register directly so
        # get_prep_pipeline()/get_processed_*() consumers wake up on these edits.
        diffraction_settings.hflip.add_observer(self)
        diffraction_settings.vflip.add_observer(self)
        diffraction_settings.transpose.add_observer(self)
        diffraction_settings.value_lower_bound_enabled.add_observer(self)
        diffraction_settings.value_lower_bound.add_observer(self)
        diffraction_settings.value_upper_bound_enabled.add_observer(self)
        diffraction_settings.value_upper_bound.add_observer(self)

    def get_detector_extent(self) -> ImageExtent:
        return ImageExtent(
            width_px=self._detector_settings.width_px.get_value(),
            height_px=self._detector_settings.height_px.get_value(),
        )

    def _get_detector_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self._detector_settings.pixel_width_m.get_value(),
            height_m=self._detector_settings.pixel_height_m.get_value(),
        )

    def get_processed_image_extent(self) -> ImageExtent:
        return self.get_prep_pipeline().compute_output_extent(self.get_detector_extent())

    def get_processed_pixel_geometry(self) -> PixelGeometry:
        return self.get_prep_pipeline().compute_output_pixel_geometry(
            self._get_detector_pixel_geometry()
        )

    def get_prep_pipeline(self) -> DiffractionPrepPipeline:
        """Snapshot live settings as an ordered preprocessing pipeline.

        Canonical order: filter → crop → binning → padding → hflip → vflip → transpose.
        """
        steps: list[DiffractionPrepStepUnion] = []

        lower_bound = (
            self._diffraction_settings.value_lower_bound.get_value()
            if self._diffraction_settings.value_lower_bound_enabled.get_value()
            else None
        )
        upper_bound = (
            self._diffraction_settings.value_upper_bound.get_value()
            if self._diffraction_settings.value_upper_bound_enabled.get_value()
            else None
        )
        if lower_bound is not None or upper_bound is not None:
            steps.append(FilterValuesStep(lower_bound=lower_bound, upper_bound=upper_bound))

        if self._diffraction_settings.crop_enabled.get_value():
            steps.append(
                CropStep(
                    center=CropCenter(
                        self._axis_x.get_safe_crop_center(),
                        self._axis_y.get_safe_crop_center(),
                    ),
                    extent=ImageExtent(
                        width_px=self._axis_x.get_crop_size(),
                        height_px=self._axis_y.get_crop_size(),
                    ),
                )
            )

        if self._diffraction_settings.binning_enabled.get_value():
            self._axis_x.validate_bin_size()
            self._axis_y.validate_bin_size()
            steps.append(
                BinningStep(
                    bin_size_x=self._axis_x.get_bin_size(),
                    bin_size_y=self._axis_y.get_bin_size(),
                )
            )

        if self._diffraction_settings.padding_enabled.get_value():
            steps.append(
                PaddingStep(
                    pad_x=self._axis_x.get_pad_size(),
                    pad_y=self._axis_y.get_pad_size(),
                )
            )

        if self._diffraction_settings.hflip.get_value():
            steps.append(HorizontalFlipStep())

        if self._diffraction_settings.vflip.get_value():
            steps.append(VerticalFlipStep())

        if self._diffraction_settings.transpose.get_value():
            steps.append(TransposeStep())

        return DiffractionPrepPipeline(steps=tuple(steps))

    def _update(self, observable: Observable) -> None:
        self.notify_observers()
