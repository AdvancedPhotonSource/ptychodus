from ptychodus.api.diffraction import CropCenter
from ptychodus.api.preprocess.diffraction import (
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
from ptychodus.api.parameters import BooleanParameter, IntegerParameter

from .settings import DiffractionSettings


class PatternAxisSizer(Observable, Observer):
    def __init__(
        self,
        crop_enabled: BooleanParameter,
        crop_size: IntegerParameter,
        crop_center: IntegerParameter,
        binning_enabled: BooleanParameter,
        bin_size: IntegerParameter,
        padding_enabled: BooleanParameter,
        pad_size: IntegerParameter,
    ) -> None:
        super().__init__()
        self._crop_enabled = crop_enabled
        self._crop_size = crop_size
        self._crop_center = crop_center
        self._binning_enabled = binning_enabled
        self._bin_size = bin_size
        self._padding_enabled = padding_enabled
        self._pad_size = pad_size

        crop_enabled.add_observer(self)
        crop_size.add_observer(self)
        crop_center.add_observer(self)
        binning_enabled.add_observer(self)
        bin_size.add_observer(self)
        padding_enabled.add_observer(self)
        pad_size.add_observer(self)

    def get_crop_size(self, detector_size: int | None) -> int:
        if self._crop_enabled.get_value():
            requested = self._crop_size.get_value()
            if detector_size is None:
                return max(1, requested)
            return Interval[int](1, detector_size).clamp(requested)
        # No crop: fall back to whatever the detector reports. Callers that need
        # a concrete extent (pipeline construction) must ensure a dataset is loaded.
        return detector_size if detector_size is not None else 0

    def get_safe_crop_center(self, detector_size: int | None) -> int:
        """Crop center clamped so the configured crop window fits inside the detector.

        CropStep slices ``[center - radius, center + radius)`` with
        ``radius = crop_size // 2``, so valid centers satisfy
        ``radius <= center <= det_size - radius``.
        """
        radius = self.get_crop_size(detector_size) // 2
        if detector_size is None:
            return max(radius, self._crop_center.get_value())
        return Interval[int](radius, detector_size - radius).clamp(self._crop_center.get_value())

    def get_bin_size(self, detector_size: int | None) -> int:
        if self._binning_enabled.get_value():
            return Interval[int](1, self.get_crop_size(detector_size)).clamp(
                self._bin_size.get_value()
            )
        return 1

    def validate_bin_size(self, detector_size: int | None) -> None:
        crop_size = self.get_crop_size(detector_size)
        bin_size = self.get_bin_size(detector_size)

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
        self,
        diffraction_settings: DiffractionSettings,
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings

        self._axis_x = PatternAxisSizer(
            diffraction_settings.crop_enabled,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_center_x_px,
            diffraction_settings.binning_enabled,
            diffraction_settings.bin_size_x,
            diffraction_settings.padding_enabled,
            diffraction_settings.pad_x,
        )
        self._axis_y = PatternAxisSizer(
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

    def get_processed_image_extent(self, detector_extent: ImageExtent | None = None) -> ImageExtent:
        if detector_extent is None:
            return ImageExtent(width_px=0, height_px=0)
        return self.get_prep_pipeline(detector_extent).compute_output_extent(detector_extent)

    def get_processed_pixel_geometry(self, raw_pixel_geometry: PixelGeometry) -> PixelGeometry:
        # Pixel geometry only depends on binning and transpose (see
        # DiffractionPrepStep.apply_to_pixel_geometry overrides); crop, filter, padding,
        # and flips are identity. Compute directly from the raw settings so this method
        # works without knowing the detector extent.
        geometry = raw_pixel_geometry
        if self._diffraction_settings.binning_enabled.get_value():
            geometry = BinningStep(
                bin_size_x=self._diffraction_settings.bin_size_x.get_value(),
                bin_size_y=self._diffraction_settings.bin_size_y.get_value(),
            ).apply_to_pixel_geometry(geometry)
        if self._diffraction_settings.transpose.get_value():
            geometry = TransposeStep().apply_to_pixel_geometry(geometry)
        return geometry

    def get_prep_pipeline(
        self, detector_extent: ImageExtent | None = None
    ) -> DiffractionPrepPipeline:
        """Snapshot live settings as an ordered preprocessing pipeline.

        When ``detector_extent`` is ``None``, axis clamping degrades gracefully — the
        pipeline can still be constructed but crop/bin bounds may not match a real
        detector. Callers that will feed real patterns through the pipeline must pass
        an extent.

        Canonical order: filter → crop → binning → padding → hflip → vflip → transpose.
        """
        det_w = detector_extent.width_px if detector_extent is not None else None
        det_h = detector_extent.height_px if detector_extent is not None else None

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
                        self._axis_x.get_safe_crop_center(det_w),
                        self._axis_y.get_safe_crop_center(det_h),
                    ),
                    extent=ImageExtent(
                        width_px=self._axis_x.get_crop_size(det_w),
                        height_px=self._axis_y.get_crop_size(det_h),
                    ),
                )
            )

        if self._diffraction_settings.binning_enabled.get_value():
            self._axis_x.validate_bin_size(det_w)
            self._axis_y.validate_bin_size(det_h)
            steps.append(
                BinningStep(
                    bin_size_x=self._axis_x.get_bin_size(det_w),
                    bin_size_y=self._axis_y.get_bin_size(det_h),
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
