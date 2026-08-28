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
    UpsampleStep,
    VerticalFlipStep,
)
from ptychodus.api.geometry import ImageExtent, Interval
from ptychodus.api.parameters import BooleanParameter, IntegerParameter

from .settings import DiffractionSettings


class _PatternAxisSizer:
    """Per-axis clamping for crop / bin / pad parameters.

    Kept as a helper so both axes share the same logic without repeating five method
    bodies inline. Stateless beyond its parameter refs; instantiated fresh on each
    build_prep_pipeline() call.
    """

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
        self._crop_enabled = crop_enabled
        self._crop_size = crop_size
        self._crop_center = crop_center
        self._binning_enabled = binning_enabled
        self._bin_size = bin_size
        self._padding_enabled = padding_enabled
        self._pad_size = pad_size

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


def build_prep_pipeline(
    settings: DiffractionSettings,
    detector_extent: ImageExtent | None = None,
) -> DiffractionPrepPipeline:
    """Snapshot live diffraction settings as an ordered preprocessing pipeline.

    When ``detector_extent`` is ``None``, axis clamping degrades gracefully — the
    pipeline can still be constructed but crop/bin bounds may not match a real
    detector. Callers that will feed real patterns through the pipeline must pass
    an extent.

    Canonical order: filter → crop → binning → upsample → padding → hflip → vflip → transpose.
    """
    axis_x = _PatternAxisSizer(
        settings.crop_enabled,
        settings.crop_width_px,
        settings.crop_center_x_px,
        settings.binning_enabled,
        settings.bin_size_x,
        settings.padding_enabled,
        settings.pad_x,
    )
    axis_y = _PatternAxisSizer(
        settings.crop_enabled,
        settings.crop_height_px,
        settings.crop_center_y_px,
        settings.binning_enabled,
        settings.bin_size_y,
        settings.padding_enabled,
        settings.pad_y,
    )

    det_w = detector_extent.width_px if detector_extent is not None else None
    det_h = detector_extent.height_px if detector_extent is not None else None

    steps: list[DiffractionPrepStepUnion] = []

    lower_bound = (
        settings.value_lower_bound.get_value()
        if settings.value_lower_bound_enabled.get_value()
        else None
    )
    upper_bound = (
        settings.value_upper_bound.get_value()
        if settings.value_upper_bound_enabled.get_value()
        else None
    )
    if lower_bound is not None or upper_bound is not None:
        steps.append(FilterValuesStep(lower_bound=lower_bound, upper_bound=upper_bound))

    if settings.crop_enabled.get_value():
        steps.append(
            CropStep(
                center=CropCenter(
                    axis_x.get_safe_crop_center(det_w),
                    axis_y.get_safe_crop_center(det_h),
                ),
                extent=ImageExtent(
                    width_px=axis_x.get_crop_size(det_w),
                    height_px=axis_y.get_crop_size(det_h),
                ),
            )
        )

    if settings.binning_enabled.get_value():
        axis_x.validate_bin_size(det_w)
        axis_y.validate_bin_size(det_h)
        steps.append(
            BinningStep(
                bin_size_x=axis_x.get_bin_size(det_w),
                bin_size_y=axis_y.get_bin_size(det_h),
            )
        )

    if settings.upsample_enabled.get_value():
        factor = settings.upsample_factor.get_value()
        if factor > 1:
            steps.append(UpsampleStep(factor=factor))

    if settings.padding_enabled.get_value():
        steps.append(
            PaddingStep(
                pad_x=axis_x.get_pad_size(),
                pad_y=axis_y.get_pad_size(),
            )
        )

    if settings.hflip.get_value():
        steps.append(HorizontalFlipStep())

    if settings.vflip.get_value():
        steps.append(VerticalFlipStep())

    if settings.transpose.get_value():
        steps.append(TransposeStep())

    return DiffractionPrepPipeline(steps=tuple(steps))
