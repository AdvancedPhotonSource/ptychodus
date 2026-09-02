from ptychodus.api.diffraction import CropCenter, CropRegion
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

from .settings import DiffractionSettings


def _clamp_size(requested: int, upper: int | None) -> int:
    """Clamp a requested bin size to ``[1, upper]``.

    When ``upper is None`` (no detector loaded yet), degrade to ``max(1, requested)``
    so pipeline construction still succeeds for GUI preview.
    """
    if upper is None:
        return max(1, requested)
    return Interval[int](1, upper).clamp(requested)


class PrepPipelineBuilder:
    """Reusable factory bound to a DiffractionSettings; each get_pipeline call
    reads live settings and folds them through partial-pipeline compute_output_extent
    calls to derive step-output-dependent clamps (crop-bounded bin size, etc.).

    Stateless on its own instance -- the per-call mutable state (the steps list) is
    a local variable. Same builder can be reused across many get_pipeline calls.
    """

    def __init__(self, settings: DiffractionSettings) -> None:
        self._settings = settings

    def get_pipeline(self, detector_extent: ImageExtent | None = None) -> DiffractionPrepPipeline:
        """Snapshot live diffraction settings as an ordered preprocessing pipeline.

        When ``detector_extent`` is ``None``, axis clamping degrades gracefully -- the
        pipeline can still be constructed but crop/bin bounds may not match a real
        detector. Callers that will feed real patterns through the pipeline must pass
        an extent.

        Canonical order: filter → crop → binning → upsample → padding → hflip → vflip → transpose.
        """
        s = self._settings
        steps: list[DiffractionPrepStepUnion] = []

        # Value filter: extent-neutral, no clamping.
        lower = s.value_lower_bound.get_value() if s.value_lower_bound_enabled.get_value() else None
        upper = s.value_upper_bound.get_value() if s.value_upper_bound_enabled.get_value() else None
        if lower is not None or upper is not None:
            steps.append(FilterValuesStep(lower_bound=lower, upper_bound=upper))

        # Crop: clamped against the pre-crop extent -- computed from the partial
        # pipeline via compute_output_extent (single source of truth).
        if s.crop_enabled.get_value():
            pre = (
                DiffractionPrepPipeline(steps=tuple(steps)).compute_output_extent(detector_extent)
                if detector_extent is not None
                else None
            )
            region = CropRegion.from_center_extent(
                CropCenter(
                    x_px=s.crop_center_x_px.get_value(),
                    y_px=s.crop_center_y_px.get_value(),
                ),
                ImageExtent(
                    width_px=max(1, s.crop_width_px.get_value()),
                    height_px=max(1, s.crop_height_px.get_value()),
                ),
            )
            if pre is not None:
                region = region.clamp_to_detector_extent(pre)
            steps.append(CropStep(region=region))

        # Binning: clamped against post-crop extent, again from compute_output_extent.
        if s.binning_enabled.get_value():
            pre = (
                DiffractionPrepPipeline(steps=tuple(steps)).compute_output_extent(detector_extent)
                if detector_extent is not None
                else None
            )
            pre_w = pre.width_px if pre is not None else None
            pre_h = pre.height_px if pre is not None else None
            bin_x = _clamp_size(s.bin_size_x.get_value(), pre_w)
            bin_y = _clamp_size(s.bin_size_y.get_value(), pre_h)
            if pre is not None and (pre.width_px % bin_x or pre.height_px % bin_y):
                raise ValueError(
                    f'Invalid binning size! (input=({pre.width_px}, {pre.height_px}), '
                    f'bin=({bin_x}, {bin_y}))'
                )
            steps.append(BinningStep(bin_size_x=bin_x, bin_size_y=bin_y))

        # Upsample / padding / flips / transpose: no clamping needed.
        if s.upsample_enabled.get_value():
            factor = s.upsample_factor.get_value()
            if factor > 1:
                steps.append(UpsampleStep(factor=factor))
        if s.padding_enabled.get_value():
            steps.append(PaddingStep(pad_x=s.pad_x.get_value(), pad_y=s.pad_y.get_value()))
        if s.hflip.get_value():
            steps.append(HorizontalFlipStep())
        if s.vflip.get_value():
            steps.append(VerticalFlipStep())
        if s.transpose.get_value():
            steps.append(TransposeStep())

        return DiffractionPrepPipeline(steps=tuple(steps))
