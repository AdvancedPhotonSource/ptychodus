from ptychodus.api.diffraction import BeamCenter, CropRegion
from ptychodus.api.preprocess.diffraction import (
    BinningStep,
    DiffractionPrepPipeline,
    DiffractionPrepPlan,
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
    """Reusable factory bound to a DiffractionSettings; each get_plan call
    reads live settings and derives a (read_region, pipeline) pair. Cropping is
    hoisted to load time so readers that support partial reads (HDF5) only fetch
    the crop rectangle from disk; the returned pipeline never re-crops.

    Stateless on its own instance -- the per-call mutable state (the steps list) is
    a local variable. Same builder can be reused across many get_plan calls.
    """

    def __init__(self, settings: DiffractionSettings) -> None:
        self._settings = settings

    def get_plan(self, detector_extent: ImageExtent | None = None) -> DiffractionPrepPlan:
        """Snapshot live diffraction settings as a load-time crop plus a pipeline.

        When ``detector_extent`` is ``None``, axis clamping degrades gracefully -- the
        plan can still be constructed but the crop region and bin bounds may not
        match a real detector. Callers that will feed real patterns through the
        plan must pass an extent.

        Pipeline canonical order after the load-time crop:
        filter → binning → upsample → padding → hflip → vflip → transpose.
        """
        s = self._settings
        steps: list[DiffractionPrepStepUnion] = []

        # Value filter: extent-neutral, no clamping. Commutes with the load-time
        # crop, so keeping it in the residual pipeline (post-crop) is equivalent
        # to filtering the full frames.
        lower = s.value_lower_bound.get_value() if s.value_lower_bound_enabled.get_value() else None
        upper = s.value_upper_bound.get_value() if s.value_upper_bound_enabled.get_value() else None
        if lower is not None or upper is not None:
            steps.append(FilterValuesStep(lower_bound=lower, upper_bound=upper))

        # Crop hoisted to load time as read_region -- passed to
        # DiffractionArray.get_patterns() so HDF5 readers slice at the file
        # level. Clamped against the detector extent when known.
        read_region: CropRegion | None = None
        if s.crop_enabled.get_value():
            region = CropRegion.from_center_extent(
                BeamCenter(
                    x_px=s.beam_center_x_px.get_value(),
                    y_px=s.beam_center_y_px.get_value(),
                ),
                ImageExtent(
                    width_px=max(1, s.crop_width_px.get_value()),
                    height_px=max(1, s.crop_height_px.get_value()),
                ),
            )
            if detector_extent is not None:
                region = region.clamp_to_detector_extent(detector_extent)
            read_region = region

        # Binning: clamped against the extent that patterns will have when the
        # pipeline runs -- the crop region's extent when cropping, otherwise the
        # full detector extent when known.
        if s.binning_enabled.get_value():
            if read_region is not None:
                pre_w: int | None = read_region.width_px
                pre_h: int | None = read_region.height_px
            elif detector_extent is not None:
                pre_w = detector_extent.width_px
                pre_h = detector_extent.height_px
            else:
                pre_w = None
                pre_h = None
            bin_x = _clamp_size(s.bin_size_x.get_value(), pre_w)
            bin_y = _clamp_size(s.bin_size_y.get_value(), pre_h)
            if pre_w is not None and pre_h is not None and (pre_w % bin_x or pre_h % bin_y):
                raise ValueError(
                    f'Invalid binning size! (input=({pre_w}, {pre_h}), bin=({bin_x}, {bin_y}))'
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

        return DiffractionPrepPlan(
            read_region=read_region,
            pipeline=DiffractionPrepPipeline(steps=tuple(steps)),
        )
