import numpy

from ...api.probe import ProbeArrayType
from ..data import Detector
from .settings import ProbeSettings


class SuperGaussianProbeInitializer:

    def __init__(self, detector: Detector, settings: ProbeSettings) -> None:
        # TODO VERIFY perhaps this should use pixel size at object plane?
        self._detector = detector
        self._settings = settings

    def __call__(self) -> ProbeArrayType:
        width_px = self._settings.probeSize.value
        height_px = width_px

        Y_px, X_px = numpy.ogrid[:height_px, :width_px]
        X_m = (X_px - width_px / 2) * float(self._detector.getPixelSizeXInMeters())
        Y_m = (Y_px - height_px / 2) * float(self._detector.getPixelSizeYInMeters())
        R_m = numpy.hypot(X_m, Y_m)

        Z = (R_m - float(self._settings.sgAnnularRadiusInMeters.value)) \
                / float(self._settings.sgProbeWidthInMeters.value)
        ZP = numpy.power(Z, 2 * float(self._settings.sgOrderParameter.value))

        return numpy.exp(-ZP / 2) + 0j
