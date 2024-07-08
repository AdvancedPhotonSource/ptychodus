from __future__ import annotations
from collections.abc import Iterator

import numpy
import numpy.typing

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, Probe, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from .builder import ProbeBuilder
from .settings import ProbeSettings


class KirkpatrickBaezMirrorProbeBuilder(ProbeBuilder):  # FIXME

    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__('kirkpatrick_baez_mirror')
        self._settings = settings
