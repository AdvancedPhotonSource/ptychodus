from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
import matplotlib
import numpy

from ...api.geometry import Interval
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import *


class MappedColorizer(Colorizer):

    def __init__(self, arrayComponent: VisualizationArrayComponent, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation],
                 variantChooser: PluginChooser[Colormap]) -> None:
        super().__init__(arrayComponent, displayRange, transformChooser)
        self._variantChooser = variantChooser
        self._variantChooser.addObserver(self)

    @classmethod
    def createColorizerList(
            cls, array: VisualizationArray, displayRange: DisplayRange,
            transformChooser: PluginChooser[ScalarTransformation]) -> list[Colorizer]:
        cyclicColormapEntries: list[PluginEntry[Colormap]] = list()
        acyclicColormapEntries: list[PluginEntry[Colormap]] = list()

        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        cyclicColormapNames = ['hsv', 'twilight', 'twilight_shifted']

        for name, cmap in matplotlib.colormaps.items():
            entry = PluginEntry[Colormap](simpleName=name, displayName=name, strategy=cmap)

            if name in cyclicColormapNames:
                cyclicColormapEntries.append(entry)
            else:
                acyclicColormapEntries.append(entry)

        cyclicColormapChooser = PluginChooser[Colormap].createFromList(cyclicColormapEntries)
        cyclicColormapChooser.setFromSimpleName('hsv')

        acyclicColormapChooser = PluginChooser[Colormap].createFromList(acyclicColormapEntries)
        acyclicColormapChooser.setFromSimpleName('viridis')

        amplitude = cls(AmplitudeArrayComponent(array), displayRange, transformChooser,
                        acyclicColormapChooser)
        phase = cls(PhaseArrayComponent(array), displayRange, transformChooser,
                    cyclicColormapChooser)
        phaseUnwrapped = cls(UnwrappedPhaseArrayComponent(array), displayRange, transformChooser,
                             acyclicColormapChooser)
        real = cls(RealArrayComponent(array), displayRange, transformChooser,
                   acyclicColormapChooser)
        imag = cls(ImaginaryArrayComponent(array), displayRange, transformChooser,
                   acyclicColormapChooser)

        return [amplitude, phase, phaseUnwrapped, real, imag]

    def getVariantNameList(self) -> list[str]:
        return self._variantChooser.getDisplayNameList()

    def getVariantName(self) -> str:
        return self._variantChooser.getCurrentDisplayName()

    def setVariantByName(self, name: str) -> None:
        self._variantChooser.setFromDisplayName(name)

    def getDataRange(self) -> Interval[Decimal]:
        values = self._arrayComponent()
        lower = Decimal(repr(values.min()))
        upper = Decimal(repr(values.max()))
        return Interval[Decimal](lower, upper)

    def __call__(self) -> RealArrayType:
        if self._displayRange.getUpper() <= self._displayRange.getLower():
            shape = self._arrayComponent().shape
            return numpy.zeros((*shape, 4))

        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)
        cmap = self._variantChooser.getCurrentStrategy()
        scalarMappable = ScalarMappable(norm, cmap)
        return scalarMappable.to_rgba(self._arrayComponent())

    def update(self, observable: Observable) -> None:
        if observable is self._variantChooser:
            self.notifyObservers()
        else:
            super().update(observable)
