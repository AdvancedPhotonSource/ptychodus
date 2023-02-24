from __future__ import annotations
from collections.abc import Callable
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
from .visarray import VisualizationArray


class MappedColorizer(Colorizer):

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation], name: str,
                 arrayComponent: Callable[[], RealArrayType],
                 variantChooser: PluginChooser[Colormap]) -> None:
        super().__init__(array, displayRange, transformChooser)
        self._name = name
        self._arrayComponent = arrayComponent
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

        amplitude = cls(array, displayRange, transformChooser, 'Amplitude', array.getAmplitude,
                        acyclicColormapChooser)
        phase = cls(array, displayRange, transformChooser, 'Phase', array.getPhaseInRadians,
                    cyclicColormapChooser)
        phaseUnwrapped = cls(array, displayRange, transformChooser, 'Phase (Unwrapped)',
                             array.getPhaseUnwrappedInRadians, acyclicColormapChooser)
        real = cls(array, displayRange, transformChooser, 'Real', array.getRealPart,
                   acyclicColormapChooser)
        imag = cls(array, displayRange, transformChooser, 'Imaginary', array.getImaginaryPart,
                   acyclicColormapChooser)

        return [amplitude, phase, phaseUnwrapped, real, imag]

    @property
    def name(self) -> str:
        return self._name

    def getVariantNameList(self) -> list[str]:
        return self._variantChooser.getDisplayNameList()

    def getVariantName(self) -> str:
        return self._variantChooser.getCurrentDisplayName()

    def setVariantByName(self, name: str) -> None:
        self._variantChooser.setFromDisplayName(name)

    def getDataArray(self) -> RealArrayType:
        transform = self._transformChooser.getCurrentStrategy()
        values = self._arrayComponent()
        return transform(values)

    def __call__(self) -> RealArrayType:
        if self._displayRange.getUpper() <= self._displayRange.getLower():
            shape = self._arrayComponent().shape
            return numpy.zeros((*shape, 4))

        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)

        cmap = self._variantChooser.getCurrentStrategy()
        scalarMappable = ScalarMappable(norm, cmap)

        return scalarMappable.to_rgba(self.getDataArray())

    def update(self, observable: Observable) -> None:
        if observable is self._variantChooser:
            self.notifyObservers()
        else:
            super().update(observable)
