from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
import matplotlib

from ...api.geometry import Interval
from ...api.image import RealArrayType
from ...api.observer import Observable
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import VisualizationArrayComponent


class MappedColorizer(Colorizer):

    def __init__(self, componentChooser: PluginChooser[VisualizationArrayComponent],
                 displayRange: DisplayRange, cyclicColormapChooser: PluginChooser[Colormap],
                 acyclicColormapChooser: PluginChooser[Colormap]) -> None:
        super().__init__('Colormap', componentChooser, displayRange)
        self._cyclicColormapChooser = cyclicColormapChooser
        self._acyclicColormapChooser = acyclicColormapChooser

    @classmethod
    def createInstance(cls, componentChooser: PluginChooser[VisualizationArrayComponent],
                       displayRange: DisplayRange) -> MappedColorizer:
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

        colorizer = cls(componentChooser, displayRange, cyclicColormapChooser,
                        acyclicColormapChooser)
        cyclicColormapChooser.addObserver(colorizer)
        acyclicColormapChooser.addObserver(colorizer)
        return colorizer

    def getVariantList(self) -> list[str]:
        return self._colormapChooser.getDisplayNameList()

    def getVariant(self) -> str:
        return self._colormapChooser.getCurrentDisplayName()

    def setVariant(self, name: str) -> None:
        self._colormapChooser.setFromDisplayName(name)

    def getDataRange(self) -> Interval[Decimal]:
        values = self._arrayComponent()
        lower = Decimal(repr(values.min()))
        upper = Decimal(repr(values.max()))
        return Interval[Decimal](lower, upper)

    def __call__(self) -> RealArrayType:
        # FIXME crash when display range reversed
        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)
        cmap = self._colormapChooser.getCurrentStrategy()
        scalarMappable = ScalarMappable(norm, cmap)
        return scalarMappable.to_rgba(self._arrayComponent())

    def update(self, observable: Observable) -> None:
        if observable is self._cyclicColormapChooser:
            self.notifyObservers()
        elif observable is self._acyclicColormapChooser:
            self.notifyObservers()
        else:
            super().update(observable)
