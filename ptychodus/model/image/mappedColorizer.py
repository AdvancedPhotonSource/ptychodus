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
from .visarray import VisualizationArrayComponent


class MappedColorizer(Colorizer):

    def __init__(self, componentChooser: PluginChooser[VisualizationArrayComponent],
                 cyclicColormapChooser: PluginChooser[Colormap],
                 acyclicColormapChooser: PluginChooser[Colormap]) -> None:
        super().__init__('Colormap', componentChooser)
        self._cyclicColormapChooser = cyclicColormapChooser
        self._acyclicColormapChooser = acyclicColormapChooser
        self._displayRange = Interval[Decimal](Decimal(0), Decimal(1))  # FIXME update

    @classmethod
    def createInstance(
            cls, componentChooser: PluginChooser[VisualizationArrayComponent]) -> MappedColorizer:
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

        colorizer = cls(componentChooser, cyclicColormapChooser, acyclicColormapChooser)
        cyclicColormapChooser.addObserver(colorizer)
        acyclicColormapChooser.addObserver(colorizer)
        return colorizer

    @property
    def _colormapChooser(self) -> PluginChooser[Colormap]:
        return self._cyclicColormapChooser if self._component.isCyclic \
                else self._acyclicColormapChooser

    def getColormapList(self) -> list[str]:
        return self._colormapChooser.getDisplayNameList()

    def getColormap(self) -> str:
        return self._colormapChooser.getCurrentDisplayName()

    def setColormap(self, name: str) -> None:
        self._colormapChooser.setFromDisplayName(name)

    def __call__(self) -> RealArrayType:
        norm = Normalize(vmin=self._displayRange.lower, vmax=self._displayRange.upper, clip=False)
        cmap = self._colormapChooser.getCurrentStrategy()
        scalarMappable = ScalarMappable(norm, cmap)
        return scalarMappable.to_rgba(self._component())

    def update(self, observable: Observable) -> None:
        if observable is self._cyclicColormapChooser:
            self.notifyObservers()
        elif observable is self._acyclicColormapChooser:
            self.notifyObservers()
        else:
            super().update(observable)
