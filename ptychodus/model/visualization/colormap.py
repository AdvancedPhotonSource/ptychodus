from collections.abc import Iterator
from typing import Final

from matplotlib.colors import Colormap
import matplotlib

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser


class ColormapParameter(Parameter[str], Observer):
    # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    CYCLIC_COLORMAPS: Final[tuple[str, ...]] = ("hsv", "twilight", "twilight_shifted")

    def __init__(self, *, isCyclic: bool) -> None:
        super().__init__()
        self._chooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            isCyclicColormap = name in ColormapParameter.CYCLIC_COLORMAPS

            if isCyclic == isCyclicColormap:
                self._chooser.registerPlugin(cmap, displayName=name)

        self.setValue("hsv" if isCyclic else "gray")
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for name in self._chooser.getDisplayNameList():
            yield name

    def getValue(self) -> str:
        return self._chooser.currentPlugin.displayName

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def getPlugin(self) -> Colormap:
        return self._chooser.currentPlugin.strategy

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
