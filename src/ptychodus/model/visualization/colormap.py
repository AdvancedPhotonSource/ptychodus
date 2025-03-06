from collections.abc import Iterator
from typing import Final

from matplotlib.colors import Colormap
import matplotlib

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser


class ColormapParameter(Parameter[str], Observer):
    # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    CYCLIC_COLORMAPS: Final[tuple[str, ...]] = ('hsv', 'twilight', 'twilight_shifted')

    def __init__(self, *, isCyclic: bool) -> None:
        super().__init__()
        self._isCyclic = isCyclic
        self._chooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            isCyclicColormap = name in ColormapParameter.CYCLIC_COLORMAPS

            if isCyclic == isCyclicColormap:
                self._chooser.register_plugin(cmap, display_name=name)

        self.setValue('hsv' if isCyclic else 'gray')
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def getValue(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def getValueAsString(self) -> str:
        return self.getValue()

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def copy(self) -> Parameter[str]:
        parameter = ColormapParameter(isCyclic=self._isCyclic)
        parameter.setValue(self.getValue())
        return parameter

    def getPlugin(self) -> Colormap:
        return self._chooser.get_current_plugin().strategy

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
