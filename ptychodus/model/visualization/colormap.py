from collections.abc import Iterator
from typing import override, Final

from matplotlib.colors import Colormap
import matplotlib

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser


class ColormapParameter(Parameter[str], Observer):
    # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
    CYCLIC_COLORMAPS: Final[tuple[str, ...]] = ('hsv', 'twilight', 'twilight_shifted')

    def __init__(self, *, isCyclic: bool) -> None:
        super().__init__('')
        self._chooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            isCyclicColormap = (name in ColormapParameter.CYCLIC_COLORMAPS)

            if isCyclic == isCyclicColormap:
                self._chooser.registerPlugin(cmap, displayName=name)

        self.setValue('hsv' if isCyclic else 'gray')
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for name in self._chooser.getDisplayNameList():
            yield name

    @override
    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)
        super().setValue(self._chooser.currentPlugin.displayName, notify=notify)

    def getPlugin(self) -> Colormap:
        return self._chooser.currentPlugin.strategy

    @override
    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            super().setValue(self._chooser.currentPlugin.displayName)
