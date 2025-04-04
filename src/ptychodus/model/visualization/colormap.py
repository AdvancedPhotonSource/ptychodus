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

    def __init__(self, *, is_cyclic: bool) -> None:
        super().__init__()
        self._is_cyclic = is_cyclic
        self._chooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            is_cyclic_colormap = name in ColormapParameter.CYCLIC_COLORMAPS

            if is_cyclic == is_cyclic_colormap:
                self._chooser.register_plugin(cmap, display_name=name)

        self.set_value('hsv' if is_cyclic else 'gray')
        self._chooser.add_observer(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def get_value(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def set_value(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def copy(self) -> Parameter[str]:
        parameter = ColormapParameter(is_cyclic=self._is_cyclic)
        parameter.set_value(self.get_value())
        return parameter

    def get_plugin(self) -> Colormap:
        return self._chooser.get_current_plugin().strategy

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
