from collections.abc import Iterator

from matplotlib.colors import Colormap

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import (
    cyclic_colormap_names,
    get_colormap_by_name,
    linear_colormap_names,
)


class ColormapParameter(Parameter[str], Observer):
    def __init__(self, *, is_cyclic: bool) -> None:
        super().__init__()
        self._is_cyclic = is_cyclic
        self._chooser = PluginChooser[Colormap]()
        cmap_name_it = cyclic_colormap_names() if is_cyclic else linear_colormap_names()

        for name in sorted(cmap_name_it):
            cmap = get_colormap_by_name(name)
            self._chooser.register_plugin(cmap, display_name=name)

        self.set_value('colorwheel' if is_cyclic else 'gray')
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

    def get_strategy(self) -> Colormap:
        return self._chooser.get_current_plugin().strategy

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
