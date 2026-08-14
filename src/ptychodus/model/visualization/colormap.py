from matplotlib.colors import Colormap

from ptychodus.api.parameters import Parameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
from ptychodus.api.visualize import (
    cyclic_colormap_names,
    get_colormap_by_name,
    linear_colormap_names,
)


class ColormapParameter(PluginChooserParameter[Colormap]):
    def __init__(self, *, is_cyclic: bool) -> None:
        self._is_cyclic = is_cyclic
        chooser = PluginChooser[Colormap]()
        cmap_name_it = cyclic_colormap_names() if is_cyclic else linear_colormap_names()

        for name in sorted(cmap_name_it):
            cmap = get_colormap_by_name(name)
            chooser.register_plugin(cmap, display_name=name)

        super().__init__(chooser)
        self.set_value('colorwheel' if is_cyclic else 'gray')

    def copy(self) -> Parameter[str]:
        parameter = ColormapParameter(is_cyclic=self._is_cyclic)
        parameter.set_value(self.get_value())
        return parameter
