from collections.abc import Iterator

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import ScalarTransformation


class ScalarTransformationParameter(Parameter[str], Observer):

    def __init__(self, chooser: PluginChooser[ScalarTransformation]) -> None:
        super().__init__(chooser.currentPlugin.displayName)
        self._chooser = chooser
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for name in self._chooser.getDisplayNameList():
            yield name

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)
        super().setValue(self._chooser.currentPlugin.displayName, notify=notify)

    def getPlugin(self) -> ScalarTransformation:
        return self._chooser.currentPlugin.strategy

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            super().setValue(self._chooser.currentPlugin.displayName)
