from collections.abc import Iterator

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import ScalarTransformation


class ScalarTransformationParameter(Parameter[str], Observer):
    def __init__(self) -> None:
        super().__init__()
        self._chooser = PluginChooser[ScalarTransformation]()
        self._chooser.register_plugin(
            ScalarTransformation.IDENTITY,
            display_name='Identity',
        )
        self._chooser.register_plugin(
            ScalarTransformation.SQRT,
            simple_name='sqrt',
            display_name='Square Root',
        )
        self._chooser.register_plugin(
            ScalarTransformation.LOG2,
            simple_name='log2',
            display_name='Logarithm (Base 2)',
        )
        self._chooser.register_plugin(
            ScalarTransformation.LOG,
            simple_name='ln',
            display_name='Natural Logarithm',
        )
        self._chooser.register_plugin(
            ScalarTransformation.LOG10,
            simple_name='log10',
            display_name='Logarithm (Base 10)',
        )
        self.set_value('Identity')
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
        parameter = ScalarTransformationParameter()
        parameter.set_value(self.get_value())
        return parameter

    def get_strategy(self) -> ScalarTransformation:
        return self._chooser.get_current_plugin().strategy

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
