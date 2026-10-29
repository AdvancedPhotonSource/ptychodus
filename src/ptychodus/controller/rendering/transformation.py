from ptychodus.api.plugins import PluginList
from ptychodus.api.visualize import ScalarTransformation
from ptychodus.model.parameters import Parameter
from ptychodus.model.plugins import PluginChooser, PluginChooserParameter


class ScalarTransformationParameter(PluginChooserParameter[ScalarTransformation]):
    def __init__(self) -> None:
        catalog = PluginList[ScalarTransformation]()
        catalog.register_plugin(
            ScalarTransformation.IDENTITY,
            display_name='Identity',
        )
        catalog.register_plugin(
            ScalarTransformation.SQRT,
            simple_name='sqrt',
            display_name='Square Root',
        )
        catalog.register_plugin(
            ScalarTransformation.LOG2,
            simple_name='log2',
            display_name='Logarithm (Base 2)',
        )
        catalog.register_plugin(
            ScalarTransformation.LOG,
            simple_name='ln',
            display_name='Natural Logarithm',
        )
        catalog.register_plugin(
            ScalarTransformation.LOG10,
            simple_name='log10',
            display_name='Logarithm (Base 10)',
        )

        super().__init__(PluginChooser(catalog))
        self.set_value('Identity')

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.set_value(self.get_value())
        return parameter
