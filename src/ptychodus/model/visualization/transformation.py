from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
from ptychodus.api.visualization import ScalarTransformation


class ScalarTransformationParameter(PluginChooserParameter[ScalarTransformation]):
    def __init__(self) -> None:
        chooser = PluginChooser[ScalarTransformation]()
        chooser.register_plugin(
            ScalarTransformation.IDENTITY,
            display_name='Identity',
        )
        chooser.register_plugin(
            ScalarTransformation.SQRT,
            simple_name='sqrt',
            display_name='Square Root',
        )
        chooser.register_plugin(
            ScalarTransformation.LOG2,
            simple_name='log2',
            display_name='Logarithm (Base 2)',
        )
        chooser.register_plugin(
            ScalarTransformation.LOG,
            simple_name='ln',
            display_name='Natural Logarithm',
        )
        chooser.register_plugin(
            ScalarTransformation.LOG10,
            simple_name='log10',
            display_name='Logarithm (Base 10)',
        )

        super().__init__(chooser)
        self.set_value('Identity')

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.set_value(self.get_value())
        return parameter
