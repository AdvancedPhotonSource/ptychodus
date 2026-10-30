from ptychodus.api.plugins import PluginList
from ptychodus.api.visualize import ScalarTransformation
from ptychodus.model.parameters import Parameter
from ptychodus.model.plugins import PluginChooser, PluginChooserParameter


class ScalarTransformationParameter(PluginChooserParameter[ScalarTransformation]):
    def __init__(self) -> None:
        catalog = PluginList[ScalarTransformation]()

        for transformation in ScalarTransformation:
            catalog.register_plugin(
                transformation,
                simple_name=transformation.simple_name,
                display_name=transformation.display_name,
            )

        super().__init__(PluginChooser(catalog))
        self.set_value('Identity')

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.set_value(self.get_value())
        return parameter
