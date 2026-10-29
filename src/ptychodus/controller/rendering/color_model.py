from ptychodus.api.plugins import PluginList
from ptychodus.api.visualize import CylindricalColorModel
from ptychodus.model.parameters import Parameter
from ptychodus.model.plugins import PluginChooser, PluginChooserParameter


class CylindricalColorModelParameter(PluginChooserParameter[CylindricalColorModel]):
    def __init__(self) -> None:
        catalog = PluginList[CylindricalColorModel]()

        for model in CylindricalColorModel:
            catalog.register_plugin(
                model, simple_name=model.simple_name, display_name=model.display_name
            )

        super().__init__(PluginChooser(catalog))
        self.set_value('HSV-V')

    def copy(self) -> Parameter[str]:
        parameter = CylindricalColorModelParameter()
        parameter.set_value(self.get_value())
        return parameter
