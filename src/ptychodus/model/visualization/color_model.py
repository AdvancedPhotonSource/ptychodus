from ptychodus.api.parameters import Parameter
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
from ptychodus.api.visualize import CylindricalColorModel


class CylindricalColorModelParameter(PluginChooserParameter[CylindricalColorModel]):
    def __init__(self) -> None:
        chooser = PluginChooser[CylindricalColorModel]()

        for model in CylindricalColorModel:
            chooser.register_plugin(
                model, simple_name=model.simple_name, display_name=model.display_name
            )

        super().__init__(chooser)
        self.set_value('HSV-V')

    def copy(self) -> Parameter[str]:
        parameter = CylindricalColorModelParameter()
        parameter.set_value(self.get_value())
        return parameter
