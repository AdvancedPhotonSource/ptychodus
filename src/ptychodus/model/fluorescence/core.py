import logging

from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    FluorescenceEnhancer,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductAPI
from ..task_manager import TaskManager
from ..visualization import VisualizationEngine
from .api import FluorescenceAPI
from .monitor import FluorescenceTaskMonitor
from .ptychozoon import PtychozoonFluorescenceEnhancer
from .settings import FluorescenceSettings
from .two_step import TwoStepFluorescenceEnhancer
from .vspi import VSPIFluorescenceEnhancer

logger = logging.getLogger(__name__)


class FluorescenceCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        upscaling_strategy_chooser: PluginChooser[UpscalingStrategy],
        deconvolution_strategy_chooser: PluginChooser[DeconvolutionStrategy],
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self._settings = FluorescenceSettings(settings_registry)
        self.two_step_enhancer = TwoStepFluorescenceEnhancer(
            self._settings, upscaling_strategy_chooser, deconvolution_strategy_chooser
        )
        self.vspi_enhancer = VSPIFluorescenceEnhancer(self._settings)

        self.enhancer_chooser = PluginChooser[FluorescenceEnhancer]()
        self.enhancer_chooser.register_plugin(
            self.two_step_enhancer,
            simple_name=TwoStepFluorescenceEnhancer.SIMPLE_NAME,
            display_name=TwoStepFluorescenceEnhancer.DISPLAY_NAME,
        )
        self.enhancer_chooser.register_plugin(
            self.vspi_enhancer,
            simple_name=VSPIFluorescenceEnhancer.SIMPLE_NAME,
            display_name=VSPIFluorescenceEnhancer.DISPLAY_NAME,
        )

        # The GPU VSPI enhancer is optional: register it only when ptychozoon is
        # installed so it silently disappears otherwise. Register it last so its
        # combo-box entry and stacked GUI page stay aligned by index. Importing
        # the top-level package is cheap and does not pull in CuPy (that happens
        # only inside the spawned subprocess), so probing it here is safe.
        self.ptychozoon_enhancer: PtychozoonFluorescenceEnhancer | None = None

        try:
            import ptychozoon  # noqa: F401
        except ModuleNotFoundError:
            logger.info('ptychozoon not found.')
        else:
            self.ptychozoon_enhancer = PtychozoonFluorescenceEnhancer(self._settings)
            self.enhancer_chooser.register_plugin(
                self.ptychozoon_enhancer,
                simple_name=PtychozoonFluorescenceEnhancer.SIMPLE_NAME,
                display_name=PtychozoonFluorescenceEnhancer.DISPLAY_NAME,
            )

        self.enhancer_chooser.synchronize_with_parameter(self._settings.algorithm)

        file_reader_chooser.synchronize_with_parameter(self._settings.file_type)
        file_writer_chooser.set_current_plugin(self._settings.file_type.get_value())

        self.visualization_engine = VisualizationEngine(is_complex=False)
        self.task_monitor = FluorescenceTaskMonitor(task_manager)
        self.fluorescence_api = FluorescenceAPI(
            task_manager,
            self._settings,
            product_api,
            self.enhancer_chooser,
            self.task_monitor,
            file_reader_chooser,
            file_writer_chooser,
        )

        logging.getLogger('ptychodus.model.fluorescence').addHandler(
            self.task_monitor.get_log_handler()
        )
