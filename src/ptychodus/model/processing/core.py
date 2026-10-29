from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstruct import (
    ReconstructorLibrary,
    SubprocessReconstructor,
    TrainingDataFileWriter,
)
from ptychodus.api.settings import SettingsRegistry

from ..parameters import PluginChooserParameter
from ..product import ProductAPI
from ..task_manager import TaskManager
from .api import ProcessingAPI, ProcessingAlgorithmParameter
from . import _ptychi_sections, _ptycho_fm_sections
from ..devices import DeviceSelection
from .monitor import ProcessingTaskMonitor
from .parameter_store import OptionsBranch, OptionsSection, ProcessingParameterStore
from .settings import ProcessingSettings

# A backend's section table, given one reconstructor's name and the options it
# declares. Both arguments because a backend may persist one reconstructor
# differently from another: pty-chi files each algorithm's additions to the
# shared branches under a group of its own, and which fields those are is
# answered by the declaration rather than by the name.
SectionTable = Callable[[str, Any], Sequence[OptionsSection]]


def _regardless(
    sections: Callable[[], Sequence[OptionsSection]],
) -> SectionTable:
    """Adapt a table that is the same for every reconstructor a backend offers."""

    def table(_name: str, _options: Any) -> Sequence[OptionsSection]:
        return sections()

    return table


# Which settings groups back which backend. **Transitional**, with the flat
# layout it serves: a backend states what it accepts, and this says where each
# option is persisted until the Qt panels stop needing a flat one. A backend
# absent here registers no groups and runs on its declared defaults.
_SECTIONS_BY_LIBRARY: Mapping[str, SectionTable] = {
    'PtychoFM': _regardless(_ptycho_fm_sections.sections),
    'pty-chi': _ptychi_sections.sections_for,
}

# Which groups back the backends whose declaration ptychodus owns. No policy and
# no overlay: the fields carry their own bounds and display text, the field name
# is the key, and this says only which branch each group covers. A backend that
# can be described this way belongs here rather than above.
_BRANCHES_BY_LIBRARY: Mapping[str, Sequence[OptionsBranch]] = {
    'PtychoPINN': (
        OptionsBranch('PtychoPINNData', root_path='data'),
        OptionsBranch('PtychoPINNModel', root_path='model'),
        OptionsBranch('PtychoPINNTraining', root_path='training'),
        OptionsBranch('PtychoPINNInference', root_path='inference'),
    ),
    'cdtools': (
        OptionsBranch('CDToolsOptimizer', root_path='optimizer'),
        OptionsBranch('CDToolsSchedule', root_path='schedule.stage1', prefix='Stage1'),
        OptionsBranch('CDToolsSchedule', root_path='schedule.stage2', prefix='Stage2'),
        OptionsBranch('CDToolsSchedule', root_path='schedule.stage3', prefix='Stage3'),
        OptionsBranch('CDToolsRefinement', root_path='refinement'),
        OptionsBranch('CDToolsFancyPtycho', root_path='model'),
    ),
}


def _no_sections(_name: str, _options: Any) -> Sequence[OptionsSection]:
    return ()


class ProcessingCore:
    """Which reconstructor backends are in play, and the live settings behind them.

    Takes the chooser the backends registered themselves with rather than a
    list drawn from it, so that this is the one place that answers which
    backends exist. Anything else that needs to know -- the panels, above all
    -- asks here instead of walking the plugin registry a second time.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        reconstructor_libraries: PluginChooser[ReconstructorLibrary],
        training_data_file_writers: PluginChooser[TrainingDataFileWriter],
    ) -> None:
        self._settings = ProcessingSettings(settings_registry)
        self._task_monitor = ProcessingTaskMonitor(task_manager)

        # Settled here, once. Plugins have all registered by the time this is
        # built, so the roster cannot grow afterwards; one registered later
        # would reach no settings group and no panel.
        # A library does not name itself; the name travels beside it from
        # the registration that gave it one.
        libraries = [(plugin.display_name, plugin.strategy) for plugin in reconstructor_libraries]
        self._library_names = tuple(name for name, _ in libraries)

        # Ahead of the first thing that asks a library for its reconstructors,
        # since that is when a library builds them and says what it found.
        # Iterating the chooser is not that: it yields what was registered.
        for _, library in libraries:
            library.get_logger().addHandler(self._task_monitor.get_log_handler())

        # Two selections, one behind the other: the chooser's own current
        # plugin is the library, and the algorithm is resolved against
        # whichever library that is.
        self.library_parameter = PluginChooserParameter(
            reconstructor_libraries, self._settings.library
        )
        self.algorithm_parameter = ProcessingAlgorithmParameter(
            self._settings.algorithm,
            self.library_parameter,
            default=self._settings.default_algorithm,
        )
        self.parameter_store = ProcessingParameterStore(settings_registry)
        self.device_selection = DeviceSelection(self._settings.devices)

        # Every reconstructor is filed under the same key the algorithm selection
        # uses, so a panel and the options handed to a run cannot disagree.
        # Every library, not only the selected one: a backend reached for the
        # first time when someone changes the selection would build itself --
        # and import torch -- wherever that selection was made.
        algorithms_by_library: dict[str, list[str]] = {}

        for name, library in libraries:
            sections = _SECTIONS_BY_LIBRARY.get(name, _no_sections)
            offered = algorithms_by_library.setdefault(name, [])

            branches = _BRANCHES_BY_LIBRARY.get(name)

            for reconstructor in library:
                offered.append(reconstructor.name)

                if branches is None:
                    self.parameter_store.register(
                        name,
                        reconstructor,
                        sections(reconstructor.name, reconstructor.get_default_options()),
                    )
                else:
                    self.parameter_store.register_declared(
                        name, reconstructor.name, reconstructor.get_default_options(), branches
                    )

                # Placement is ptychodus's to decide, so it is attached here
                # rather than passed through every backend's factory.
                if isinstance(reconstructor, SubprocessReconstructor):
                    reconstructor.device_environment = self.device_selection.environment

        self.processing_api = ProcessingAPI(
            task_manager,
            product_api,
            self.library_parameter,
            self.algorithm_parameter,
            algorithms_by_library,
            self._task_monitor,
            self.parameter_store,
            training_data_file_writers,
        )

    def library_names(self) -> Sequence[str]:
        """Every backend in play, named as it names itself, in the chooser's order.

        A method returning a fixed sequence rather than an attribute, so a
        caller cannot edit its copy into disagreement with what was actually
        registered.
        """
        return self._library_names
