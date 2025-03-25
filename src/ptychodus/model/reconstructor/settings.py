from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('Reconstructor')
        self._settings_group.add_observer(self)

        self.algorithm = self._settings_group.create_string_parameter(
            'Algorithm', 'Tike/lstsq_grad'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()
