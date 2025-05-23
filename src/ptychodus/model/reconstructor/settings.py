from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Reconstructor')
        self._group.add_observer(self)

        self.algorithm = self._group.create_string_parameter('Algorithm', 'pty-chi/LSQML')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
