from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProcessingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Processing')
        self._group.add_observer(self)

        self.algorithm = self._group.create_string_parameter('Algorithm', 'pty-chi_LSQML')
        # Device identifiers, comma separated; empty means every device the
        # process can see. Identifiers rather than indices, so the value still
        # names the same hardware on a host with a different number of cards.
        self.devices = self._group.create_string_parameter('Devices', '')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
