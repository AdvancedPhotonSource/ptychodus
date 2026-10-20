from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProcessingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Processing')
        self._group.add_observer(self)

        # Library before Algorithm because a settings file is applied in the
        # order parameters were created, so the library is selected before the
        # algorithm is resolved against it. Belt and braces only: the algorithm
        # re-resolves from the persisted name whenever the library changes, so
        # a file carrying the two keys in either order still lands correctly.
        # The simple name, which is what a chooser writes back and so what
        # a saved file holds; the display name resolves too.
        self.library = self._group.create_string_parameter('Library', 'PtyChi')
        self.algorithm = self._group.create_string_parameter('Algorithm', 'LSQML')
        # Device identifiers, comma separated; empty means every device the
        # process can see. Identifiers rather than indices, so the value still
        # names the same hardware on a host with a different number of cards.
        self.devices = self._group.create_string_parameter('Devices', '')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
