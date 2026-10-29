from pathlib import Path
from typing import Final

from ..observer import Observable, Observer
from ..settings import SettingsRegistry


# What the enhancers are filed under in the parameter store, and the prefix of the
# settings groups their options persist in.
ENHANCER_LIBRARY: Final = 'Fluorescence'


class FluorescenceSettings(Observable, Observer):
    """Which dataset to enhance, and which enhancer to run on it.

    What each enhancer accepts is not written out here: it declares its own options,
    and they persist in a group of that enhancer's own.
    """

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Fluorescence')
        self._group.add_observer(self)

        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/dataset.h5'))
        self.file_type = self._group.create_string_parameter('FileType', 'XRF-Maps')
        self.algorithm = self._group.create_string_parameter('Algorithm', 'vspi')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
