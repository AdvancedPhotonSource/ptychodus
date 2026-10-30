from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...declared import ParameterDescription, declare_in
from ...observer import Observable, Observer
from ...parameters import Parameter
from ...settings import SettingsRegistry


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Object')
        self._group.add_observer(self)

        self.builder = self._group.create_string_parameter('Builder', 'random')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/object.npy'))
        self.file_type = self._group.create_string_parameter('FileType', 'NPY')

        self.object_layer_spacing_m = self._group.create_real_sequence_parameter(
            'ObjectLayerSpacingInMeters', []
        )

        self.extra_padding_x = self._group.create_integer_parameter('ExtraPaddingX', 1, minimum=0)
        self.extra_padding_y = self._group.create_integer_parameter('ExtraPaddingY', 1, minimum=0)

    def declare(self, described: Sequence[ParameterDescription]) -> Mapping[str, Parameter[Any]]:
        """Create the parameters an object generator declares, and return them by key.

        The generator-specific keys of this section are not written out above, because
        which of them exist is settled by which generators registered. They land in the
        same group all the same, so one section still holds everything an object is
        configured by.
        """
        return declare_in(self._group, described)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
