"""The one builder that carries any registered scan pattern."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy

from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.simulate.probe_positions import ProbePositionsGenerator

from ...declared import ParameterDescription, describe, restore
from ...parameters import Parameter
from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class PluginProbePositionsBuilder(ProbePositionsBuilder):
    """Holds the live values of one generator's declared options.

    One class for every pattern, because a generator states what it accepts
    rather than hard-coding the parameters that carry it. What used to be a
    subclass per pattern, each repeating the same copy-from-settings and
    copy-for-duplication, is now this.

    Its parameters are copies of the ones the settings group holds, keyed by
    the name the generator declared rather than by the name the settings file
    uses, so the names a caller supplies are the names in the declaration.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        name: str,
        generator: ProbePositionsGenerator,
        parameters: Mapping[str, Parameter[Any]],
    ) -> None:
        super().__init__(rng, settings, name)
        self._settings = settings
        self._generator = generator
        self._settings_parameters = parameters
        self._defaults = generator.get_default_options()
        self._described: Sequence[ParameterDescription] = describe(self._defaults, expand_num=True)

        for description in self._described:
            self._add_parameter(description.name, parameters[description.key].copy())

    def describe_parameters(self) -> Sequence[ParameterDescription]:
        """What this builder's generator declared, in declaration order.

        Paired with :meth:`parameters`, which holds the live value of each
        under the same declared name, this is everything needed to put an
        editor on screen without knowing which pattern it came from.
        """
        return self._described

    def copy(self) -> PluginProbePositionsBuilder:
        builder = PluginProbePositionsBuilder(
            self._rng, self._settings, self.get_name(), self._generator, self._settings_parameters
        )

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self) -> Sequence[ProbePosition]:
        held = {
            description.key: self.parameters()[description.name] for description in self._described
        }
        return self._generator.generate(restore(self._defaults, self._described, held))
