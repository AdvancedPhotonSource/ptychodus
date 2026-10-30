"""The one builder that carries any registered object generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import ObjectGenerator

from ...declared import ParameterDescription, describe, restore
from ...parameters import Parameter
from .builder import ObjectBuilder
from .settings import ObjectSettings


class PluginObjectBuilder(ObjectBuilder):
    """Holds the live values of one generator's declared options.

    One class for every object model, because a generator states what it accepts rather
    than hard-coding the parameters that carry it. Its parameters are copies of the
    ones the settings group holds, keyed by the name the generator declared rather than
    by the name the settings file uses, so the names a caller supplies are the names in
    the declaration.

    Every input other than the options is owned here rather than by the generator: the
    random draws come from the model's generator, and the probe from the product this
    builder is bound into.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        name: str,
        generator: ObjectGenerator,
        parameters: Mapping[str, Parameter[Any]],
    ) -> None:
        super().__init__(settings, name)
        self._rng = rng
        self._settings = settings
        self._generator = generator
        self._settings_parameters = parameters
        self._defaults = generator.get_default_options()
        self._described: Sequence[ParameterDescription] = describe(self._defaults, expand_num=True)

        for description in self._described:
            self._add_parameter(description.name, parameters[description.key].copy())

    def describe_parameters(self) -> Sequence[ParameterDescription]:
        """What this builder's generator declared, in declaration order.

        Paired with :meth:`parameters`, which holds the live value of each under the
        same declared name, this is everything needed to put an editor on screen
        without knowing which object model it came from.
        """
        return self._described

    def copy(self) -> PluginObjectBuilder:
        builder = PluginObjectBuilder(
            self._rng,
            self._settings,
            self.get_name(),
            self._generator,
            self._settings_parameters,
        )

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def reads_diffraction_data(self) -> bool:
        return self._generator.reads_diffraction_data

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        held = {
            description.key: self.parameters()[description.name] for description in self._described
        }
        options = restore(self._defaults, self._described, held)
        # Resolved only for a generator that reads it, so a probe on other pixels is
        # reported where it changes the object and nowhere else.
        probe = (
            self._get_probe_on_object_pixels(
                geometry_provider, consequence=f'building "{self.get_name()}" without it'
            )
            if self._generator.reads_probe
            else None
        )
        object_ = self._generator.generate(options, geometry_provider, rng=self._rng, probe=probe)
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
