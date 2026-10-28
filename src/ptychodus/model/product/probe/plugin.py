"""The one builder that carries any registered probe generator."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeVar

import numpy
from pydantic import BaseModel

from ptychodus.api.observer import Observable
from ptychodus.api.parameters import Parameter
from ptychodus.api.plugins import Plugin, PluginChooser
from ptychodus.api.probe import ProbeEnsemble, ProbeExtent, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    KirkpatrickBaezMirrorPreset,
    ProbeGenerator,
)

from ...declared import ParameterDescription, describe, fields_of, restore
from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings

PresetT = TypeVar('PresetT', bound=BaseModel)


def _read_size_fields(described: Sequence[ParameterDescription]) -> Mapping[str, str]:
    """Map each declared size field to the :class:`ProbeExtent` quantity it says it is.

    A field opts in with ``json_schema_extra={'probe_extent': <ProbeExtent property>}``,
    and the extent is measured at the field named ``defocus_distance_m``, so a
    declaration with any size field must also declare that one.

    Raises:
        ValueError: for a quantity :class:`ProbeExtent` does not have, or size fields
            without a defocus to measure them at. Both are faults in the declaration.
    """
    size_fields: dict[str, str] = {}

    for description in described:
        quantity = description.extra.get('probe_extent')

        if quantity is None:
            continue

        if not isinstance(getattr(ProbeExtent, quantity, None), property):
            raise ValueError(f'"{description.name}" names no ProbeExtent quantity {quantity!r}.')

        size_fields[description.name] = quantity

    if size_fields and not any(d.name == 'defocus_distance_m' for d in described):
        raise ValueError('Size fields are measured at a defocus, but none is declared.')

    return size_fields


class PluginProbeBuilder(ProbeEnsembleBuilder):
    """Holds the live values of one generator's declared options.

    One class for every probe model, because a generator states what it accepts rather
    than hard-coding the parameters that carry it. Its parameters are copies of the
    ones the settings group holds, keyed by the name the generator declared rather than
    by the name the settings file uses, so the names a caller supplies are the names in
    the declaration.

    Every source of option values other than the settings is owned here rather than by
    the generator: a size measured from the diffraction data fills the fields declared
    as sizes, and a preset fills the fields it names.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        name: str,
        generator: ProbeGenerator,
        parameters: Mapping[str, Parameter[Any]],
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        kb_mirror_chooser: PluginChooser[KirkpatrickBaezMirrorPreset],
    ) -> None:
        # Before the base class adds parameters, since a change to any of them reaches
        # _update, which reads it.
        self._is_assigning = False
        super().__init__(rng, settings, name)
        self._settings = settings
        self._generator = generator
        self._settings_parameters = parameters
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser
        self._kb_mirror_chooser = kb_mirror_chooser
        self._defaults = generator.get_default_options()
        self._described: Sequence[ParameterDescription] = describe(self._defaults, expand_num=True)
        self._size_fields = _read_size_fields(self._described)

        for description in self._described:
            self._add_parameter(description.name, parameters[description.key].copy())

    def describe_parameters(self) -> Sequence[ParameterDescription]:
        """What this builder's generator declared, in declaration order.

        Paired with :meth:`parameters`, which holds the live value of each under the
        same declared name, this is everything needed to put an editor on screen
        without knowing which probe model it came from.
        """
        return self._described

    def copy(self) -> PluginProbeBuilder:
        builder = PluginProbeBuilder(
            self._rng,
            self._settings,
            self.get_name(),
            self._generator,
            self._settings_parameters,
            self._fresnel_zone_plate_chooser,
            self._kb_mirror_chooser,
        )

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def reads_diffraction_data(self) -> bool:
        return self._generator.reads_diffraction_data

    @property
    def estimates_size(self) -> bool:
        """Whether the generator declares any field as a size the data can measure."""
        return bool(self._size_fields)

    def can_estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        return self.estimates_size and geometry_provider.get_assembled_data() is not None

    def estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        if not self.estimates_size:
            return False

        defocus_distance_m = self.parameters()['defocus_distance_m'].get_value()
        extent = self._estimate_extent(geometry_provider, defocus_distance_m)

        if extent is None:
            return False

        self._assign(
            {name: getattr(extent, quantity) for name, quantity in self._size_fields.items()}
        )
        return True

    def _fitting(self, chooser: PluginChooser[PresetT]) -> Iterator[Plugin[PresetT]]:
        """The presets in *chooser* whose every field this generator declares.

        Read off each preset itself, so a preset registered after this builder was made
        is offered all the same, and a chooser with nothing registered offers nothing.
        """
        declared = {description.name for description in self._described}

        for plugin in chooser:
            if set(fields_of(type(plugin.strategy))) <= declared:
                yield plugin

    def labels_for_presets(self) -> Iterator[str]:
        """The names of the presets whose fields this generator declares, in menu order."""
        for zone_plate in self._fitting(self._fresnel_zone_plate_chooser):
            yield zone_plate.display_name

        for mirrors in self._fitting(self._kb_mirror_chooser):
            yield mirrors.display_name

    def apply_presets(self, label: str) -> None:
        """Fill the fields the preset *label* names, leaving every other field alone.

        *label* matches a simple or display name regardless of case, as a chooser's own
        lookup does. Looked up rather than selected, so applying a preset leaves the
        chooser's selection where it was.
        """
        wanted = label.casefold()
        presets: list[BaseModel] = [
            plugin.strategy
            for plugin in self._fitting(self._fresnel_zone_plate_chooser)
            if wanted in (plugin.simple_name.casefold(), plugin.display_name.casefold())
        ]
        presets.extend(
            plugin.strategy
            for plugin in self._fitting(self._kb_mirror_chooser)
            if wanted in (plugin.simple_name.casefold(), plugin.display_name.casefold())
        )

        if not presets:
            raise KeyError(f'No preset "{label}" for probe builder "{self.get_name()}".')

        self._assign(presets[0].model_dump())

    def _assign(self, values: Mapping[str, Any]) -> None:
        """Set several declared fields at once, announcing the change a single time.

        Each parameter still tells its own observers, so an editor showing it follows
        along. This builder holds its own announcement until every field is set,
        because each would rebuild the probe, and an intermediate combination of the
        fields need not describe a probe that builds at all.
        """
        parameters = self.parameters()
        before = {name: parameters[name].get_value() for name in values}
        self._is_assigning = True

        try:
            for name, value in values.items():
                parameters[name].set_value(value)
        finally:
            self._is_assigning = False

        if any(parameters[name].get_value() != before[name] for name in values):
            self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if not self._is_assigning:
            super()._update(observable)

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        held = {
            description.key: self.parameters()[description.name] for description in self._described
        }
        options = restore(self._defaults, self._described, held)
        return self._rescale_to_photon_count(
            self._generator.generate(options, geometry_provider), geometry_provider
        )
