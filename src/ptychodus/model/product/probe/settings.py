from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...declared import ParameterDescription, declare_in
from ...observer import Observable, Observer
from ...parameters import Parameter
from ...settings import SettingsRegistry


class ProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Probe')
        self._group.add_observer(self)

        self.builder = self._group.create_string_parameter('Builder', 'disk')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/probe.npy'))
        self.file_type = self._group.create_string_parameter('FileType', 'NPY')

        self.num_incoherent_modes = self._group.create_integer_parameter(
            'NumberOfIncoherentModes', 1, minimum=1
        )
        self.orthogonalize_incoherent_modes = self._group.create_boolean_parameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.incoherent_mode_decay_type = self._group.create_string_parameter(
            'IncoherentModeDecayType', 'Polynomial'
        )
        self.incoherent_mode_decay_ratio = self._group.create_real_parameter(
            'IncoherentModeDecayRatio', 1.0, minimum=0.0, maximum=1.0
        )
        self.incoherent_mode_strategy = self._group.create_string_parameter(
            'IncoherentModeStrategy', 'MomentPolynomial'
        )
        self.moment_polynomial_damping_width = self._group.create_real_parameter(
            'MomentPolynomialDampingWidth', 1.0, minimum=0.0
        )
        self.gaussian_schell_beam_width_m = self._group.create_real_parameter(
            'GaussianSchellBeamWidthInMeters', 1e-6, minimum=0.0
        )
        self.gaussian_schell_beam_height_m = self._group.create_real_parameter(
            'GaussianSchellBeamHeightInMeters', 1e-6, minimum=0.0
        )
        self.gaussian_schell_coherence_width_m = self._group.create_real_parameter(
            'GaussianSchellCoherenceWidthInMeters', 1e-6, minimum=0.0
        )
        self.gaussian_schell_coherence_height_m = self._group.create_real_parameter(
            'GaussianSchellCoherenceHeightInMeters', 1e-6, minimum=0.0
        )
        self.num_coherent_modes = self._group.create_integer_parameter(
            'NumberOfCoherentModes', 1, minimum=1
        )
        self.opr_weight_policy = self._group.create_string_parameter('OPRWeightPolicy', 'Average')

    def declare(self, described: Sequence[ParameterDescription]) -> Mapping[str, Parameter[Any]]:
        """Create the parameters a probe generator declares, and return them by key.

        The generator-specific keys of this section are not written out above, because
        which of them exist is settled by which generators registered. They land in the
        same group all the same, so one section still holds everything a probe is
        configured by.
        """
        return declare_in(self._group, described)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
