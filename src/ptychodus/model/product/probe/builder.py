from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable, Mapping
import logging

import numpy

from ptychodus.api.parameters import ParameterGroup
from ptychodus.api.simulate.probe import (
    GaussianSchellStrategy,
    IncoherentModeStrategy,
    ProbeMomentPolynomialStrategy,
    ProbeModeDecayType,
    RandomPhaseRampStrategy,
    generate_coherent_probe_modes,
    generate_incoherent_probe_modes,
    rescale_probe_intensity,
)
from ptychodus.api.probe import (
    OPRWeightPolicy,
    Probe,
    ProbeSequence,
    ProbeFileReader,
    ProbeGeometryProvider,
    conform_opr_weights,
    resample_probe_sequence,
)

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeSequenceBuilder(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings, name: str) -> None:
        super().__init__()
        self._rng = rng

        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

        self.num_incoherent_modes = settings.num_incoherent_modes.copy()
        self._add_parameter('num_incoherent_modes', self.num_incoherent_modes)

        self.orthogonalize_incoherent_modes = settings.orthogonalize_incoherent_modes.copy()
        self._add_parameter('orthogonalize_incoherent_modes', self.orthogonalize_incoherent_modes)

        self.incoherent_mode_decay_type = settings.incoherent_mode_decay_type.copy()
        self._add_parameter('incoherent_mode_decay_type', self.incoherent_mode_decay_type)

        self.incoherent_mode_decay_ratio = settings.incoherent_mode_decay_ratio.copy()
        self._add_parameter('incoherent_mode_decay_ratio', self.incoherent_mode_decay_ratio)

        self.incoherent_mode_strategy = settings.incoherent_mode_strategy.copy()
        self._add_parameter('incoherent_mode_strategy', self.incoherent_mode_strategy)

        self.moment_polynomial_damping_width = settings.moment_polynomial_damping_width.copy()
        self._add_parameter('moment_polynomial_damping_width', self.moment_polynomial_damping_width)

        self.gaussian_schell_beam_width_m = settings.gaussian_schell_beam_width_m.copy()
        self._add_parameter('gaussian_schell_beam_width_m', self.gaussian_schell_beam_width_m)

        self.gaussian_schell_beam_height_m = settings.gaussian_schell_beam_height_m.copy()
        self._add_parameter('gaussian_schell_beam_height_m', self.gaussian_schell_beam_height_m)

        self.gaussian_schell_coherence_width_m = settings.gaussian_schell_coherence_width_m.copy()
        self._add_parameter(
            'gaussian_schell_coherence_width_m', self.gaussian_schell_coherence_width_m
        )

        self.gaussian_schell_coherence_height_m = settings.gaussian_schell_coherence_height_m.copy()
        self._add_parameter(
            'gaussian_schell_coherence_height_m', self.gaussian_schell_coherence_height_m
        )

        self.num_coherent_modes = settings.num_coherent_modes.copy()
        self._add_parameter('num_coherent_modes', self.num_coherent_modes)

        self.opr_weight_policy = settings.opr_weight_policy.copy()
        self._add_parameter('opr_weight_policy', self.opr_weight_policy)

        # Set for a single ingest, by a caller that has already chosen; see
        # `_conform_opr_weights`. Deliberately not a parameter: a one-off choice
        # belongs to that ingest, not to the stored settings.
        self._opr_weight_policy_override: OPRWeightPolicy | None = None

    def get_name(self) -> str:
        return self._name.get_value()

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ProbeSequenceBuilder:
        pass

    @abstractmethod
    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        """Return the raw, unconditioned probe.

        Implementations must NOT expand the incoherent or coherent (OPR) modes;
        `build` owns the conditioning pipeline. Generative implementations should
        return `self._rescale_to_photon_count(probe, geometry_provider)`, which
        normalizes the intensity and widens the 3-D `Probe` they generated to the
        4-D `ProbeSequence` this method returns.
        """
        pass

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        """Return the conditioned probe: incoherent modes, then coherent (OPR) modes.

        Overriding this method is reserved for builders whose probe is already
        conditioned; see `FromMemoryProbeBuilder`. Every builder that generates or
        ingests a raw probe must leave it alone and implement `_build_raw`
        instead.
        """
        return self._condition_probe(self._build_raw(geometry_provider), geometry_provider)

    def _rescale_to_photon_count(
        self, probe: Probe, geometry_provider: ProbeGeometryProvider
    ) -> ProbeSequence:
        """Normalize a freshly generated probe to the expected photon count and
        widen it to the `ProbeSequence` that `_build_raw` returns.

        Only generative builders call this. A probe read from file already carries
        the intensity it was reconstructed at; rescaling it would silently decouple
        it from a matching from-file object, because the data constrains the
        product of probe and object, not either one alone.
        """
        rescaled = rescale_probe_intensity(probe, geometry_provider.probe_photon_count)
        return ProbeSequence.from_probe(rescaled)

    def _get_imode_decay(self) -> tuple[ProbeModeDecayType, float]:
        """Resolve the incoherent-mode decay settings, tolerating values the api rejects.

        The settings permit a zero ratio and any string at all, while the api strategies
        raise on a non-positive ratio; a bad stored value should leave the probe with a
        single occupied mode, not stop the build.
        """
        imode_decay_ratio = self.incoherent_mode_decay_ratio.get_value()
        imode_decay_type_text = self.incoherent_mode_decay_type.get_value()
        imode_decay_type = ProbeModeDecayType.NONE

        if imode_decay_ratio > 0.0:
            try:
                imode_decay_type = ProbeModeDecayType[imode_decay_type_text.upper()]
            except KeyError:
                logger.debug(f'Unknown probe mode decay type "{imode_decay_type_text}"')

        return imode_decay_type, imode_decay_ratio

    def _create_moment_polynomial_strategy(self) -> IncoherentModeStrategy:
        decay_type, decay_ratio = self._get_imode_decay()
        return ProbeMomentPolynomialStrategy(
            decay_type=decay_type,
            decay_ratio=decay_ratio,
            damping_width=self.moment_polynomial_damping_width.get_value(),
        )

    def _create_random_phase_ramp_strategy(self) -> IncoherentModeStrategy:
        decay_type, decay_ratio = self._get_imode_decay()
        return RandomPhaseRampStrategy(self._rng, decay_type=decay_type, decay_ratio=decay_ratio)

    def _create_gaussian_schell_strategy(self) -> IncoherentModeStrategy:
        beam_size_x_m = self.gaussian_schell_beam_width_m.get_value()
        coherence_length_x_m = self.gaussian_schell_coherence_width_m.get_value()
        beam_size_y_m = self.gaussian_schell_beam_height_m.get_value()
        coherence_length_y_m = self.gaussian_schell_coherence_height_m.get_value()
        widths = (beam_size_x_m, coherence_length_x_m, beam_size_y_m, coherence_length_y_m)

        # The settings permit zero where the model requires a positive width, and a
        # source of no size describes nothing to build modes from.
        if any(width <= 0.0 for width in widths):
            logger.debug('Gaussian-Schell source has a non-positive width')
            return self._create_moment_polynomial_strategy()

        return GaussianSchellStrategy(
            beam_size_x_m=beam_size_x_m,
            coherence_length_x_m=coherence_length_x_m,
            beam_size_y_m=beam_size_y_m,
            coherence_length_y_m=coherence_length_y_m,
        )

    def _create_imode_strategy(self) -> IncoherentModeStrategy:
        """Build the strategy the settings name, falling back on anything unrecognized.

        The settings are deliberately looser than the api strategies: the names are free
        strings and the decay ratio may be zero, both of which the strategies reject. A
        stored value they would refuse should leave a usable probe, not stop the build.
        """
        name = self.incoherent_mode_strategy.get_value()
        namecf = name.casefold()

        for candidate, factory in _IMODE_STRATEGY_FACTORIES.items():
            if candidate.casefold() == namecf:
                return factory(self)

        logger.debug(f'Unknown incoherent mode strategy "{name}"')
        return next(iter(_IMODE_STRATEGY_FACTORIES.values()))(self)

    def _get_opr_weight_policy(self) -> OPRWeightPolicy:
        """Resolve the OPR weight policy setting, tolerating a name the api does not know.

        The setting is a free string, as the incoherent-mode settings are, so a stored
        value with no matching policy should leave a usable probe rather than stop the
        build.
        """
        text = self.opr_weight_policy.get_value()

        try:
            return OPRWeightPolicy[text.upper()]
        except KeyError:
            logger.debug(f'Unknown OPR weight policy "{text}"')
            return OPRWeightPolicy.AVERAGE

    def _conform_opr_weights(
        self, probe_seq: ProbeSequence, geometry_provider: ProbeGeometryProvider
    ) -> ProbeSequence:
        """Reconcile an ingested probe's OPR weights with this run's probe positions.

        The weights carry one row per probe position, so a probe solved on another scan
        cannot initialize this one until those rows are resolved. The policy setting
        names how, and is consulted only when the counts actually disagree -- a warm
        start whose weights already fit comes through untouched.

        An override supplied for a single ingest takes precedence and applies even when
        the counts agree, which is what lets a caller ask for a fresh basis on a scan of
        the same size.
        """
        weights = probe_seq.get_opr_weights_or_none()

        if weights is None:
            return probe_seq

        num_positions = geometry_provider.num_scan_points

        if num_positions < 1:
            # No probe positions bound yet; the observer chain re-runs the build.
            return probe_seq

        policy = self._opr_weight_policy_override

        if policy is None:
            if weights.shape[0] == num_positions:
                return probe_seq

            policy = self._get_opr_weight_policy()

        logger.info(
            f'Probe carries OPR weights for {weights.shape[0]} probe position(s)'
            f' and this run has {num_positions}; applying {policy.name}.'
        )
        return conform_opr_weights(self._rng, probe_seq, num_positions, policy)

    def _condition_probe(
        self, probe_seq: ProbeSequence, geometry_provider: ProbeGeometryProvider
    ) -> ProbeSequence:
        """Expand the probe to the requested mode structure, never shrinking it.

        Every step is expand-only, so the pipeline is idempotent: conditioning an
        already-conditioned probe returns it unchanged. That matters because the
        generators in `ptychodus.api.simulate.probe` are not safe to re-apply.
        `generate_incoherent_probe_modes` re-orthogonalizes and renormalizes every
        incoherent mode to the decay profile, and `generate_coherent_probe_modes`
        fills the whole output with fresh Gaussian noise, keeps only coherent mode
        zero of its input, and regenerates the OPR weights from scratch. Run
        either one on a converged probe and the reconstruction is gone.

        The guards are data-driven rather than provenance-driven, so they live
        here rather than in the ingesting subclasses. Generative builders always
        emit a single coherent, single incoherent mode, which makes every guard
        inert on that path.

        The OPR weights are reconciled first, so a `COLLAPSE` or `DISCARD` policy leaves
        a single-coherent-mode probe that then goes through the expand-only pipeline
        below like any other -- which is what lets discarding a stale basis and
        requesting a new mode count compose into a basis sized to this run.
        """
        probe_seq = self._conform_opr_weights(probe_seq, geometry_provider)
        num_imodes_requested = self.num_incoherent_modes.get_value()
        num_cmodes_requested = self.num_coherent_modes.get_value()

        if probe_seq.num_coherent_modes > 1 or probe_seq.get_opr_weights_or_none() is not None:
            # There is no non-destructive way to extend a solved OPR basis, so
            # leave the whole mode structure alone.
            if (
                num_cmodes_requested > probe_seq.num_coherent_modes
                or num_imodes_requested > probe_seq.num_incoherent_modes
            ):
                logger.info(
                    'Probe already has an OPR mode basis'
                    f' ({probe_seq.num_coherent_modes} coherent,'
                    f' {probe_seq.num_incoherent_modes} incoherent);'
                    ' leaving its mode structure unchanged.'
                )

            return probe_seq

        probe = probe_seq.get_probe_no_opr()
        num_imodes_actual = probe.num_incoherent_modes

        if num_imodes_actual < num_imodes_requested:
            probe = generate_incoherent_probe_modes(
                probe,
                num_imodes_requested,
                strategy=self._create_imode_strategy(),
                orthogonalize=self.orthogonalize_incoherent_modes.get_value(),
            )
        elif num_imodes_actual > num_imodes_requested:
            logger.info(
                f'Probe has {num_imodes_actual} incoherent mode(s);'
                f' keeping them rather than discarding down to {num_imodes_requested}.'
            )

        if num_cmodes_requested > 1:
            probe_seq = generate_coherent_probe_modes(
                self._rng,
                probe,
                num_cmodes=num_cmodes_requested,
                num_diffraction_patterns=geometry_provider.num_scan_points,
            )
        else:
            probe_seq = ProbeSequence.from_probe(probe)

        logger.debug(f'Conditioned probe {probe_seq.get_array().shape=}')
        return probe_seq


# The single source of truth for the selectable strategies. A chooser offers these keys
# and the builder resolves against this same mapping, so a name can never be offered
# without a way to build it, nor a strategy be buildable but never offered.
_IMODE_STRATEGY_FACTORIES: Mapping[
    str, Callable[[ProbeSequenceBuilder], IncoherentModeStrategy]
] = {
    'MomentPolynomial': ProbeSequenceBuilder._create_moment_polynomial_strategy,
    'RandomPhaseRamp': ProbeSequenceBuilder._create_random_phase_ramp_strategy,
    'GaussianSchell': ProbeSequenceBuilder._create_gaussian_schell_strategy,
}

INCOHERENT_MODE_STRATEGY_NAMES = tuple(_IMODE_STRATEGY_FACTORIES)
"""Selectable incoherent-mode strategies, in the order a chooser should offer them.

The first entry is what an unrecognized stored name falls back to.
"""


class FromMemoryProbeBuilder(ProbeSequenceBuilder):
    """A probe that has already been conditioned.

    Two things produce these. Reconstruction output, which `ProcessingTaskMonitor`
    re-assigns to the output product item on every reconstructor iteration (see
    `model/processing/monitor.py`), and products loaded from HDF5/NPZ, whose probe
    was conditioned before it was saved. In both cases the incoherent and coherent
    (OPR) mode structure is already what the reconstructor solved for, so
    re-running the mode generators would be catastrophic rather than merely lossy:
    `generate_incoherent_probe_modes` re-orthogonalizes and renormalizes every
    incoherent mode to the decay profile, and `generate_coherent_probe_modes`
    replaces every coherent mode but the first with fresh Gaussian noise and
    regenerates the OPR weights from scratch -- once per iteration. `build`
    therefore deliberately bypasses the conditioning pipeline.

    The expand-only guards in `_condition_probe` would in fact catch most of this
    on their own, but the bypass is explicit so that the invariant does not depend
    on them.

    Reconciling the OPR weights is the one step that still runs, because a probe
    copied onto a scan of a different size is unusable until its weight rows are
    resolved. It runs no generator and is a strict no-op whenever the row count
    already matches, which it always does for reconstruction output.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        probe: ProbeSequence,
    ) -> None:
        super().__init__(rng, settings, 'from_memory')
        self._settings = settings
        self._probe = probe.copy()

    def copy(self) -> FromMemoryProbeBuilder:
        builder = FromMemoryProbeBuilder(self._rng, self._settings, self._probe)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe_geometry = geometry_provider.get_probe_geometry()

        try:
            pixel_geometry = self._probe.get_pixel_geometry()
        except ValueError:
            pixel_geometry = probe_geometry.get_pixel_geometry()

        # TODO regrid probe as needed based on probe geometry from file/provider
        return self._conform_opr_weights(
            ProbeSequence(
                self._probe.get_array(),
                self._probe.get_opr_weights_or_none(),
                pixel_geometry,
            ),
            geometry_provider,
        )

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        return self._build_raw(geometry_provider)


class FromFileProbeBuilder(ProbeSequenceBuilder):
    """A probe read from file, conditioned on the way in.

    Unlike `FromMemoryProbeBuilder` this is an ingest path, so the mode settings
    do apply -- warm-starting a mixed-state run from a single-mode probe file is a
    real workflow, and before the conditioning pipeline existed those settings
    were silently ignored here. `_condition_probe` is expand-only, so a file that
    already carries more modes, or an OPR basis, keeps what it has.

    A file that records its own pixel size is reconciled to the run's, since a
    probe sampled by another geometry is an illumination of the wrong physical
    size. None of the probe formats currently record one, so today this only ever
    fires for a reader that supplies it.

    An OPR basis read from a product solved on another scan carries weights sized for
    that scan. `_condition_probe` reconciles them against this run's probe positions
    before anything else, under the policy setting or under an override passed here for
    a single ingest.

    The photon-count rescale is deliberately not applied; see
    `ProbeSequenceBuilder._rescale_to_photon_count`.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        file_reader: ProbeFileReader,
        *,
        opr_weight_policy: OPRWeightPolicy | None = None,
    ) -> None:
        super().__init__(rng, settings, 'from_file')
        self._settings = settings
        self._file_reader = file_reader

        if opr_weight_policy is not None:
            self._opr_weight_policy_override = opr_weight_policy
            self.opr_weight_policy.set_value(opr_weight_policy.name)

        self.file_path = settings.file_path.copy()
        self._add_parameter('file_path', self.file_path)

        self.file_type = settings.file_type.copy()
        self._add_parameter('file_type', self.file_type)

    def copy(self) -> FromFileProbeBuilder:
        builder = FromFileProbeBuilder(
            self._rng,
            self._settings,
            self._file_reader,
            opr_weight_policy=self._opr_weight_policy_override,
        )

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            probe_from_file = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        probe_geometry = geometry_provider.get_probe_geometry()

        try:
            pixel_geometry = probe_from_file.get_pixel_geometry()
        except ValueError:
            # The format records no pixel size, so there is nothing to reconcile and the
            # array is taken to be sampled the way this run is.
            pixel_geometry = probe_geometry.get_pixel_geometry()

        return resample_probe_sequence(
            ProbeSequence(
                probe_from_file.get_array(),
                probe_from_file.get_opr_weights_or_none(),
                pixel_geometry,
            ),
            probe_geometry,
        )
