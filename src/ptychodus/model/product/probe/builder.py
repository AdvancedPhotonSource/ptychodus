from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from enum import auto, IntEnum
import logging

import numpy

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe_gen import (
    generate_coherent_probe_modes,
    generate_incoherent_probe_modes,
    rescale_probe_intensity,
)
from ptychodus.api.probe import (
    Probe,
    ProbeSequence,
    ProbeFileReader,
    ProbeGeometryProvider,
)

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeModeDecayType(IntEnum):
    NONE = auto()
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()

    def get_weights(self, num_modes: int, decay_ratio: float) -> Sequence[float]:
        match self.value:
            case ProbeModeDecayType.EXPONENTIAL:
                b = 1.0 / decay_ratio
                return [b**-n for n in range(num_modes)]
            case ProbeModeDecayType.POLYNOMIAL:
                b = numpy.log(decay_ratio) / numpy.log(2.0)
                return [(n + 1) ** b for n in range(num_modes)]
            case _:
                return [1.0] + [0.0] * (num_modes - 1)


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

        self.num_coherent_modes = settings.num_coherent_modes.copy()
        self._add_parameter('num_coherent_modes', self.num_coherent_modes)

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
        return ProbeSequence(
            array=rescaled.get_array(),
            opr_weights=None,
            pixel_geometry=rescaled.get_pixel_geometry(),
        )

    def _get_imode_weights(self) -> Sequence[float]:
        imode_decay_ratio = self.incoherent_mode_decay_ratio.get_value()
        imode_decay_type_text = self.incoherent_mode_decay_type.get_value()
        imode_decay_type = ProbeModeDecayType.NONE

        if imode_decay_ratio > 0.0:
            try:
                imode_decay_type = ProbeModeDecayType[imode_decay_type_text.upper()]
            except KeyError:
                logger.debug(f'Unknown probe mode decay type "{imode_decay_type_text}"')

        num_imodes = self.num_incoherent_modes.get_value()
        return imode_decay_type.get_weights(num_imodes, imode_decay_ratio)

    def _condition_probe(
        self, probe_seq: ProbeSequence, geometry_provider: ProbeGeometryProvider
    ) -> ProbeSequence:
        """Expand the probe to the requested mode structure, never shrinking it.

        Every step is expand-only, so the pipeline is idempotent: conditioning an
        already-conditioned probe returns it unchanged. That matters because the
        generators in `ptychodus.api.probe_gen` are not safe to re-apply.
        `generate_incoherent_probe_modes` re-orthogonalizes and renormalizes every
        incoherent mode to the decay profile, and `generate_coherent_probe_modes`
        fills the whole output with fresh Gaussian noise, keeps only coherent mode
        zero of its input, and regenerates the OPR weights from scratch. Run
        either one on a converged probe and the reconstruction is gone.

        The guards are data-driven rather than provenance-driven, so they live
        here rather than in the ingesting subclasses. Generative builders always
        emit a single coherent, single incoherent mode, which makes every guard
        inert on that path.
        """
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
                self._rng,
                probe,
                self._get_imode_weights(),
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
            probe_seq = ProbeSequence(
                array=probe.get_array(),
                opr_weights=None,
                pixel_geometry=probe.get_pixel_geometry(),
            )

        logger.debug(f'Conditioned probe {probe_seq.get_array().shape=}')
        return probe_seq


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
        return ProbeSequence(
            self._probe.get_array(),
            self._probe.get_opr_weights_or_none(),
            pixel_geometry,
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

    The photon-count rescale is deliberately not applied; see
    `ProbeSequenceBuilder._rescale_to_photon_count`.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        file_reader: ProbeFileReader,
    ) -> None:
        super().__init__(rng, settings, 'from_file')
        self._settings = settings
        self._file_reader = file_reader

        self.file_path = settings.file_path.copy()
        self._add_parameter('file_path', self.file_path)

        self.file_type = settings.file_type.copy()
        self._add_parameter('file_type', self.file_type)

    def copy(self) -> FromFileProbeBuilder:
        builder = FromFileProbeBuilder(self._rng, self._settings, self._file_reader)

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
            pixel_geometry = probe_geometry.get_pixel_geometry()

        # TODO regrid probe as needed based on probe geometry from file/provider
        return ProbeSequence(
            probe_from_file.get_array(),
            probe_from_file.get_opr_weights_or_none(),
            pixel_geometry,
        )
