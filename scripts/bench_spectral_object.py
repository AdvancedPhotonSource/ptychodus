#!/usr/bin/env python
"""Benchmark object initializations against pty-chi LSQML reconstructions of real data.

Each case is one PtychoPINN training directory, as written by the PtychoPINN data
generator: ``<name>.npz`` holding the paired diffraction patterns as photon counts, and
``product.h5`` holding the long LSQML reconstruction they were exported from, which is the
reference. An initialization is built on the reference's object canvas, from either the
model probe a user would start from (``model``) or the reconstructed probe (``lsqml``),
then aligned to the reference and scored:

- ``corr``: complex correlation with the reference over the scoring region, after the
  shift, scale, phase offset and phase ramp are removed;
- ``detail``: the same correlation after a high-pass filter of width ``sigma_px``, the
  probe's RMS radius, which scores structure finer than the probe and ignores the slow
  phase a few LSQML epochs settle;
- ``phase_ssim``: SSIM of the wrapped phase;
- ``hotness``: p99 / p50 of the modulus, smoothed over ``sigma_px``, over the scoring
  region; near the reference's own value is good, much larger is an estimate dominated
  by a few bright regions.

``--warm-start-epochs`` also runs pty-chi LSQML from each initialization, with the model
probe optimized, and scores the object at each listed epoch.

Example usage:

    uv run --no-sync python scripts/bench_spectral_object.py \\
        --data-root ../ptychodus-extra/PtychoPINN/out \\
        --output-dir /tmp/spectral --dataset nxs_data_ic_1_64 \\
        --method stxm_dpc --method spectral --spectral tolerance=1e-4 --label tight

Needs matplotlib for figures and the ptychi extra for warm starts. Not shipped with
the wheel.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy
from scipy.ndimage import gaussian_filter

from ptychodus.api.accumulate import (
    DiffractionMoments,
    compute_diffraction_moments,
    estimate_incident_beam,
)
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.constants import PI
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import load_product
from ptychodus.api.metrics import (
    compute_phase_pixelwise_metrics,
    compute_product_scoring_region,
    estimate_reconstruction_ambiguities,
)
from ptychodus.api.object import Object, ObjectGeometry, ObjectGeometryProvider, shift_object
from ptychodus.api.parallax import estimate_defocus_from_assembled_data
from ptychodus.api.probe import Probe, ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.product import Product
from ptychodus.api.propagate import compute_far_field_pixel_geometry
from ptychodus.api.simulate.object import (
    compute_stxm_dpc_signals,
    generate_stxm_dpc_object,
    generate_uniform_object,
)
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    generate_average_pattern_probe,
    generate_disk_probe,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
    rescale_probe_intensity,
)
from ptychodus.api.typing import ComplexArrayType, RealArrayType
from ptychodus.plugins.object.phase_retrieval import SpectralObjectGenerator

logger = logging.getLogger(Path(__file__).stem)

MAX_SHIFT_PX = 8
"""How far registration searches: the probes' centres may differ by a few pixels."""


@dataclass(frozen=True)
class ModelProbeSpec:
    """The model probe the instrument's settings describe, before any reconstruction."""

    zone_plate: FresnelZonePlate | None = None
    defocus_distance_m: float = 0.0
    disk_diameter_m: float = 1.0e-6

    def build(self, geometry: ProbeGeometry, wavelength_m: float) -> Probe:
        if self.zone_plate is None:
            return generate_disk_probe(geometry, radius_m=self.disk_diameter_m / 2.0)

        return generate_fresnel_zone_plate_probe(
            geometry,
            self.zone_plate,
            photon_wavelength_m=wavelength_m,
            defocus_distance_m=self.defocus_distance_m,
        )


_APS_ZONE_PLATE = ModelProbeSpec(
    zone_plate=FresnelZonePlate(
        zone_plate_diameter_m=180e-6,
        outermost_zone_width_m=50e-9,
        central_beamstop_diameter_m=60e-6,
    ),
    defocus_distance_m=0.8e-3,
)
_ALS_DISK = ModelProbeSpec(disk_diameter_m=1.0e-6)

MODEL_PROBES: Mapping[str, ModelProbeSpec] = {
    'als': _ALS_DISK,
    'fly001': _APS_ZONE_PLATE,
    'microLFPstack_single_exposure': _ALS_DISK,
    'microLFPstack_double_exposure': _ALS_DISK,
    'nxs_data_gold_tp_1': _APS_ZONE_PLATE,
    'nxs_data_gold_tp_2': _APS_ZONE_PLATE,
    'nxs_data_ic_1': _APS_ZONE_PLATE,
    'nxs_data_ic_2': _APS_ZONE_PLATE,
    'nxs_data_ncm_1': _APS_ZONE_PLATE,
}


@dataclass(frozen=True)
class Case:
    """One dataset: its patterns, its LSQML reference and the probes to initialize from."""

    name: str
    assembled_data: AssembledDiffractionData
    reference: Product
    probes: Mapping[str, Probe]
    mean_amplitude_transmission: float

    @property
    def wavelength_m(self) -> float:
        return self.reference.metadata.photon_wavelength_m

    @property
    def distance_m(self) -> float:
        return self.reference.metadata.detector_distance_m


def _instrument_name(case_name: str) -> str:
    return case_name.rsplit('_', 1)[0]


def _estimate_defocus_m(
    assembled_data: AssembledDiffractionData,
    reference: Product,
    probe_geometry: ProbeGeometry,
    wavelength_m: float,
) -> float:
    """The parallax defocus, as a new GUI product measures it, or zero when unresolved."""
    estimate = estimate_defocus_from_assembled_data(
        assembled_data,
        reference.probe_positions,
        reference.object_.get_geometry(),
        probe_geometry,
        photon_wavelength_m=wavelength_m,
    )

    if estimate is None or not estimate.is_resolved():
        return 0.0

    return estimate.defocus_distance_m


def load_case(data_root: Path, name: str) -> Case:
    directory = data_root / name
    reference = load_product(directory / 'product.h5')

    with numpy.load(directory / f'{name}.npz') as npz:
        patterns = numpy.asarray(npz['diff3d'], dtype=numpy.float32)

    indexes = reference.probe_positions.get_indexes()

    if len(indexes) != patterns.shape[0]:
        raise ValueError(f'{name}: {patterns.shape[0]} patterns but {len(indexes)} positions!')

    lsqml_probe = reference.probes.get_mean_probe()
    probe_pixel_geometry = lsqml_probe.get_pixel_geometry()
    height_px, width_px = patterns.shape[-2:]
    extent = ImageExtent(width_px=width_px, height_px=height_px)
    metadata = reference.metadata
    detector_pixel_geometry = compute_far_field_pixel_geometry(
        probe_pixel_geometry,
        extent,
        wavelength_m=metadata.photon_wavelength_m,
        propagation_distance_m=metadata.detector_distance_m,
    )
    assembled_data = AssembledDiffractionData(
        numpy.asarray(indexes),
        patterns,
        detector_pixel_geometry,
        numpy.zeros((height_px, width_px), dtype=bool),
    )
    moments = compute_diffraction_moments(
        assembled_data.get_indexes(), patterns, bad_pixels=assembled_data.get_bad_pixels()
    )

    if moments is None:
        raise ValueError(f'{name}: no moments!')

    beam = estimate_incident_beam(moments)
    probe_geometry = ProbeGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=probe_pixel_geometry.width_m,
        pixel_height_m=probe_pixel_geometry.height_m,
    )
    model_probe = rescale_probe_intensity(
        MODEL_PROBES[_instrument_name(name)].build(probe_geometry, metadata.photon_wavelength_m),
        beam.incident_photon_count,
    )
    defocus_m = _estimate_defocus_m(
        assembled_data, reference, probe_geometry, metadata.photon_wavelength_m
    )
    average_probe = rescale_probe_intensity(
        generate_average_pattern_probe(
            probe_geometry,
            moments.mean_pattern,
            bad_pixels=assembled_data.get_bad_pixels(),
            photon_wavelength_m=metadata.photon_wavelength_m,
            detector_distance_m=metadata.detector_distance_m,
            detector_pixel_geometry=detector_pixel_geometry,
            defocus_distance_m=defocus_m,
        ),
        beam.incident_photon_count,
    )
    logger.info('%s: parallax defocus %.4g m', name, defocus_m)
    return Case(
        name=name,
        assembled_data=assembled_data,
        reference=reference,
        probes={'model': model_probe, 'average': average_probe, 'lsqml': lsqml_probe},
        mean_amplitude_transmission=beam.mean_amplitude_transmission,
    )


def _parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    for pair in pairs:
        key, _, text = pair.partition('=')

        try:
            overrides[key] = json.loads(text)
        except json.JSONDecodeError:
            overrides[key] = text

    return overrides


def build_flat(case: Case, probe: Probe, overrides: Mapping[str, Any]) -> Object:
    return generate_uniform_object(case.reference.object_.get_geometry())


def build_stxm_dpc(case: Case, probe: Probe, overrides: Mapping[str, Any]) -> Object:
    height_px, width_px = case.assembled_data.get_patterns().shape[-2:]
    return generate_stxm_dpc_object(
        case.reference.object_.get_geometry(),
        compute_stxm_dpc_signals(case.assembled_data),
        case.reference.probe_positions,
        detector_extent=ImageExtent(width_px=width_px, height_px=height_px),
        probe=probe,
        **{'correct_shadow': False, **overrides},
    )


class CaseGeometryProvider(ObjectGeometryProvider):
    """A case's data and reference canvas, as the GUI's product would bind them."""

    def __init__(self, case: Case) -> None:
        self._case = case
        self._moments = compute_diffraction_moments(
            case.assembled_data.get_indexes(),
            case.assembled_data.get_patterns(),
            bad_pixels=case.assembled_data.get_bad_pixels(),
        )

    @property
    def photon_wavelength_m(self) -> float:
        return self._case.wavelength_m

    @property
    def mean_amplitude_transmission(self) -> float:
        return self._case.mean_amplitude_transmission

    @property
    def exit_wave_propagation_distance_m(self) -> float:
        return self._case.distance_m

    @property
    def far_field(self) -> bool:
        return True

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return self._case.reference.probe_positions

    def get_object_geometry(self) -> ObjectGeometry:
        return self._case.reference.object_.get_geometry()

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return self._case.assembled_data

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        return self._moments


def build_spectral(case: Case, probe: Probe, overrides: Mapping[str, Any]) -> Object:
    """The registered generator with its default options, less any *overrides*."""
    generator = SpectralObjectGenerator()
    options = generator.get_default_options().model_validate(overrides)
    return generator.generate(
        options, CaseGeometryProvider(case), rng=numpy.random.default_rng(0), probe=probe
    )


BUILDERS: Mapping[str, Callable[[Case, Probe, Mapping[str, Any]], Object]] = {
    'flat': build_flat,
    'stxm_dpc': build_stxm_dpc,
    'spectral': build_spectral,
}


def compute_probe_rms_radius_px(probe: Probe) -> float:
    intensity = numpy.sum(numpy.square(numpy.abs(probe.get_array())), axis=0)
    height_px, width_px = intensity.shape
    y, x = numpy.mgrid[:height_px, :width_px]
    total = float(numpy.sum(intensity))
    cy = float(numpy.sum(y * intensity)) / total
    cx = float(numpy.sum(x * intensity)) / total
    r2 = numpy.square(y - cy) + numpy.square(x - cx)
    return math.sqrt(float(numpy.sum(r2 * intensity)) / total)


def _smooth_within(
    values: ComplexArrayType, mask: RealArrayType, sigma_px: float
) -> ComplexArrayType:
    """Gaussian smoothing by normalized convolution, blind to everything outside *mask*."""
    inside = (mask > 0.0).astype(numpy.float64)
    norm = numpy.maximum(gaussian_filter(inside, sigma_px), 1e-12)
    masked = values * inside
    real = gaussian_filter(masked.real, sigma_px)
    imag = gaussian_filter(masked.imag, sigma_px)
    return (real + 1j * imag) / norm


def _high_pass(values: ComplexArrayType, mask: RealArrayType, sigma_px: float) -> ComplexArrayType:
    return values - _smooth_within(values, mask, sigma_px)


def _complex_correlation(a: ComplexArrayType, b: ComplexArrayType) -> float:
    """``|<a, b>| / (|a| |b|)``; zero when either has an RMS below 1e-5, no content."""
    norm_a = float(numpy.linalg.norm(a))
    norm_b = float(numpy.linalg.norm(b))

    if norm_a <= 1e-5 * math.sqrt(a.size) or norm_b <= 1e-5 * math.sqrt(b.size):
        return 0.0

    return float(abs(numpy.vdot(a, b))) / (norm_a * norm_b)


def _hotness(values: ComplexArrayType, mask: RealArrayType, sigma_px: float) -> float:
    smoothed = numpy.abs(_smooth_within(numpy.abs(values) + 0j, mask, sigma_px))[mask > 0.0]
    median = float(numpy.median(smoothed))
    return float(numpy.quantile(smoothed, 0.99)) / median if median > 0.0 else math.inf


@dataclass(frozen=True)
class Scores:
    corr: float
    detail: float
    band: float
    fine: float
    phase_ssim: float
    hotness: float
    reference_hotness: float
    sigma_px: float
    shift_px: float


def _remove_phase_ramp(values: ComplexArrayType, mask: RealArrayType) -> ComplexArrayType:
    """*values* less the phase ramp of the weighted mean neighbour phasor along each axis."""
    k_x = numpy.angle(numpy.sum(mask[:, 1:] * values[:, 1:] * numpy.conj(values[:, :-1])))
    k_y = numpy.angle(numpy.sum(mask[1:, :] * values[1:, :] * numpy.conj(values[:-1, :])))
    y, x = numpy.indices(values.shape)
    return values * numpy.exp(-1j * (k_x * x + k_y * y))


def _local_contrast(
    values: ComplexArrayType, mask: RealArrayType, sigma_px: float
) -> ComplexArrayType:
    """``x / S(x) - 1`` for a smoothing ``S`` over ``8 sigma``, after removing the phase
    ramp: blind to ramps and slow envelopes, and zero for a flat object."""
    values = _remove_phase_ramp(values, mask)
    envelope = _smooth_within(values, mask, 8.0 * sigma_px)
    magnitude = numpy.abs(envelope)
    floor = 1e-6 * float(numpy.max(magnitude))
    ratio: ComplexArrayType = numpy.divide(
        values, envelope, out=numpy.ones_like(values), where=magnitude > floor
    )
    return ratio - 1.0


def _standardize(reference: Product, test: Product, weights: RealArrayType) -> ComplexArrayType:
    ambiguities = estimate_reconstruction_ambiguities(test, reference=reference, weights=weights)
    return ambiguities.standardize_product(test).object_.get_layers_flattened()


def _estimate_small_shift(
    reference: ComplexArrayType,
    test: ComplexArrayType,
    mask: RealArrayType,
    *,
    sigma_px: float,
    max_shift_px: int,
) -> tuple[float, float]:
    """The shift of *test* onto *reference*, searched within ``max_shift_px`` of zero.

    Correlates the ramp-free local contrast, which a featureless or weak estimate cannot
    drag far, and refines the peak with a parabola along each axis.
    """
    a = _local_contrast(reference, mask, sigma_px) * mask
    b = _local_contrast(test, mask, sigma_px) * mask
    correlation = numpy.abs(numpy.fft.ifft2(numpy.fft.fft2(a) * numpy.conj(numpy.fft.fft2(b))))
    offsets = numpy.arange(-max_shift_px, max_shift_px + 1)
    window = correlation[numpy.ix_(offsets % correlation.shape[0], offsets % correlation.shape[1])]
    row, column = numpy.unravel_index(numpy.argmax(window), window.shape)
    shift = []

    for index, profile in ((row, window[:, column]), (column, window[row, :])):
        delta = 0.0

        if 0 < index < len(profile) - 1:
            left, centre, right = profile[index - 1], profile[index], profile[index + 1]
            curvature = left - 2.0 * centre + right
            delta = 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0

        shift.append(float(offsets[index]) + delta)

    return shift[0], shift[1]


def _align(
    reference: Product, test: Product, weights: RealArrayType, sigma_px: float
) -> tuple[ComplexArrayType, ComplexArrayType, float]:
    """The reference and the test, registered within a few pixels and standardized.

    Every estimate is built on the reference's canvas from its positions, so only the
    pixel or two by which the probes' centres differ is registered.
    """
    ref = reference.object_.get_layers_flattened().astype(numpy.complex128)
    est = _standardize(reference, test, weights).astype(numpy.complex128)
    shift_y_px, shift_x_px = _estimate_small_shift(
        ref, est, weights, sigma_px=sigma_px, max_shift_px=MAX_SHIFT_PX
    )
    shifted = replace(
        test,
        object_=shift_object(test.object_, shift_y_px=shift_y_px, shift_x_px=shift_x_px),
    )
    est = _standardize(reference, shifted, weights).astype(numpy.complex128)
    return ref, est, math.hypot(shift_y_px, shift_x_px)


def score_object(case: Case, object_: Object) -> tuple[Scores, ComplexArrayType, ComplexArrayType]:
    reference = case.reference
    test = replace(reference, object_=object_)
    shape = (reference.object_.height_px, reference.object_.width_px)
    weights = compute_product_scoring_region(reference, shape).get_array_weights()
    sigma_px = compute_probe_rms_radius_px(reference.probes.get_mean_probe())
    ref, est, shift_px = _align(reference, test, weights, sigma_px)
    ref = ref.astype(numpy.complex128)
    est = est.astype(numpy.complex128)
    inside = weights > 0.0
    ref_contrast = _local_contrast(ref, weights, sigma_px)
    est_contrast = _local_contrast(est, weights, sigma_px)
    scores = Scores(
        corr=_complex_correlation(ref[inside], est[inside]),
        detail=_complex_correlation(
            _high_pass(ref, weights, sigma_px)[inside], _high_pass(est, weights, sigma_px)[inside]
        ),
        band=_complex_correlation(
            _smooth_within(ref_contrast, weights, 0.5 * sigma_px)[inside],
            _smooth_within(est_contrast, weights, 0.5 * sigma_px)[inside],
        ),
        fine=_complex_correlation(
            _high_pass(ref_contrast, weights, 0.5 * sigma_px)[inside],
            _high_pass(est_contrast, weights, 0.5 * sigma_px)[inside],
        ),
        phase_ssim=float(
            compute_phase_pixelwise_metrics(ref, est, mask=inside).structural_similarity
        ),
        hotness=_hotness(est, weights, sigma_px),
        reference_hotness=_hotness(ref, weights, sigma_px),
        sigma_px=sigma_px,
        shift_px=shift_px,
    )
    return scores, ref, est


def save_panel(path: Path, title: str, ref: ComplexArrayType, est: ComplexArrayType) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    for row, (label, values) in enumerate((('LSQML', ref), ('estimate', est))):
        amplitude = numpy.abs(values)
        low, high = numpy.quantile(amplitude, (0.01, 0.99))
        axes[row, 0].imshow(amplitude, cmap='gray', vmin=low, vmax=high)
        axes[row, 0].set_title(f'{label} amplitude')
        axes[row, 1].imshow(numpy.angle(values), cmap='twilight', vmin=-PI, vmax=PI)
        axes[row, 1].set_title(f'{label} phase')

    for axis in axes.flat:
        axis.set_axis_off()

    fig.suptitle(title)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run_warm_starts(
    case: Case, initial: Object, epochs: Sequence[int], *, seed: int
) -> dict[str, Any]:
    from ptychodus.api.plugins import PluginRegistry
    from ptychodus.api.reconstruct import prepare_reconstruct_input

    registry = PluginRegistry.load_plugins()
    library = registry.reconstructor_libraries.get_strategy_by_name('PtyChi')
    reconstructor = next(r for r in library if r.name == 'LSQML')
    options: Any = reconstructor.get_default_options()
    task = options.task
    task.reconstructor_options.num_epochs = max(epochs)
    task.reconstructor_options.batch_size = 96
    task.reconstructor_options.random_seed = seed
    task.reconstructor_options.allow_nondeterministic_algorithms = False
    task.object_options.optimizable = True
    task.object_options.build_preconditioner_with_all_modes = True
    task.probe_options.optimizable = True
    task.probe_options.orthogonalize_incoherent_modes.enabled = True
    task.probe_options.center_constraint.enabled = True
    task.probe_options.center_constraint.optimization_plan.stride = 50
    task.probe_position_options.optimizable = False
    options.num_sync_epochs = math.gcd(*epochs)

    probe = generate_incoherent_probe_modes(case.probes['model'], 4)
    product = replace(
        case.reference,
        probes=ProbeEnsemble.from_probe(probe),
        object_=Object(
            array=initial.get_array() * case.mean_amplitude_transmission,
            pixel_geometry=initial.get_pixel_geometry(),
            center=initial.get_center(),
        ),
        losses=[],
    )
    rows: list[dict[str, float]] = []
    losses: list[tuple[int, float]] = []

    for output in reconstructor.reconstruct(
        prepare_reconstruct_input(case.assembled_data, product), options=options
    ):
        losses = [(int(loss.epoch), float(loss.value)) for loss in output.product.losses]

        if output.progress in epochs:
            scores, _, _ = score_object(case, output.product.object_)
            rows.append(
                {
                    'epoch': int(output.progress),
                    'loss': losses[-1][1] if losses else math.nan,
                    'corr': scores.corr,
                    'detail': scores.detail,
                    'phase_ssim': scores.phase_ssim,
                    'band': scores.band,
                    'fine': scores.fine,
                }
            )

    return {'checkpoints': rows, 'losses': losses}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--probe', action='append', choices=('model', 'average', 'lsqml'))
    parser.add_argument('--method', action='append', choices=tuple(BUILDERS))
    parser.add_argument('--spectral', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--stxm-dpc', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--save-arrays', action='store_true')
    parser.add_argument('--warm-start-epochs', type=int, nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--remove-ramp', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    overrides = {
        'flat': {},
        'stxm_dpc': _parse_overrides(args.stxm_dpc),
        'spectral': _parse_overrides(args.spectral),
    }
    run_directory = args.output_dir / args.label
    run_directory.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / 'results.jsonl').open('a') as results:
        for name in args.dataset:
            case = load_case(args.data_root, name)

            for probe_name in args.probe or ['model']:
                for method in args.method or ['spectral']:
                    tic = time.perf_counter()
                    object_ = BUILDERS[method](case, case.probes[probe_name], overrides[method])

                    if args.remove_ramp:
                        array = object_.get_array()[0]
                        weights = numpy.abs(array - 1.0) > 1e-12
                        object_ = Object(
                            array=_remove_phase_ramp(array, weights.astype(float))[numpy.newaxis],
                            pixel_geometry=object_.get_pixel_geometry(),
                            center=object_.get_center(),
                        )
                    seconds = time.perf_counter() - tic
                    scores, ref, est = score_object(case, object_)
                    row: dict[str, Any] = {
                        'label': args.label,
                        'dataset': name,
                        'probe': probe_name,
                        'method': method,
                        'overrides': overrides[method],
                        'seconds': seconds,
                        **scores.__dict__,
                    }
                    stem = f'{name}_{probe_name}_{method}'

                    if args.warm_start_epochs:
                        row['warm_start'] = run_warm_starts(
                            case, object_, args.warm_start_epochs, seed=args.seed
                        )

                        for checkpoint in row['warm_start']['checkpoints']:
                            print(
                                f'{"":>16s} {"":>34s} warm start epoch {checkpoint["epoch"]:4d}'
                                f'  loss={checkpoint["loss"]:.4g} detail={checkpoint["detail"]:.3f}'
                                f' band={checkpoint["band"]:.3f} fine={checkpoint["fine"]:.3f}'
                                f' ssim={checkpoint["phase_ssim"]:.3f}',
                                flush=True,
                            )

                    print(
                        f'{args.label:>16s} {name:>34s} {probe_name:>5s} {method:>8s}'
                        f'  detail={scores.detail:.3f} band={scores.band:.3f} fine={scores.fine:.3f}'
                        f' ssim={scores.phase_ssim:.3f} hot={scores.hotness:.2f}'
                        f' shift={scores.shift_px:.1f}'
                        f'/{scores.reference_hotness:.2f} t={seconds:.1f}s',
                        flush=True,
                    )
                    results.write(json.dumps(row) + '\n')
                    results.flush()

                    if args.figures:
                        save_panel(run_directory / f'{stem}.png', f'{args.label}: {stem}', ref, est)

                    if args.save_arrays:
                        numpy.save(run_directory / f'{stem}.npy', est.astype(numpy.complex64))

                        if not (args.output_dir / f'{name}_reference.npy').exists():
                            numpy.save(
                                args.output_dir / f'{name}_reference.npy',
                                ref.astype(numpy.complex64),
                            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
