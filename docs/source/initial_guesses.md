# Initial Guess Generation

Ptychodus can create initial data products from existing files or from generator routines. The easiest public entry point is the workflow API:

```python
from pathlib import Path

from ptychodus.model import ModelCore

with ModelCore(Path('settings.ini')) as model:
    product_api = model.workflow_api.create_product(
        'initial_guess',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0e6,
    )

    product_api.generate_probe_positions(
        'rectangular_raster',
        {
            'num_points_x': 20,
            'num_points_y': 20,
            'step_size_x_m': 75e-9,
            'step_size_y_m': 75e-9,
        },
    )

    product_api.generate_probe(
        'disk',
        {
            'diameter_m': 1.0e-6,
            'defocus_distance_m': 0.0,
            'num_incoherent_modes': 1,
            'num_coherent_modes': 1,
        },
    )

    product_api.generate_object(
        'random',
        {
            'amplitude_mean': 1.0,
            'amplitude_deviation': 0.0,
            'phase_deviation_turns': 0.1,
            'blur_deviation_px': 0.0,
        },
    )

    product = product_api.get_product()
    product_api.save_product(Path('initial_guess.h5'), file_type='HDF5')
```

The product contains four main pieces of initial data:

- metadata: detector distance, probe energy, photon count, exposure time, etc.;
- probe positions: a {py:class}`ptychodus.api.probe_positions.ProbePositionSequence`;
- probe: a {py:class}`ptychodus.api.probe.ProbeSequence`;
- object: a {py:class}`ptychodus.api.object.Object`.

The workflow methods select generator builders by name. Passing no name uses the current settings. Passing a name plus a parameter mapping creates a builder for that product and immediately rebuilds the corresponding data.

## Data Shapes

Ptychodus stores complex arrays with explicit leading mode or layer dimensions.

Probe
: A single {py:class}`ptychodus.api.probe.Probe` stores its array as `(n_incoherent_modes, height, width)`. A two-dimensional complex input is promoted to `(1, height, width)`.

Probe sequence
: {py:class}`ptychodus.api.probe.ProbeSequence` stores its array as `(n_coherent_modes, n_incoherent_modes, height, width)`. The first dimension is the OPR/coherent-mode dimension. The second dimension is the mutually incoherent probe-mode dimension. A two-dimensional complex input is promoted to `(1, 1, height, width)`; a three-dimensional input is promoted to `(1, n_incoherent_modes, height, width)`.

OPR weights
: If OPR/coherent modes are active, `ProbeSequence` may also carry `opr_weights` with shape `(n_scan_points, n_coherent_modes)`. When a scan point is indexed from a `ProbeSequence`, Ptychodus forms the position-specific dominant incoherent mode as the weighted sum of the coherent/OPR modes and then returns a regular `Probe`.

Object
: {py:class}`ptychodus.api.object.Object` stores its complex transmission array as `(n_layers, height, width)`. A two-dimensional complex input is promoted to `(1, height, width)`. For multislice objects, the `layer_spacing_m` sequence must contain `n_layers - 1` spacings.

Probe positions
: Each {py:class}`ptychodus.api.probe_positions.ProbePosition` stores an integer scan index and physical `x`/`y` coordinates in meters. Internally, `ProbePositionSequence` keeps the coordinate array in `(y, x)` order, but the public object properties are named `coordinate_x_m` and `coordinate_y_m`.

## Pixel Size Of A File-Supplied Guess

An object or probe is sampled at the pixel size its own run implied, which the photon energy and the detector distance fix. Reusing one as an initial guess for a run that samples differently means starting from a guess of the wrong physical scale, and nothing downstream reveals it: the reconstruction simply converges to the wrong answer.

Ptychodus therefore reconciles a file-supplied guess to the run's sampling. {py:func}`ptychodus.api.object.resample_object_to_pixel_size` re-expresses an object at the run's pixel size, preserving its physical extent and center and letting the pixel count absorb the difference; {py:func}`ptychodus.api.probe.resample_probe_sequence` re-expresses every probe mode on the run's probe grid, scaling amplitudes so the photon count survives. Both interpolate with a monotone cubic, so no resampled amplitude or phase overshoots the source, and both return the input untouched when the two samplings already agree.

```{note}
This reconciliation only ever fires for a file format that records its pixel size. None of the registered object or probe formats do — NPY and CSV store a bare array — so a guess loaded from one of those is taken to be sampled the way the current run is, exactly as before.
```

A product opened from HDF5 or NPZ does carry its pixel size, and that path behaves differently on purpose. Such a product is a finished record rather than an initial guess, and the same path also carries live reconstruction output, so a disagreement is reported as a warning naming both samplings and the array is left alone. The disagreement arises when a product is rebound to another dataset, or when its photon energy or detector distance is edited after loading.

## Probe Position Generators

Use `WorkflowProductAPI.generate_probe_positions(name, parameters)`.

Supported generator names are:

`rectangular_raster`
: Cartesian grid, row-major order.

`rectangular_snake`
: Cartesian grid with alternating scan direction on each row.

`triangular_raster` and `triangular_snake`
: Cartesian grid with staggered rows.

`square_raster` and `square_snake`
: Equilateral Cartesian grid with equal `x`/`y` step sizes.

`hexagonal_raster` and `hexagonal_snake`
: Equilateral staggered grid. The row spacing is derived from the `x` step by multiplying by `sqrt(3/4)`.

`concentric`
: Concentric shell scan.

`spiral`
: Spiral scan.

`lissajous`
: Lissajous scan.

All position builders share these transform parameters:

`affine00`, `affine01`, `affine02`, `affine10`, `affine11`, `affine12`
: Affine transform applied to generated positions.

`jitter_radius_m`
: Optional random jitter radius in meters. A value of zero disables jitter.

The Cartesian builders use:

`num_points_x`, `num_points_y`
: Number of grid points along each axis.

`step_size_x_m`, `step_size_y_m`
: Physical step size in meters. Equilateral variants derive the effective `y` step from `step_size_x_m`.

## Probe Generators

Use `WorkflowProductAPI.generate_probe(name, parameters)`. All generated probe builders first create a base complex probe, rescale its total intensity to the product metadata field `probe_photon_count`, then expand it into incoherent and OPR/coherent modes according to the mode settings described below.

`disk`
: Generates a binary circular aperture with parameter `diameter_m`. The aperture is optionally propagated by `defocus_distance_m` with the angular-spectrum propagator. This is the default probe builder.

`rectangular`
: Generates a binary rectangular aperture with `width_m` and `height_m`. It is also optionally propagated by `defocus_distance_m` with the angular-spectrum propagator.

`super_gaussian`
: Generates a super-Gaussian amplitude profile. Parameters are `annular_radius_m`, `full_width_at_half_maximum_m` (or the builder parameter name `fwhm_m` in the low-level function), and `order_parameter`. `annular_radius_m = 0` gives a Gaussian-like spot; a positive `annular_radius_m` gives a ring or donut-like profile. This is not a propagated zone-plate simulation; it is an analytic amplitude profile.

`fresnel_zone_plate`
: Simulates a Fresnel zone plate optic and propagates it to the sample plane with the Fresnel-transform propagator. Parameters are `zone_plate_diameter_m`, `outermost_zone_width_m`, `central_beamstop_diameter_m`, and `defocus_distance_m`. The central beamstop creates the donut-like zone-plate aperture. The focal length is computed as `zone_plate_diameter_m * outermost_zone_width_m / probe_wavelength_m` and the propagation distance is `focal_length_m + defocus_distance_m`.

`kb_mirror`
: Simulates a Kirkpatrick-Baez mirror pair and propagates it to the sample plane. Each mirror takes an `acceptance_length_m`, a `grazing_angle_rad` and a `focus_distance_m`, from which the projected aperture `L sin(theta)` and the numerical aperture follow; `astigmatism_m` separates the two axis foci along the beam and `defocus_distance_m` moves the sample off the focal plane. Grazing angles are stored in radians rather than the turns the GUI uses elsewhere, because that is the number an optics log records. The pupil window is `probe_wavelength_m * z / probe_pixel_size_m`, so a high-aperture optic needs a fine probe pixel size; an aperture that does not fit is refused rather than silently clipped, since clipping would quietly change the optic.

`average_pattern`
: Estimates a probe from diffraction data by taking the square root of the mean assembled diffraction pattern and back-propagating it by `detector_distance_m` with the Fresnel-transform propagator. This requires assembled diffraction data to already be available in the model.

`zernike`
: Generates a probe as a superposition of Zernike polynomial modes inside a disk with parameter `diameter_m`. Through the builder object, callers can set the Zernike order and individual coefficients before rebuilding. The workflow parameter mapping covers the common builder parameters, but coefficient-level editing is done on the `ZernikeProbeBuilder` object.

### Mirror Presets

Kirkpatrick-Baez mirror pairs register the same way zone plates do, through `registry.kb_mirrors`. None ship yet: a preset states an instrument's measured optics, and those numbers reach a reconstruction, so the presets menu stays empty until real measurements are available and the builder works from its own settings in the meantime.

### Zone Plate Presets

Ptychodus registers Fresnel zone plate presets as plugins. The current presets are:

- `APS 2-ID-D`: `160e-6` m diameter, `70e-9` m outermost zone width, `60e-6` m central beamstop.
- `CNM/APS 26-ID-C Hard X-ray Nanoprobe`: `160e-6` m diameter, `30e-9` m outermost zone width, `80e-6` m central beamstop.
- `APS 31-ID-E LamNI`: `114.8e-6` m diameter, `60e-9` m outermost zone width, `40e-6` m central beamstop.
- `APS 33-ID-C PtychoProbe`: `180e-6` m diameter, `15e-9` m outermost zone width, `15e-6` m central beamstop.
- `APS 33-ID-C VelociProbe`: `180e-6` m diameter, `50e-9` m outermost zone width, `60e-6` m central beamstop.

The GUI builder exposes these as presets. Programmatic workflow use can pass the physical values directly.

## Probe Mode Generation

Every generated probe type uses the shared `ProbeSequenceBuilder._build_probe_modes` path:

1. Generate one base {py:class}`ptychodus.api.probe.Probe`.
2. Expand to `num_incoherent_modes` with {py:func}`ptychodus.api.simulate.probe.generate_incoherent_probe_modes`.
3. Expand to `num_coherent_modes` OPR modes with {py:func}`ptychodus.api.simulate.probe.generate_coherent_probe_modes`.
4. Store the final result as a `ProbeSequence` with array shape `(num_coherent_modes, num_incoherent_modes, height, width)`.

### Incoherent Modes

The incoherent-mode controls are:

`num_incoherent_modes`
: Number of mutually incoherent modes to store in axis 1 of the final `ProbeSequence`.

`orthogonalize_incoherent_modes`
: If true and more than one incoherent mode is requested, the generated modes are orthogonalized with `scipy.linalg.orth`.

`incoherent_mode_decay_type`
: `none`, `polynomial`, or `exponential`.

`incoherent_mode_decay_ratio`
: Relative strength of later modes. The builder converts this to a list of unnormalized mode weights.

The low-level `generate_incoherent_probe_modes` routine takes a mode count and, optionally, an `IncoherentModeStrategy`. A strategy answers only two questions: what shape the new modes take, and how power would divide across a sequence of a given length. Everything else is the same for every strategy and belongs to the routine itself.

What the routine guarantees, whatever strategy is in play:

- Modes the probe already carries are **kept in order, by identity** — not merely their span — and the strategy supplies only the shortfall.
- Asking for fewer modes than the probe has keeps the strongest.
- Preserved modes keep the share of the power they arrived with; the new modes divide the share the strategy's profile assigns to their indices. A 90/10 two-mode probe expanded to four at a decay ratio of 0.5 comes out `[0.727, 0.073, 0.133, 0.067]`, so the 9:1 ratio survives.
- **Total power is unchanged**, expanding or truncating, and a mode dropped as linearly dependent passes its share to the modes that survive rather than leaving the probe dimmer. The number of modes used to represent a beam is a modeling choice; the illumination is not, and a probe read from file is never rescaled afterward.
- Orthogonalization is modified Gram-Schmidt in mode order, which is what keeps mode `n` paired with mode `n`. `mode_dependence_floor` sets how much of its own length a mode's orthogonal part must retain to be kept.

Which strategy runs is the `IncoherentModeStrategy` setting, one of `MomentPolynomial`, `RandomPhaseRamp` or `GaussianSchell`; an unrecognized value falls back to the first. In the GUI it is a combo box, and the parameters below it change with the selection so that only the ones the chosen strategy reads are shown — the decay profile for the first two, the source size and coherence length for Gaussian-Schell, which predicts its own spectrum and never consults a decay profile.

`ProbeMomentPolynomialStrategy` is the default, and needs no physical parameters at all. It measures the intensity-weighted centroid and variance of the probe it is given — from the incoherent sum of its modes, which is the footprint the illumination actually covers — and uses that width as its own length scale, so one call suits any illumination without being told how the probe was formed. A new mode is the dominant mode times `u**m * v**n`, with `u` and `v` the transverse coordinates centered on the probe and measured in units of its rms width, damped by a Gaussian of that same width so the polynomial does not push the higher orders into the tails. Orders are graded by total degree and skip the constant term. Weights come from `ProbeModeDecayType` and depend on a mode's index rather than its order.

`RandomPhaseRampStrategy` creates each new mode by duplicating the dominant mode and applying random separable phase wraps along the horizontal and vertical axes, so the added modes share the dominant mode's amplitude structure and differ only by a random linear phase ramp in `x` and `y`. It is the only strategy that consumes randomness, and so the only one that takes an `rng`.

`GaussianSchellStrategy` builds the Hermite-Gauss eigenmodes of a Gaussian-Schell source and *predicts* the weights rather than taking them from a decay profile: given a beam size and coherence length per axis, the two-dimensional spectrum is the outer product of two geometric series and the strongest modes are kept. That spectrum describes its own eigenmodes, so on a probe that already carries modes the weights at the preserved indices are approximate.

The weight rules are:

`none`
: `[1.0, 0.0, 0.0, ...]`.

`polynomial`
: For mode index `n`, weight `(n + 1) ** b` where `b = log(decay_ratio) / log(2)`.

`exponential`
: For mode index `n`, weight `(1 / decay_ratio) ** -n`.

Because the weights are normalized before intensity rescaling, only their relative values matter.

### OPR / Coherent Modes

Ptychodus uses the term `coherent modes` for the leading `ProbeSequence` dimension. In the reconstruction context this is the OPR mode dimension. The control is:

`num_coherent_modes`
: Number of OPR/coherent modes to store in axis 0 of the final `ProbeSequence`.

For `num_coherent_modes == 1`, no OPR weights are stored and the probe sequence length is one.

For `num_coherent_modes > 1`:

- `opr_weights` is initialized with shape `(num_diffraction_patterns, num_coherent_modes)`.
- The first OPR weight is set to `1.0` for every scan point.
- Other weights are initialized as small random normal values with default scale `1.0e-6`.
- OPR mode 0 copies the full incoherent-mode probe.
- Additional OPR modes are initialized as random complex arrays only in incoherent mode 0. Other incoherent modes in those OPR modes are left zero.
- The random OPR mode images are normalized by their mean intensity when `normalize_cmodes` is true.

### Reusing an OPR Probe on Another Scan

OPR weights carry one row per probe position, so a probe solved on one scan cannot initialize a reconstruction of another until those rows are resolved. {py:func}`ptychodus.api.probe.conform_opr_weights` does that under a policy named by {py:class}`ptychodus.api.probe.OPRWeightPolicy`:

| Policy | Effect |
| --- | --- |
| `KEEP` | Use the weights unchanged. Raises when they describe a different number of probe positions. |
| `AVERAGE` | Give every probe position the mean of the loaded weight rows, keeping the coherent modes. |
| `REINITIALIZE` | Keep the coherent modes and start the weights where a fresh OPR run would: `1.0` on the primary mode, small noise on the rest. |
| `COLLAPSE` | Combine the coherent modes under the mean weight row into a single mode, then drop the OPR basis. |
| `DISCARD` | Keep coherent mode 0, dropping the remaining modes along with the weights. |

The setting is `OPRWeightPolicy` in the `Probe` group, defaulting to `Average`. It is consulted **only when the row count disagrees** with the run's probe positions, so a warm start whose weights already fit is never altered. The reconciliation happens before the expand-only mode pipeline, which means `DISCARD` together with a larger `num_coherent_modes` discards the stale basis and builds a fresh one sized to the current scan.

Opening a probe file that carries OPR weights in the GUI prompts for the policy, stating both counts, and that choice applies to that one load whether or not the counts agree. `Use as is` is offered only when they do.

```{note}
`COLLAPSE` applies the mean weight row across every incoherent mode. Indexing a position out of a {py:class}`ptychodus.api.probe.ProbeSequence` instead rewrites only incoherent mode 0 from the coherent basis.
```

### Examples

#### Generate a multimode probe from scratch

This example generates only the probe component of a product: a defocused Fresnel zone plate probe with three incoherent modes and two OPR/coherent modes. The product's current scan length is still used internally to size the OPR weights, but no object or explicit scan-position setup is needed for this probe-only example.

```python
from pathlib import Path

from ptychodus.model import ModelCore

with ModelCore(Path('settings.ini')) as model:
    product_api = model.workflow_api.create_product(
        'generated_multimode_probe',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0e6,
    )

    product_api.generate_probe(
        'fresnel_zone_plate',
        {
            'zone_plate_diameter_m': 180e-6,
            'outermost_zone_width_m': 50e-9,
            'central_beamstop_diameter_m': 60e-6,
            'defocus_distance_m': 10e-6,
            'num_incoherent_modes': 3,
            'orthogonalize_incoherent_modes': True,
            'incoherent_mode_decay_type': 'polynomial',
            'incoherent_mode_decay_ratio': 0.25,
            'num_coherent_modes': 2,
        },
    )

    product = product_api.get_product()
    probe_array = product.probes.get_array()
    opr_weights = product.probes.get_opr_weights()

    assert probe_array.shape[:2] == (2, 3)
    assert opr_weights.shape == (len(product.probe_positions), 2)

    product_api.save_product(Path('generated_multimode_probe.h5'), file_type='HDF5')
```

The same mode parameters work with other generated probe types such as `disk`, `rectangular`, `super_gaussian`, and `zernike`.

#### Add incoherent and OPR modes to an existing single-mode probe

The workflow API can load an existing probe with `load_probe()`, but the current high-level `generate_probe()` call replaces the probe with a named generator rather than augmenting the loaded probe in place. To augment an already loaded single-mode probe, use the low-level mode-expansion routines on the loaded `ProbeSequence` and then register the updated product back with the workflow API.

```python
from pathlib import Path

from ptychodus.api.product import Product
from ptychodus.api.simulate.probe import (
    generate_coherent_probe_modes,
    generate_incoherent_probe_modes,
)
from ptychodus.model import ModelCore

with ModelCore(Path('settings.ini')) as model:
    product_api = model.workflow_api.create_product(
        'loaded_single_mode_probe',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0e6,
    )

    product_api.load_probe(Path('single_mode_probe.npy'), file_type='NPY')

    product = product_api.get_product()

    # Collapse any OPR handling in the loaded ProbeSequence and start from
    # the first coherent/OPR mode as a regular Probe.
    base_probe = product.probes.get_probe_no_opr()

    probe_with_incoherent_modes = generate_incoherent_probe_modes(
        base_probe,
        3,
        orthogonalize=True,
    )

    expanded_probe_sequence = generate_coherent_probe_modes(
        model.rng,
        probe_with_incoherent_modes,
        num_cmodes=2,
        num_diffraction_patterns=len(product.probe_positions),
    )

    expanded_product = Product(
        metadata=product.metadata,
        probe_positions=product.probe_positions,
        probes=expanded_probe_sequence,
        object_=product.object_,
        losses=product.losses,
    )

    expanded_product_api = model.workflow_api.register_product(expanded_product)
    expanded_product_api.rename_product('loaded_probe_with_added_modes')
    expanded_product_api.save_product(
        Path('loaded_probe_with_added_modes.h5'),
        file_type='HDF5',
    )
```

This replaces the probe with a new `ProbeSequence` whose shape is `(2, 3, height, width)` and whose OPR weights have shape `(n_scan_points, 2)`.

## Object Generators

Use `WorkflowProductAPI.generate_object(name, parameters)`. Object geometry is derived from the current probe geometry and scan bounding box, then optional extra padding is applied by `extra_padding_x` and `extra_padding_y`.

`random`
: Generates a complex object from Gaussian amplitude and phase fields. Parameters are `amplitude_mean`, `amplitude_deviation`, `phase_deviation_turns`, and `blur_deviation_px`. The generated field is `amplitude * exp(2j * pi * phase_turns)`.

`grf`
: Generates a complex Gaussian random field by spectral synthesis. Parameter: `correlation_length_px`.

`fractal_noise`
: Generates a complex fractal-noise object by summing multiple octaves of simplex noise. Parameters are `grid_scale_px`, `num_octaves`, `gain`, and `lacunarity`.

`dead_leaves`
: Generates a layered random disk texture. Parameters include `leaf_radius_lower_px`, `leaf_radius_upper_px`, `leaf_radius_power_law_exponent`, `leaf_amplitude_lower`, `leaf_amplitude_upper`, `leaf_phase_lower_tr`, and `leaf_phase_upper_tr`.

`stxm`
: Generates an STXM-like object from assembled diffraction data by interpolating per-position diffraction counts onto the object grid. This requires assembled diffraction data and probe positions.

### Multislice Objects

Object builders call `generate_layers` after creating an object. If `object_layer_spacing_m` is empty, the object keeps the slices it already has -- an unset spacing expresses no preference, so it never collapses a multislice warm start. If spacings are supplied, the requested number of slices is `len(object_layer_spacing_m) + 1` and the object is resliced onto that depth grid.

Reslicing is non-destructive. {py:func}`ptychodus.api.object.resample_object_layers` interpolates the cumulative complex optical path -- the running sum of each layer's complex logarithm -- and takes differences of it, so every input layer contributes, the layers still compose multiplicatively, and `get_layers_flattened()` is preserved to round-off whichever slice count is requested. Working in the logarithm is also what keeps a total phase outside `(-pi, pi]` correct: each layer's own phase is small enough not to wrap, and the sum accumulates past `pi` without ever passing through `numpy.angle`.

Expanding a single slice into several is the degenerate case, since there is no depth axis to resample along. It falls back to {py:func}`ptychodus.api.object.homogenize_object_layers`, which spreads the whole transmission evenly so that each of the `n_slices` layers is the `n_slices`-th root of the original.

The layers are phase screens at depths `0` through the total thickness, so `n_slices` layers leave `n_slices - 1` gaps and `Object.get_total_thickness_m()` -- the sum of those gaps -- is the full thickness. {py:func}`ptychodus.api.object.compute_uniform_layer_spacing_m` builds an even grid on that convention.

## Low-Level Generator API

The workflow API is usually simpler, but the low-level functions can be used directly when the caller already has geometry objects and wants arrays without registering a product.

Typical low-level imports are:

```python
import numpy

from ptychodus.api.object import ObjectGeometry
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    generate_coherent_probe_modes,
    generate_disk_probe,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
    rescale_probe_intensity,
)
from ptychodus.api.simulate.probe_positions import generate_cartesian_probe_positions

rng = numpy.random.default_rng(0)

positions = list(
    generate_cartesian_probe_positions(
        num_points_x=20,
        num_points_y=20,
        step_size_x=75e-9,
        step_size_y=75e-9,
    )
)

probe_geometry = ProbeGeometry(
    width_px=128,
    height_px=128,
    pixel_width_m=10e-9,
    pixel_height_m=10e-9,
)

base_probe = generate_fresnel_zone_plate_probe(
    probe_geometry,
    FresnelZonePlate(
        zone_plate_diameter_m=180e-6,
        outermost_zone_width_m=50e-9,
        central_beamstop_diameter_m=60e-6,
    ),
    probe_wavelength_m=1.239841984e-10,  # 10 keV
    defocus_distance_m=0.0,
)
base_probe = rescale_probe_intensity(base_probe, 1.0e6)

probe_with_imodes = generate_incoherent_probe_modes(
    base_probe,
    3,
    orthogonalize=True,
)
probe_sequence = generate_coherent_probe_modes(
    rng,
    probe_with_imodes,
    num_cmodes=2,
    num_diffraction_patterns=len(positions),
)

object_geometry = ObjectGeometry(
    width_px=512,
    height_px=512,
    pixel_width_m=10e-9,
    pixel_height_m=10e-9,
    center_x_m=0.0,
    center_y_m=0.0,
)
object_guess = generate_random_object(
    rng,
    object_geometry,
    amplitude_mean=1.0,
    amplitude_deviation=0.0,
    phase_mean=0.0,
    phase_deviation_tr=0.1,
    blur_deviation_px=0.0,
)
```

## Using Generated Guesses With Pty-Chi

Ptychodus has a Pty-Chi adapter in `ptychodus.model.ptychi`. When using the normal Ptychodus reconstruction path, the adapter receives a {py:class}`ptychodus.api.reconstruct.ReconstructInput`, builds Pty-Chi task options from the settings, and passes the following data from the product to `PtychographyTask` as keyword arguments:

- object initial guess: `product.object_.get_array()`;
- probe initial guess: `product.probes.get_array()`;
- OPR weights: `product.probes.get_opr_weights()` if available, otherwise a one-dimensional default weight vector with first entry `1.0`;
- probe positions: physical Ptychodus positions are mapped through the Ptychodus object geometry to object pixel coordinates before being passed to Pty-Chi.

### Probe And Object Sampling

Pty-Chi 2.1.0 and later allow the probe to be sampled differently from the object: `probe_options.pixel_size_m` and `probe_options.pixel_size_aspect_ratio` each fall back to the matching `object_options` value when left at `None`, and Pty-Chi Fourier-rescales the object onto the probe grid whenever the two disagree.

The adapter derives both fields from `product.probes.get_pixel_geometry()`, alongside the object fields it derives from `product.object_.get_pixel_geometry()`. When a probe value matches the object's it is left at `None` — stating the inheritance explicitly and keeping Pty-Chi's no-resampling path — and width and aspect ratio are decided independently, matching how Pty-Chi inherits them. A product whose probe carries no pixel geometry leaves both fields unset.

### Shape Compatibility

Pty-Chi's documented probe convention is `(n_opr_modes, n_modes, height, width)`:

- `n_opr_modes` is the OPR/eigenmode dimension;
- `n_modes` is the mutually incoherent probe-mode dimension.

This matches Ptychodus `ProbeSequence.get_array()`:

- Ptychodus `num_coherent_modes` maps to Pty-Chi `n_opr_modes`;
- Ptychodus `num_incoherent_modes` maps to Pty-Chi `n_modes`.

Pty-Chi requires a four-dimensional probe initial guess. Therefore, when calling Pty-Chi manually, pass `product.probes.get_array()` as the `probe_data` argument to `PtychographyTask` rather than indexing the probe sequence down to a single `Probe`.

Pty-Chi's planar object convention is `(n_slices, height, width)`, which matches Ptychodus `Object.get_array()`. Pty-Chi also expects multislice `slice_spacings_m` to contain `n_slices - 1` spacings, matching Ptychodus `Object.layer_spacing_m`.

### OPR Weights

Pty-Chi accepts OPR weights as either:

- `(n_scan_points, n_opr_modes)`: per-position weights; or
- `(n_opr_modes,)`: one weight vector broadcast to every scan point.

Ptychodus generated OPR weights use `(n_scan_points, n_coherent_modes)` when `num_coherent_modes > 1`. This is directly compatible with Pty-Chi.

If the probe has more than one OPR mode, Pty-Chi requires initial OPR weights. Ptychodus satisfies this when the OPR modes were generated by `generate_coherent_probe_modes`. If a custom `ProbeSequence` is created manually with multiple leading modes, provide matching `opr_weights`.

### Positions and Object Origin

Ptychodus stores probe positions in meters. Pty-Chi stores probe positions in object pixel units, with order `(y, x)`. The Ptychodus Pty-Chi helper converts positions by calling `object_geometry.map_coordinates_probe_to_object(scan_point)` and then passing the resulting pixel `x` and `y` arrays to `PtychographyTask` as `probe_position_x_px` and `probe_position_y_px`.

The current Ptychodus helper chooses Pty-Chi's `SPECIFIED` position-origin mode and returns an origin coordinate of zero. This works together with the absolute object-pixel coordinates produced by the mapping above. If bypassing the Ptychodus helper and creating Pty-Chi options directly, make sure the position origin convention is consistent with the positions you pass:

- Use object-pixel coordinates in `(y, x)` order.
- Keep the object large enough for every extracted patch: `position_range + probe_shape` must fit inside the object support.
- If using zero-centered positions with Pty-Chi's `SUPPORT` origin mode, ensure `-positions.min()` is approximately `positions.max()` along both axes.

### Minimal Manual Pty-Chi Mapping

When bypassing the Ptychodus reconstructor and calling Pty-Chi manually, the data needed from a generated Ptychodus product is:

```python
product = product_api.get_product()

object_initial_guess = product.object_.get_array()
probe_initial_guess = product.probes.get_array()

try:
    opr_initial_weights = product.probes.get_opr_weights()
except ValueError:
    opr_initial_weights = None

object_pixel_size_m = product.object_.get_pixel_geometry().width_m
slice_spacings_m = product.object_.layer_spacing_m or None

object_geometry = product.object_.get_geometry()
position_y_px = []
position_x_px = []
for scan_point in product.probe_positions:
    object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
    position_y_px.append(object_point.coordinate_y_px)
    position_x_px.append(object_point.coordinate_x_px)
```

Pty-Chi separates settings from data. Set these on the options object:

- `object_options.pixel_size_m = object_pixel_size_m`;
- `object_options.slice_spacings_m = slice_spacings_m` for multislice.

Pass the rest as `PtychographyTask` keyword arguments:

- `object_data=object_initial_guess`;
- `probe_data=probe_initial_guess`;
- `probe_position_x_px=position_x_px`;
- `probe_position_y_px=position_y_px`;
- `opr_mode_weights_data=opr_initial_weights` when `probe_initial_guess.shape[0] > 1`;
- `diffraction_data` and `valid_pixel_mask` for the measured patterns and their mask.

```{note}
Pty-Chi still accepts this data through the corresponding `*_options` fields — `object_options.initial_guess`, `probe_position_options.position_x_px`, and so on — but that path is deprecated and emits a `DeprecationWarning` for each field. Prefer the keyword arguments.
```

## Practical Notes

- The generated probe pixel size is derived from detector geometry, detector distance, and probe energy through the product geometry. For physically meaningful generated probes, create or load the diffraction metadata before generating the product.
- `average_pattern` and `stxm` depend on assembled diffraction data. Analytic generators such as `disk`, `super_gaussian`, `fresnel_zone_plate`, `random`, `grf`, `fractal_noise`, and `dead_leaves` do not require measured patterns.
- `super_gaussian` is the closest built-in Gaussian-like probe. Ptychodus does not currently register a separate builder literally named `gaussian`.
- In workflow parameter mappings, use the builder parameter names, not the settings-file names. For example use `diameter_m` rather than `DiskDiameterInMeters`.
