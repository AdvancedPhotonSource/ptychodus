# Initial Guesses

Ptychodus can create initial data products from existing files or from generator routines. A product is plain data from the stateless `ptychodus.api`, so one can be assembled directly from the routines the generators are built on:

```python
from pathlib import Path

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import save_product
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.simulate.probe import generate_disk_probe, rescale_probe_intensity
from ptychodus.api.simulate.probe_positions import generate_cartesian_probe_positions

rng = numpy.random.default_rng()
metadata = ProductMetadata(
    name='initial_guess',
    comments='',
    detector_distance_m=1.0,
    photon_energy_eV=10_000.0,
    probe_photon_count=1.0e6,
    exposure_time_s=1.0,
    mass_attenuation_m2_per_kg=0.0,
    tomography_angle_deg=0.0,
)
positions = ProbePositionSequence(list(generate_cartesian_probe_positions(20, 20, 75e-9, 75e-9)))

pixel_m = 10e-9
probe_geometry = ProbeGeometry(
    width_px=128, height_px=128, pixel_width_m=pixel_m, pixel_height_m=pixel_m
)
probe = rescale_probe_intensity(
    generate_disk_probe(probe_geometry, radius_m=0.5e-6), metadata.probe_photon_count
)

phase_rad = rng.normal(scale=0.6, size=(512, 512))
object_ = Object(
    array=numpy.exp(1j * phase_rad),
    pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
    center=ObjectCenter(x_m=0.0, y_m=0.0),
)

product = Product(
    metadata=metadata,
    probe_positions=positions,
    probes=ProbeEnsemble.from_probe(probe),
    object_=object_,
    losses=[],
)
save_product(Path('initial_guess.h5'), product)
```

The product contains four main pieces of initial data:

- metadata: detector distance, probe energy, photon count, exposure time, etc.;
- probe positions: a {py:class}`ptychodus.api.probe_positions.ProbePositionSequence`;
- probe: a {py:class}`ptychodus.api.probe.ProbeEnsemble`;
- object: a {py:class}`ptychodus.api.object.Object`.

Each generator below is also a plugin, found by name in the plugin registry — `PluginRegistry.load_plugins()`, under `probe_position_generators`, `probe_generators` and `object_generators` — with its parameters declared by `get_default_options()`. The GUI and the settings file select them by that name, and the drivers and `ptychodus-stream` take them by name too.

## Data Shapes

Ptychodus stores complex arrays with explicit leading mode or layer dimensions.

Probe
: A single {py:class}`ptychodus.api.probe.Probe` stores its array as `(n_incoherent_modes, height, width)`. A two-dimensional complex input is promoted to `(1, height, width)`.

Probe sequence
: {py:class}`ptychodus.api.probe.ProbeEnsemble` stores its array as `(n_coherent_modes, n_incoherent_modes, height, width)`. The first dimension is the OPR/coherent-mode dimension. The second dimension is the mutually incoherent probe-mode dimension. A two-dimensional complex input is promoted to `(1, 1, height, width)`; a three-dimensional input is promoted to `(1, n_incoherent_modes, height, width)`.

OPR weights
: If OPR/coherent modes are active, `ProbeEnsemble` may also carry `opr_weights` with shape `(n_probe_positions, n_coherent_modes)`. Asking a `ProbeEnsemble` for the probe at a probe position forms the position-specific dominant incoherent mode as the weighted sum of the coherent/OPR modes and returns a regular `Probe`. Without `opr_weights` one probe describes every position, which is why a `ProbeEnsemble` is addressed by position rather than being a sequence with a length.

Object
: {py:class}`ptychodus.api.object.Object` stores its complex transmission array as `(n_layers, height, width)`. A two-dimensional complex input is promoted to `(1, height, width)`. For multislice objects, the `layer_spacing_m` sequence must contain `n_layers - 1` spacings.

Probe positions
: Each {py:class}`ptychodus.api.probe_positions.ProbePosition` stores an integer scan index and physical `x`/`y` coordinates in meters. Internally, `ProbePositionSequence` keeps the coordinate array in `(y, x)` order, but the public object properties are named `coordinate_x_m` and `coordinate_y_m`.

## Pixel Size Of A File-Supplied Guess

An object or probe is sampled at the pixel size its own run implied, which the photon energy and the detector distance fix. Reusing one as an initial guess for a run that samples differently means starting from a guess of the wrong physical scale, and nothing downstream reveals it: the reconstruction simply converges to the wrong answer.

Ptychodus therefore reconciles a file-supplied guess to the run's sampling. {py:func}`ptychodus.api.object.resample_object_to_pixel_size` re-expresses an object at the run's pixel size, preserving its physical extent and center and letting the pixel count absorb the difference; {py:func}`ptychodus.api.probe.resample_probe_ensemble` re-expresses every probe mode on the run's probe grid, scaling amplitudes so the photon count survives. Both interpolate the real and imaginary parts with a cubic B-spline, which is linear in the field: the result does not depend on the guess's arbitrary global phase, and resampled probe modes still mix by their OPR weights. A cubic overshoots by a few percent at a hard edge. Both return the input untouched when the two samplings already agree.

```{note}
This reconciliation only ever fires for a file format that records its pixel size. None of the registered object or probe formats do — NPY and CSV store a bare array — so a guess loaded from one of those is taken to be sampled the way the current run is, exactly as before.
```

A product opened from a product file does carry its pixel size, and that path behaves differently on purpose. Such a product is a finished record rather than an initial guess, and the same path also carries live reconstruction output, so a disagreement is reported as a warning naming both samplings and the array is left alone. The disagreement arises when a product is rebound to another dataset, or when its photon energy or detector distance is edited after loading.

## Probe Position Generators

Each pattern is a plugin that states the parameters it accepts, so the names below are what it registered under and the parameter names are the ones it declares. A pattern added to `ptychodus.plugins` appears here, in the settings file and in the editor dialog without any of them being told about it.

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

Every probe-position source, generated or read from a file, shares these transform parameters:

`affine00`, `affine01`, `affine02`, `affine10`, `affine11`, `affine12`
: Affine transform applied to generated positions.

`jitter_radius_m`
: Optional random jitter radius in meters. A value of zero disables jitter.

The Cartesian generators use:

`num_points_x`, `num_points_y`
: Number of grid points along each axis.

`step_size_x_m`, `step_size_y_m`
: Physical step size in meters. The equilateral variants derive their `y` step from `step_size_x_m` and do not accept a `step_size_y_m` at all.

The other patterns use:

`num_points`
: Total number of points, for `spiral` and `lissajous`. Persisted as `NumberOfPoints`, which the two share and the Cartesian patterns do not read.

`radial_step_size_m`, `num_shells`, `num_points_in_first_shell`
: Shell spacing and population, for `concentric`. Each shell holds a multiple of the first shell's count.

`radius_scalar_m`
: Spiral radius at unit index, for `spiral`.

`amplitude_x_m`, `amplitude_y_m`, `angular_step_x_deg`, `angular_step_y_deg`, `angular_shift_deg`
: Figure extent, the two angular rates whose ratio sets the figure, and the phase offset between them, for `lissajous`. The angular values are in degrees and may be negative.

## Probe Generators

Every probe generator first creates a base complex probe, rescale its total intensity to the product metadata field `probe_photon_count`, then expand it into incoherent and OPR/coherent modes according to the mode settings described below.

Each probe generator is a plugin that states the parameters it accepts, so the names below are what it registered under and the parameter names are the ones it declares. A generator added to `ptychodus.plugins` appears here, in the settings file and in the editor dialog without any of them being told about it. A plugin knows nothing about where its values come from: the settings file, a command-line option, a preset and a size measured from the data are all just sources of the same declared parameters. Parameters with the same name share one settings key and one value across generators — `propagation_distance_m`, and the `diameter_m` that `disk` and `zernike` both take as the diameter of a circular pupil — so switching generators keeps them.

Every probe generator generates at the product's defocus, how far past the focus the object sits: the negated product metadata field `focus_object_distance_m`, positive downstream (diverging) and negative upstream (converging), shown in the product editor as both **Focus-Object Distance** and **Defocus Distance**. In far field a generated probe carries the curvature of a beam sampled that far past its focus. In near field it is generated in the equivalent parallel-beam frame the magnification already places it in (see [Near-Field Ptychography](near_field.md)): an aperture model stays flat-phased, and an optic is simulated at the defocus and has the spherical wave of its focus divided out by {py:func}`ptychodus.api.simulate.probe.remove_defocus_phase`. Every generator centres its beam axis on pixel `N // 2` of each axis, the FFT origin and the middle pixel of an odd axis, and that pixel is the one placed on each probe position.

`disk`
: Generates a binary circular aperture with parameter `diameter_m`. In far field it first takes the quadratic phase of a beam sampled the product's defocus past its focus. The aperture is then optionally propagated by `propagation_distance_m` with the angular-spectrum propagator. This is the default probe generator for a product with no diffraction data. With far-field diffraction data loaded, **Estimate from Diffraction** sets `diameter_m` from the second moments of the mean pattern: the bright field's angular spread times the defocus is the geometric shadow the aperture casts, so set the product's focus first. It sets the parameter once and leaves it there. A zone plate's annular bright field reads about 5% larger than its outer diameter at a central stop of a third, and at zero defocus the estimate falls back to the transform-limited waist with a warning.

`rectangular`
: Generates a binary rectangular aperture with `width_m` and `height_m`. It takes the product's defocus and `propagation_distance_m` exactly as `disk` does, and its **Estimate from Diffraction** sets both sides from the spread of the bright field along x and y.

`super_gaussian`
: Generates a super-Gaussian amplitude profile. Parameters are `annular_radius_m`, `full_width_at_half_maximum_m` (`fwhm_m` in the low-level function), and `order_parameter`. `annular_radius_m = 0` gives a Gaussian-like spot; a positive `annular_radius_m` gives a ring or donut-like profile. This is not a zone-plate simulation: the profile is analytic and flat-phased, and takes the product's defocus and `propagation_distance_m` exactly as `disk` does. Its **Estimate from Diffraction** sets `full_width_at_half_maximum_m` to that of a Gaussian beam with the measured angular spread, at its waist plus the spread of the cone over the defocus; the annular radius and order are left as chosen.

`fresnel_zone_plate`
: Simulates a Fresnel zone plate optic and propagates it to the sample plane with the Fresnel-transform propagator. Parameters are `zone_plate_diameter_m`, `outermost_zone_width_m` and `central_beamstop_diameter_m`. The central beamstop creates the donut-like zone-plate aperture. The focal length is computed as `zone_plate_diameter_m * outermost_zone_width_m / photon_wavelength_m` and the propagation distance is the focal length plus the product's defocus. In near field the spherical wave of the focus is then divided out with `remove_defocus_phase`.

`kb_mirror`
: Simulates a Kirkpatrick-Baez mirror pair and propagates it to the sample plane. Each mirror takes an `acceptance_length_m`, a `grazing_angle_rad` and a `focus_distance_m`, from which the projected aperture `L sin(theta)` and the numerical aperture follow; `astigmatism_m` separates the two axis foci along the beam, and the product's defocus places the sample off the focal plane, with the near-field curvature removed about it as for the zone plate; the astigmatism stays in the probe. Grazing angles are in radians, the number an optics log records, where other geometric angles are in degrees. The pupil window is `photon_wavelength_m * z / probe_pixel_size_m`, so a high-aperture optic needs a fine probe pixel size; an aperture that does not fit is refused rather than silently clipped, since clipping would quietly change the optic.

`average_pattern`
: Estimates a probe from diffraction data by taking the square root of the mean assembled diffraction pattern and back-propagating it to the object plane. This is the default probe generator for a product created with diffraction data; it reproduces the measured mean pattern by construction, which a probe generated from the wrong optic parameters does not. Far field takes the square root as the probe's flat-phase angular spectrum and inverse-transforms it onto the probe grid, giving the beam at its waist; the phase the Fresnel transform adds across the detector is not the probe's and is not undone, and no detector pitch is needed. Near field uses the angular spectrum over the demagnified distance, which preserves pitch. Because the far-field estimate is the waist, the product's defocus carries it to the object. The near-field back-propagation already lands in the object plane of the equivalent geometry, so there the defocus is not applied again. `propagation_distance_m` refines where the probe plane sits without altering the geometry the object pixel size comes from; the two act on the same field and add. Bad pixels are inpainted rather than zeroed, since a hole in the bright field rings through the probe. A mean pattern with no signal once corrected — a detector crop that misses the beam — is refused with an error rather than giving an all-zero probe. `bright_field_threshold` (off by default) keeps only the bright field of a far-field mean pattern: the background pedestal is subtracted and the noise around the bright field zeroed, while the bright field keeps its measured values. This requires assembled diffraction data to already be available in the model.

`hermite_gauss`
: Generates a probe as a superposition of Hermite–Gauss modes. `beam_waist_x_m` and `beam_waist_y_m` are the radii at which the intensity of mode `(0, 0)` falls to `1/e²` of its peak; the Gaussian envelope gives every mode a finite extent, about `sqrt(2m + 1)` waists along its axis, so the scale of the polynomials and the size of the probe are one width. `hermite_gauss_coefficients` is a list of complex amplitudes in total-degree order, `(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), ...`, whose length is the number of modes. Each mode is normalized to unit power, so a coefficient is that mode's amplitude relative to the others.

`zernike`
: Generates a probe as a superposition of Zernike polynomial modes inside a disk with parameter `diameter_m`. `zernike_coefficients` is a list of complex amplitudes in OSA/ANSI order, `(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), ...`, whose length is the number of modes. A coefficient list is a parameter like any other, so options set it directly, as complex values or as strings such as `"0.5-0.25j"`; the settings file stores it the same way, as a JSON array of those strings.

**Estimate from Diffraction** fills the parameters a generator declares as sizes, measured at the product's defocus; today that is `disk`, `rectangular` and `super_gaussian`. A size written into a shared parameter is the value every generator sharing it uses, so estimating a disk's diameter also sets the diameter `zernike` uses.

### Mirror Presets

Kirkpatrick-Baez mirror pairs register the same way zone plates do, as a {py:class}`ptychodus.api.simulate.probe.KirkpatrickBaezMirrorPreset` through `registry.kb_mirrors`. Its six fields carry the `kb_mirror` parameter names, which is what lets a registered pair fill them. None ship yet: a preset states an instrument's measured optics, and those numbers reach a reconstruction, so the presets menu stays empty until real measurements are available and the generator works from its own settings in the meantime.

A preset is offered to whichever model declares every field the preset names, and applying one writes those fields and leaves the rest — a defocus, an astigmatism — as they were.

### Zone Plate Presets

Ptychodus registers Fresnel zone plate presets as plugins, each under the one name below, which is also what the reconstruction drivers' `--fzp-preset` takes. The current presets are:

- `APS 2-ID-D`: `160e-6` m diameter, `70e-9` m outermost zone width, `60e-6` m central beamstop.
- `CNM/APS 26-ID-C Hard X-ray Nanoprobe`: `160e-6` m diameter, `30e-9` m outermost zone width, `80e-6` m central beamstop.
- `APS 31-ID-E LamNI`: `114.8e-6` m diameter, `60e-9` m outermost zone width, `40e-6` m central beamstop.
- `APS 33-ID-C PtychoProbe`: `180e-6` m diameter, `15e-9` m outermost zone width, `15e-6` m central beamstop.
- `APS 33-ID-C VelociProbe`: `180e-6` m diameter, `50e-9` m outermost zone width, `60e-6` m central beamstop.

The GUI offers these as presets. Programmatic use can pass the physical values directly.

### Measuring the Defocus

Every probe generator generates at the product's defocus, the negated `focus_object_distance_m`, and the data measure it: the parallax between bright-field images, refined by autofocus (below). The defocus is a guess until stated or measured. **Estimate Defocus** in the product editor runs the measurement on the bound diffraction data. `EstimateDefocusFromParallax`, on by default, runs it on every newly created far-field product that states no focus; an opened product file or a reconstruction result keeps the focus it carries. The reconstruction drivers measure it whenever far-field data states no focus (see [Reconstruction Drivers](reconstruction_drivers.md)). Far-field data only.

A defocused probe lights each point of its bright field through a different point of the sample, so the image one bright-field detector pixel records as the probe scans is the sample shifted by `-lambda * z * q`. The measurement bins every pattern into bins of `ParallaxBinSizeInPixels` detector pixels a side, maps each bright-field bin's values onto the object grid, registers the images against their mean, and fits the shifts to that line, which gives the defocus with its sign ({py:func}`ptychodus.api.defocus.estimate_defocus_from_parallax`). Larger bins cost less memory but leave fewer bins inside the bright field; at least four must remain away from its edge. At most 64 bright-field bins are mapped, an evenly spread subset when the bright field holds more, so the images' memory stays bounded however large the detector's bright field.

The measurement needs:

- little motion during an exposure compared with the object pixel, which a continuous fly scan with long exposures can fail;
- scan steps finer than the parallax shift, about the defocus times the numerical aperture, which a coarse step scan can fail;
- a textured sample;
- a defocus beyond about one depth of focus;
- accurate positions.

The thresholds below catch an estimate lost in noise, but not a systematic error in the positions or their timing, which shifts every image alike. Check the logged error and Fresnel number, and state the focus when in doubt.

A parallax defocus is **resolved** when it reaches `ParallaxMinSignificance` standard errors and `ParallaxMinDefocusDof` depths of focus, and only then is the focus position set. The depth of focus is `lambda / NA**2`, from the second moments of the mean pattern. Within about that distance of the focus the bright-field images are differently weighted phase-contrast images rather than shifted copies of one image, and registering them still finds a slope that its scatter cannot reveal, so a probe at focus would otherwise report a defocus of about half a depth of focus. The same threshold read the other way is a Fresnel number: the defocus in depths of focus equals `R**2 / (lambda |z|)` for the radius `R` of the probe's shadow, the geometric-optics measure the parametric probe generators warn on below one. Both appear in the log.

**Autofocus** then refines a resolved parallax defocus ({py:func}`ptychodus.api.defocus.estimate_defocus_by_autofocus`). The parallax underestimates the defocus by up to half on some data and overestimates it on other, while autofocus asks the patterns directly: for seven defoci on a log grid from a third to three times the parallax value, plus two that refine the best, it generates the `average_pattern` probe at that defocus, takes one step of the `spectral` refinement from the STXM-DPC estimate, and scores the refined object by how well it predicts the measured amplitudes; a parabola through the best three gives the defocus. More than one step amplifies the probe's own errors, and the refinement's Rayleigh quotient is no criterion, since a compact probe raises it whatever the defocus. Near focus the residual is too flat to place the defocus better than the parallax does, so the parallax value is kept when it lies within one depth of focus of focus or the residual's minimum within three. The residuals are measured over a central block of the scan of about 2500 patterns, whatever the detector crop, which takes roughly 15 s on a CPU with 64 × 64 patterns and up to a minute with 128 × 128. `RefineDefocusByAutofocus` (on by default) switches it in the GUI, `--defocus-estimate` in the reconstruction drivers and `defocus_estimate` in `ptychodus-stream`, where only each scan's final estimate holds the patterns autofocus needs.

Pressing the button on a product that already states a focus replaces it, reporting a disagreement in sign or beyond `ParallaxRtol` of the stated value — the check that catches a flipped detector or a mistyped focus. The settings-driven run never reaches that case, since it measures only while the focus is still unset.

Fitting a full matrix to the same shifts instead of one slope also measures the detector's orientation relative to the scan, since a flipped or transposed detector maps `q` onto the shifts through that flip; a half-turn of the detector looks exactly like a defocus of the opposite sign, so a reflection is unambiguous but a half-turn needs the defocus sign. `python scripts/check_detector_orientation.py` runs that check on any instrument's files and reports the `hflip`, `vflip` and `transpose` steps for both readings ({py:func}`ptychodus.api.parallax.estimate_parallax_transform`).

## Probe Mode Generation

Every generated probe type goes through the same `ProbeEnsembleBuilder.build` path:

1. Generate one base {py:class}`ptychodus.api.probe.Probe` and rescale its total intensity to `probe_photon_count`.
2. Expand to `num_incoherent_modes` with {py:func}`ptychodus.api.simulate.probe.generate_incoherent_probe_modes`.
3. Expand to `num_coherent_modes` OPR modes with {py:func}`ptychodus.api.simulate.probe.generate_coherent_probe_modes`.
4. Store the final result as a `ProbeEnsemble` with array shape `(num_coherent_modes, num_incoherent_modes, height, width)`.

### Incoherent Modes

The incoherent-mode controls are:

`num_incoherent_modes`
: Number of mutually incoherent modes to store in axis 1 of the final `ProbeEnsemble`.

`orthogonalize_incoherent_modes`
: If true and more than one incoherent mode is requested, the generated modes are orthogonalized with `scipy.linalg.orth`.

`incoherent_mode_decay_type`
: `none`, `polynomial`, or `exponential`.

`incoherent_mode_decay_ratio`
: Relative strength of later modes. ptychodus converts this to a list of unnormalized mode weights.

The low-level `generate_incoherent_probe_modes` routine takes a mode count and, optionally, an `IncoherentModeStrategy`. A strategy answers only two questions: what shape the new modes take, and how power would divide across a sequence of a given length. Everything else is the same for every strategy and belongs to the routine itself.

What the routine guarantees, whatever strategy is in play:

- Modes the probe already carries are **kept in order, by identity** — not merely their span — and the strategy supplies only the shortfall.
- Asking for fewer modes than the probe has keeps the strongest.
- Preserved modes keep the share of the power they arrived with; the new modes divide the share the strategy's profile assigns to their indices. A 90/10 two-mode probe expanded to four at a decay ratio of 0.5 comes out `[0.727, 0.073, 0.133, 0.067]`, so the 9:1 ratio survives.
- **Total power is unchanged**, expanding or truncating, and a mode dropped as linearly dependent passes its share to the modes that survive rather than leaving the probe dimmer. The number of modes used to represent a beam is a modeling choice; the illumination is not, and a probe read from file is never rescaled afterward.
- Orthogonalization is modified Gram-Schmidt in mode order, which is what keeps mode `n` paired with mode `n`. `mode_dependence_floor` sets how much of its own length a mode's orthogonal part must retain to be kept.

Which strategy runs is the `IncoherentModeStrategy` setting, one of `MomentPolynomial`, `Derivative`, `RandomPhaseRamp` or `GaussianSchell`; an unrecognized value falls back to the first. In the GUI it is a combo box, and the parameters below it change with the selection so that only the ones the chosen strategy reads are shown — the decay profile for the first three, the source size and coherence length for Gaussian-Schell, which predicts its own spectrum and never consults a decay profile.

`ProbeMomentPolynomialStrategy` is the default, and needs no physical parameters at all. It measures the intensity-weighted centroid and variance of the probe it is given — from the incoherent sum of its modes, which is the footprint the illumination actually covers — and uses that width as its own length scale, so one call suits any illumination without being told how the probe was formed. A new mode is the dominant mode times `u**m * v**n`, with `u` and `v` the transverse coordinates centered on the probe and measured in units of its rms width, damped by a Gaussian of that same width so the polynomial does not push the higher orders into the tails. Orders are graded by total degree and skip the constant term. Weights come from `ProbeModeDecayType` and depend on a mode's index rather than its order.

`ProbeDerivativeStrategy` fills with spatial derivatives of the dominant mode, `∂^(m+n) P / ∂x^m ∂y^n`, taken exactly by multiplying its spectrum by `kx^m ky^n`. A probe jittered by a small offset `d` is `P(r − d) ≈ P − d · ∇P`, so these are the modes positional vibration during an exposure produces; the first two are the x and y gradients. It is the Fourier counterpart of the moment polynomials, whose real-space factors correspond to derivatives in angle — source size and angular jitter rather than positional jitter. Orders are graded by total degree and weights come from `ProbeModeDecayType`.

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

Ptychodus uses the term `coherent modes` for the leading `ProbeEnsemble` dimension. In the reconstruction context this is the OPR mode dimension. The control is:

`num_coherent_modes`
: Number of OPR/coherent modes to store in axis 0 of the final `ProbeEnsemble`.

For `num_coherent_modes == 1`, no OPR weights are stored and the probe sequence length is one.

For `num_coherent_modes > 1`:

- `opr_weights` is initialized with shape `(num_positions, num_coherent_modes)`: one row per probe position, which the index matcher later conditions to one row per diffraction pattern.
- The first OPR weight is set to `1.0` for every probe position.
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
`COLLAPSE` applies the mean weight row across every incoherent mode. Asking a {py:class}`ptychodus.api.probe.ProbeEnsemble` for the probe at a position instead rewrites only incoherent mode 0 from the coherent basis.
```

### Inspecting an OPR Probe

{py:func}`ptychodus.api.probe.compute_opr_mode_series` reduces a probe ensemble to per-position diagnostics: the weights, the power in the composed mode, the power in its difference from the across-scan mean, and per-mode statistics. {py:attr}`ptychodus.api.probe.OPRModeSeries.relative_variation` reports the share of the composed mode's power that varies across the scan — near zero means the OPR basis is not modeling anything — and {py:attr}`ptychodus.api.probe.OPRModeSeries.effective_mode_count` reports how many coherent modes account for most of that variation, which is the number to compare against `num_coherent_modes`.

In the GUI this is **Analyze → Coherent (OPR) Modes...** on the probe panel. The dialog plays through the probe positions while plotting the counts and the per-mode weights against scan index, and maps either quantity over the scan grid. Because a few-percent variation is invisible against the composed probe on a fixed color scale, the display selector also offers the deviation from the across-scan mean — where a beam translation reads as a dipole and an intensity drift as a concentric change — and the coherent basis modes themselves. When the diffraction dataset is loaded, the measured per-frame photon counts are overlaid on the predicted curve, which is what distinguishes OPR tracking real flux drift from OPR absorbing noise.

### Examples

#### Generate a multimode probe from scratch

This example generates only the probe: a defocused Fresnel zone plate probe with three incoherent modes and two OPR/coherent modes. The OPR weights hold one row per probe position, so the number of positions is all it needs to know.

```python
import numpy

from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
    rescale_probe_intensity,
)
from ptychodus.api.constants import energy_eV_to_wavelength_m

num_positions = 400
pixel_m = 10e-9
probe_geometry = ProbeGeometry(
    width_px=128, height_px=128, pixel_width_m=pixel_m, pixel_height_m=pixel_m
)
zone_plate = FresnelZonePlate(
    zone_plate_diameter_m=180e-6,
    outermost_zone_width_m=50e-9,
    central_beamstop_diameter_m=60e-6,
)
probe = generate_fresnel_zone_plate_probe(
    probe_geometry,
    zone_plate,
    photon_wavelength_m=energy_eV_to_wavelength_m(10_000.0),
    defocus_distance_m=10e-6,  # the object 10 um downstream of the focus
)
probe = rescale_probe_intensity(probe, 1.0e6)
probe = generate_incoherent_probe_modes(probe, 3, orthogonalize=True)
probes = generate_coherent_probe_modes(
    numpy.random.default_rng(), probe, num_cmodes=2, num_positions=num_positions
)

assert probes.get_array().shape[:2] == (2, 3)
assert probes.get_opr_weights().shape == (num_positions, 2)
```

The same mode parameters work with other generated probe types such as `disk`, `rectangular`, `super_gaussian`, and `zernike`.

#### Add incoherent and OPR modes to an existing single-mode probe

The same mode-expansion routines augment a probe loaded from a file rather than replacing it with a generated one.

```python
from pathlib import Path

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import Probe
from ptychodus.api.simulate.probe import (
    generate_coherent_probe_modes,
    generate_incoherent_probe_modes,
)

# An NPY file is a bare array, so the run's own pixel size is stated alongside it.
base_probe = Probe(
    array=numpy.load(Path('single_mode_probe.npy')),
    pixel_geometry=PixelGeometry(width_m=10e-9, height_m=10e-9),
)
probe = generate_incoherent_probe_modes(base_probe, 3, orthogonalize=True)
probes = generate_coherent_probe_modes(
    numpy.random.default_rng(), probe, num_cmodes=2, num_positions=400
)

assert probes.get_array().shape[:2] == (2, 3)
```

The result is a `ProbeEnsemble` whose shape is `(2, 3, height, width)` and whose OPR weights have shape `(n_probe_positions, 2)`.

## Object Generators

Each object generator is a plugin, registered under the name below, and its parameters are the fields it declares; the settings file holds each under the same name in PascalCase with its unit in words (`phase_deviation_rad` is `PhaseDeviationInRadians` in `[Object]`). Phases are in radians throughout, as the `_rad` suffix says, so an object is `amplitude * exp(1j * phase_rad)`.

Object geometry is derived from the current probe geometry and scan bounding box. The generated object is then widened by `extra_padding_x` and `extra_padding_y` and scaled to the sample's measured mean transmission (see below); both apply to every generated object and to no file-supplied one.

`random`
: Generates a complex object from independent Gaussian amplitude and phase fields. This is the default object generator for a product with no diffraction data. Parameters are `amplitude_mean`, `amplitude_deviation`, `phase_deviation_rad`, and `blur_deviation_px`, the width of a Gaussian blur applied to both fields.

`gaussian_random_field`
: Generates a complex Gaussian random field by spectral synthesis. Parameter: `correlation_length_px`.

`fractal_noise`
: Generates a complex fractal-noise object by summing multiple octaves of simplex noise. Parameters are `grid_scale_px`, the grid scale of the first octave, `num_octaves`, `gain`, the amplitude ratio of each octave to the one before, and `lacunarity`, the factor each octave divides the grid scale by.

`dead_leaves`
: Generates an exact sample of the stationary dead-leaves model: disks dropped by a Poisson process, each hiding those beneath it, circular in physical space. Radii have density proportional to `r ** -leaf_radius_power_law_exponent` between `leaf_radius_lower_px` and `leaf_radius_upper_px`, in pixel widths; an exponent of three makes the texture scale invariant. Each disk's phase is uniform between `leaf_phase_lower_rad` and `leaf_phase_upper_rad`, and its intensity transmission uniform between the squares of `leaf_amplitude_lower` and `leaf_amplitude_upper`. Each pixel is the mean complex transmission over `num_subsamples_per_axis` squared jittered points. See [Training Data](training.md#the-dead-leaves-model) for its use as synthetic training data.

`siemens_star`
: Generates a Siemens star resolution target: `2 * num_spokes` wedges alternate between the spoke and background values inside a disk whose radius is `outer_radius_fraction` of the canvas's smaller physical half-extent, and pixels outside it take the background values. The star is circular in physical space, also on non-square pixels, and each pixel is the mean complex transmission over `num_subsamples_per_axis` squared jittered points, so wedge edges and the center are box filtered. Parameters are `num_spokes`, `outer_radius_fraction`, `spoke_amplitude`, `background_amplitude`, `spoke_phase_rad`, `background_phase_rad`, and `num_subsamples_per_axis`, which `dead_leaves` shares.

The four random textures draw from the application's random number generator, so each regeneration draws a fresh texture.

`stxm_dpc`
: Generates a coarse complex object from far-field diffraction data by STXM differential phase contrast. This is the default object generator for a product created with diffraction data. It requires assembled diffraction data, probe positions and far-field metadata. Near-field data gives a uniform object, with a warning, because the center-of-mass theorem it rests on relates a *far-field* centroid to the phase gradient of the exit wave. Parameters are `shadow_correction` and `regularization_fov_fraction`, described below.

- **Amplitude.** The per-position detector sums, mapped linearly between the positions and divided by their mean over the supported region, as STXM gives them. Detector sums have any detector background or flat field the data still carries applied and their bad pixels inpainted, and are corrected for beam decay by the I0 the probe positions carry, each scaled by `mean(I0) / I0_j`, so a drifting beam does not read as absorption. It reads them from the diffraction moments the dataset accumulates once per load, so regenerating it does not pass over the patterns again.
- **Support.** Without a probe, the supported region is the convex hull of the positions. With the product's probe on the object's pixels, it is where the probe's illumination reaches 5% of its peak, which follows the footprint past the outermost positions. The phase is integrated and referenced over it. The values themselves are always interpolated linearly: each is already the probe-averaged field at its position, so weighting the map by the probe again would blur it twice.
- **Phase.** Each pattern's intensity centroid, taken over the whole frame with its bad pixels inpainted, is the intensity-weighted mean phase gradient under the probe. The crop bounds both what is lost off the detector and how much high-angle shot noise enters, so data from an integrating detector with read noise should be cropped tightly. The mean centroid is subtracted, which removes the beam-center offset and any net phase wedge across the sample. The gradient is integrated by a least-squares FFT Poisson solve and referenced to zero over the most transparent 5% of the scan — the empty beam, when the scan includes one. Resolution is set by the probe size and the scan step.
- **Orientation.** The centroids are taken to lie in the frame the forward model simulates, with the detector's x and y along the scan's. A detector rotated or flipped relative to the scan should be oriented with the probe-position affine transform or the diffraction `hflip`, `vflip` and `transpose` steps, which the reconstruction needs in any case. With a defocused probe and a textured sample, `python scripts/check_detector_orientation.py`, a checkout tool that is not installed with the package, names those steps from the raw files (see [Measuring the Defocus](#measuring-the-defocus) above).
- **Probe shadow.** A centroid also measures the probe's own phase gradient, weighted by the sample's transmission. For a single material under a probe of RMS width σ defocused by Δz, ignoring that term rescales the phase by roughly `1 ± 4π(β/δ)σ²/(λ|Δz|)`, which is of order one for micron-scale defocused hard X-ray probes at moderate β/δ and nothing for a pure phase object. `shadow_correction` (on by default, as in the command-line drivers) subtracts a first-order estimate using the product's current probe; it matters for a defocused probe on an absorbing or strongly scattering sample and vanishes for a probe at focus. The probe sets the support and the shadow correction when the object is generated, and the object is not regenerated when the probe changes, so regenerate it after changing the probe.
- **Regularization.** Integrating noisy gradients leaves its error as drift across the field of view. `regularization_fov_fraction` adds a Tikhonov term that damps phase variations longer than that fraction of the object canvas's larger dimension, halving them at exactly that length. 0 disables it, and values beyond about 2 have little effect.

`paganin`
: Generates a complex object by Paganin single-material phase retrieval on the STXM intensity of the diffraction data, corrected for beam decay by I0 as `stxm_dpc`'s amplitude is. It requires assembled diffraction data. The wavelength and propagation distance are the product's; the one parameter is the sample's `delta_over_beta`. With the product's probe on the object's pixels, the mean intensity is taken over the probe's illumination rather than the convex hull of the positions. The object is not regenerated when the probe changes, so regenerate it after changing the probe.

`spectral`
: Generates a complex object by spectral refinement: a few power-method steps from the STXM-DPC estimate with the intensity-weighted covariance `Z = Σ_j Σ_k S_jk† diag(T(y_j) + c) S_jk`, where `S_jk` places incoherent probe mode `k` at position `j`, multiplies it into the object and propagates it to the detector, exactly as the forward model does. It reads every pixel of the assembled diffraction data, needs the product's probe on the object's pixels, and works in far and near field. Parameters are `start`, `shadow_correction`, `iterations`, `preprocessing`, `preprocessing_lower_bound`, `threshold_fraction` and `modulus`, described below. The operator is that of L. Valzania, J. Dong and S. Gigan, *Opt. Lett.* **46**, 1357 (2021), <https://doi.org/10.1364/OL.406156>.

- **Why a few steps, not the eigenvector.** The spectral estimate proper is `Z`'s leading eigenvector. On measured data with a focused probe it is the wrong target: `Z` acts locally on a canvas many probe widths across, its leading eigenvalues lie within about a percent of each other, and each of their eigenvectors concentrates on a different patch of the most strongly scattering features. Its modulus then measures scattering strength rather than transmission, which shows as a few hot regions: smoothed over the probe, its 99th percentile is tens to thousands of times its median. On nine measured far-field datasets with an average-pattern probe, LSQML started from the eigenvector had not recovered after a hundred epochs on six, and on three even when the eigenvector supplies only the phase and the start the modulus. A few steps from an estimate spread over the whole field resolve the local structure, whose spectral gaps are large, before the global competition, whose gaps are small, concentrates the estimate. `iterations` (3 by default) counts the steps. More sharpen detail where the probe is right and amplify its error where it is not; with gaps of a percent or less, the steps approach the eigenvector only after hundreds to thousands.
- **Start.** `stxm_dpc` (the default) starts from the STXM-DPC estimate, generated as `stxm_dpc` generates it without regularization, which supplies the low-resolution object; the refinement adds the detail finer than the probe. `shadow_correction` (on by default) removes the probe's own phase gradient from the start's centroids, as `stxm_dpc`'s option of that name does; without it, a defocused probe on an absorbing sample gives the refinement a start whose phase is rescaled, and the refinement keeps that error. STXM-DPC needs far-field data, so near field starts uniform, with a warning. `uniform` starts from the illuminated region, uniformly, and needs more steps to reach the same detail.
- **Modulus.** The estimate's phase is the refinement's. With `modulus` `start` (the default) its modulus is the start's smoothed over the probe's RMS radius: the low-resolution transmission STXM measured, without the stripes that fluctuating detector sums leave in it, and uniform when the refinement starts uniform. `estimate` keeps the refinement's own modulus, which carries an absorbing sample's transmission but also the scattering-strength bias above, and starts LSQML more slowly on a phase object.
- **Intensities.** As for `stxm_dpc`, any detector background or flat field the data still carries is applied as `(pattern - background) / flat_field` clipped at zero, and each pattern is corrected for beam decay by `mean(I0) / I0_j`. The bad pixels, widened by the pixels those maps make unusable, take zero weight, so module gaps and a masked beamstop drop out of `Z` exactly rather than being inpainted. The stack is then divided by its mean over every good pixel of every pattern. That keeps the absorption contrast between positions, which dividing each pattern by its own mean would discard. Generation waits for the diffraction moments, as `stxm_dpc` and `paganin` do, so it never runs on a dataset still loading.
- **Preprocessing.** `identity` (the default) weights every pixel by its intensity (E. J. Candès, X. Li and M. Soltanolkotabi, *IEEE Trans. Inf. Theory* **61**, 1985 (2015)), so the dark field, where the detail finer than the probe is recorded, takes part. `optimal` is `max(-β, 1 - 1/y)` with `β = preprocessing_lower_bound`, the preprocessing W. Luo, W. Alghamdi and Y. M. Lu prove uniformly optimal for the eigenvector of noiseless intensities, *IEEE Trans. Signal Process.* **67**, 2347 (2019), <https://doi.org/10.1109/TSP.2019.2904918>. Its weights are offset by `c = β` so none is negative, which makes the steps follow the largest eigenvalues rather than the most negative; every pixel below `1 / (1 + β)` of the mean then takes no weight, so the refinement slows as more of the detector lies in the dark field, and a 128 × 128 crop needs several times the steps a 64 × 64 one does. `threshold` sets the brightest `threshold_fraction` of good pixels to one and the rest to zero (S. Marchesini, Y.-C. Tu and H.-T. Wu, *Appl. Comput. Harmon. Anal.* **41**, 815 (2016)); the best fraction depends on how much of the detector the bright field covers, and finding it takes one temporary copy of the data.
- **Cost.** `Z` is never formed. Each step is one forward and one adjoint propagation per pattern and mode, batched over chunks of patterns, in single precision and accumulated in double. A far-field Fresnel transform needs one FFT pair per pattern and mode, its quadratic phases folded into the probe and the weights. Up to 1 GiB, the weights and then the probe shifted onto every position are computed once and kept; what does not fit is recomputed for every step. A default refinement of about a thousand 64 × 64 patterns with a four-mode probe takes a few seconds on a workstation CPU.
- **Normalization.** The global phase is fixed so the sum over the support — where the summed illumination reaches 5% of its peak — is real and positive. The modulus is scaled to a mean of one there, and pixels beyond the support are one, before the object is scaled to the measured transmission.
- **The probe.** The estimate is only as good as the probe it assumes, its defocus above all: on measured zone-plate data the detail it recovers falls from LSQML quality to little as the defocus moves a third of the way toward focus. A probe whose far-field intensity does not resemble the mean diffraction pattern, such as a disk where the beam is a zone-plate annulus, refines toward that probe's errors; the `average_pattern` probe reproduces the mean pattern by construction. The generator logs a warning when the far-field intensity the probe predicts correlates with the mean pattern below 0.5. The object is not regenerated when the probe changes, so regenerate it after changing the probe.

### Incident Photon Count and Transmission

The product's `probe_photon_count` and the mean transmission that generated objects are scaled to are estimated together from the diffraction moments the dataset accumulates once per load ({py:func}`ptychodus.api.accumulate.estimate_incident_beam`). Each pattern's detector sum covers its whole frame with the bad pixels inpainted. **I0** is the incident-beam reading — an upstream ion chamber, diode or scaler — that a position reader may record for each exposure, in whatever units the instrument uses. When I0 is recorded, every detector sum is first corrected for beam decay, scaled by `mean(I0) / I0_j`. Then one of three rules applies, and the log names it:

- **I0 with a calibration.** `I0PhotonsPerCount` (the product's **I0 Calibration [photons/count]**, or `--i0-photons-per-count` on the standard drivers) converts I0 to incident photons. The incident photon count is `factor * mean(I0)`, and the intensity transmission is `mean(detector_sum_j / (factor * I0_j))`. A transmission above about one or a vanishing fraction of it is reported, since it usually means a wrong calibration, but the calibration is kept.
- **I0 without a calibration.** The brightest corrected detector sums stand in for the empty beam: the incident photon count is their 0.99 quantile.
- **No I0.** The same, over the uncorrected detector sums.

Without a calibration the intensity transmission is the mean total over the incident photon count. Either way the mean amplitude transmission is its square root, clipped to `[0, 1]`. The photon count is only filled while it is zero, so a value from the user or a file is kept; **Estimate Probe Photon Count** in the product editor replaces it.

### Multislice Objects

Every object, generated or read from a file, is passed through `generate_layers`. If `object_layer_spacing_m` is empty, the object keeps the slices it already has — an unset spacing expresses no preference, so it never collapses a multislice warm start. If spacings are supplied, the requested number of slices is `len(object_layer_spacing_m) + 1` and the object is resliced onto that depth grid.

Reslicing is non-destructive. {py:func}`ptychodus.api.object.resample_object_layers` interpolates the cumulative complex optical path — the running sum of each layer's complex logarithm — and takes differences of it, so every input layer contributes, the layers still compose multiplicatively, and `get_layers_flattened()` is preserved to round-off whichever slice count is requested. Working in the logarithm is also what keeps a total phase outside `(-pi, pi]` correct: each layer's own phase is small enough not to wrap, and the sum accumulates past `pi` without ever passing through `numpy.angle`.

Expanding a single slice into several is the degenerate case, since there is no depth axis to resample along. It falls back to {py:func}`ptychodus.api.object.homogenize_object_layers`, which spreads the whole transmission evenly so that each of the `n_slices` layers is the `n_slices`-th root of the original.

The layers are phase screens at depths `0` through the total thickness, so `n_slices` layers leave `n_slices - 1` gaps and `Object.get_total_thickness_m()` — the sum of those gaps — is the full thickness. {py:func}`ptychodus.api.object.compute_uniform_layer_spacing_m` builds an even grid on that convention.

## Editing a Reconstruction Result

A reconstruction result's probe and object are kept exactly as solved, so their editors offer no mode or layer controls at first. **Make Editable** in either editor lets the result be conditioned like a generated one: the mode counts and the layer spacing start at what the result already has, so nothing changes until they are raised, and existing modes are kept when more are added. To keep the original, **Duplicate** the product first and make the copy editable. A product that is reconstructed into again reverts to the result as solved.

## Initial Guesses for cdtools

The **cdtools** reconstructor library (the `cdtools` extra) offers **FancyPtycho**, cdtools' far-field model. Unlike cdtools' own `from_dataset`, it generates no initial guess of its own: the probe, object and positions all come from the product, so any generator above applies. The `average_pattern` probe is the same estimate cdtools calls a SHARP-style probe, and the `stxm_dpc` object is a good initial object to pair with it.

- **Probe.** Every incoherent mode is used. cdtools models no orthogonal probe relaxation, so a probe with OPR coherent modes is reconstructed from its primary coherent mode alone, with a warning, and the result has none.
- **Object.** A single layer, on the probe's pixel grid. A multislice object is refused.
- **Canvas.** Every probe window must lie inside the object array. With `RefinePositions` on, a position can move only within the canvas, so leave room with `ExtraPaddingX` / `ExtraPaddingY` when generating the object.
- **Detector maps.** A background the data carries starts cdtools' additive background term (`UseBackground`), refined unless `RefineBackground` is off. A flat field multiplies the model (`UseFlatField`) and is never refined.
- **Scale.** The probe is measured against cdtools' forward model before the run and carried back after it, so the product it returns is in ptychodus's conventions and can warm-start another library.

## Low-Level Generator API

The low-level functions return arrays directly, for a caller that already has geometry objects, as the example at the top of this page does.

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
    photon_wavelength_m=1.239841984e-10,  # 10 keV
    defocus_distance_m=0.0,
)
base_probe = rescale_probe_intensity(base_probe, 1.0e6)

probe_with_imodes = generate_incoherent_probe_modes(
    base_probe,
    3,
    orthogonalize=True,
)
probe_ensemble = generate_coherent_probe_modes(
    rng,
    probe_with_imodes,
    num_cmodes=2,
    num_positions=len(positions),
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
    phase_mean_rad=0.0,
    phase_deviation_rad=0.6,
    blur_deviation_px=0.0,
)
```

## Using Generated Guesses With Pty-Chi

Ptychodus has a Pty-Chi adapter in `ptychodus.plugins.ptychi`. When using the normal Ptychodus reconstruction path, the adapter receives a {py:class}`ptychodus.api.reconstruct.ReconstructInput` together with the Pty-Chi options the settings hold, reconciles those options with the product, and passes the following data from the product to `PtychographyTask` as keyword arguments:

- object initial guess: `product.object_.get_array()`;
- probe initial guess: `product.probes.get_array()`;
- OPR weights: `product.probes.get_opr_weights()` if available, otherwise a one-dimensional default weight vector with first entry `1.0`;
- probe positions: physical Ptychodus positions are mapped through the Ptychodus object geometry to object pixel coordinates before being passed to Pty-Chi.

### Probe And Object Sampling

Pty-Chi allows the probe to be sampled differently from the object: `probe_options.pixel_size_m` and `probe_options.pixel_size_aspect_ratio` each fall back to the matching `object_options` value when left at `None`, and Pty-Chi Fourier-rescales the object onto the probe grid whenever the two disagree.

The adapter derives both fields from `product.probes.get_pixel_geometry()`, alongside the object fields it derives from `product.object_.get_pixel_geometry()`. When a probe value matches the object's it is left at `None` — stating the inheritance explicitly and keeping Pty-Chi's no-resampling path — and width and aspect ratio are decided independently, matching how Pty-Chi inherits them. A product whose probe carries no pixel geometry leaves both fields unset.

### Shape Compatibility

Pty-Chi's documented probe convention is `(n_opr_modes, n_modes, height, width)`:

- `n_opr_modes` is the OPR/eigenmode dimension;
- `n_modes` is the mutually incoherent probe-mode dimension.

This matches Ptychodus `ProbeEnsemble.get_array()`:

- Ptychodus `num_coherent_modes` maps to Pty-Chi `n_opr_modes`;
- Ptychodus `num_incoherent_modes` maps to Pty-Chi `n_modes`.

Pty-Chi requires a four-dimensional probe initial guess. Therefore, when calling Pty-Chi manually, pass `product.probes.get_array()` as the `probe_data` argument to `PtychographyTask` rather than indexing the probe sequence down to a single `Probe`.

Pty-Chi's planar object convention is `(n_slices, height, width)`, which matches Ptychodus `Object.get_array()`. Pty-Chi also expects multislice `slice_spacings_m` to contain `n_slices - 1` spacings, matching Ptychodus `Object.layer_spacing_m`.

### OPR Weights

Pty-Chi accepts OPR weights as either:

- `(n_probe_positions, n_opr_modes)`: per-position weights; or
- `(n_opr_modes,)`: one weight vector broadcast to every probe position.

Ptychodus generated OPR weights use `(n_probe_positions, n_coherent_modes)` when `num_coherent_modes > 1`. This is directly compatible with Pty-Chi.

If the probe has more than one OPR mode, Pty-Chi requires initial OPR weights. Ptychodus satisfies this when the OPR modes were generated by `generate_coherent_probe_modes`. If a custom `ProbeEnsemble` is created manually with multiple leading modes, provide matching `opr_weights`.

### Positions and Object Origin

Ptychodus stores probe positions in meters. Pty-Chi stores probe positions in object pixel units, with order `(y, x)`. The Ptychodus Pty-Chi helper converts positions by calling `object_geometry.map_coordinates_probe_to_object(probe_position)` and then passing the resulting pixel `x` and `y` arrays to `PtychographyTask` as `probe_position_x_px` and `probe_position_y_px`.

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
for probe_position in product.probe_positions:
    object_point = object_geometry.map_coordinates_probe_to_object(probe_position)
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
- `average_pattern`, `stxm_dpc`, `paganin` and `spectral` depend on assembled diffraction data. Analytic generators such as `disk`, `super_gaussian`, `fresnel_zone_plate`, `random`, `gaussian_random_field`, `fractal_noise`, `dead_leaves`, and `siemens_star` do not require measured patterns.
- `super_gaussian` is the closest built-in Gaussian-like probe. Ptychodus does not currently register a separate generator literally named `gaussian`.
- In options and options documents, use the generator parameter names, not the settings-file names. For example use `full_width_at_half_maximum_m` rather than `FullWidthAtHalfMaximumInMeters`.
