# Near-Field Ptychography

Near-field ptychography records Fresnel holograms rather than far-field diffraction patterns. The sample sits close enough to the detector that the beam footprint, not the diffraction envelope, sets what each frame covers, and a single frame already spans the full field of view.

Ptychodus treats the propagation regime as a property of the product, not a choice on the reconstructor. Declaring it is sufficient: the sampling, the propagation distance and the initial probe all follow from the declaration, and a reconstruction backend is handed a product that is already correct for the regime.

## The Regime Is A Product Property

`far_field` lives on the product metadata, stored as `[Products] FarField` and editable in the **Far Field** column of the Products table or the row of the same name in the product editor. There is no near-field algorithm and no propagator setting; both are consequences rather than inputs.

Declaring near field changes three things about the product itself:

- the object is sampled at the detector pitch back-projected through the beam, not through the Fraunhofer relation;
- the propagation distance becomes the finite `z_d / M` rather than infinity;
- `average_pattern` back-propagates with the angular spectrum, which preserves pitch.

```{warning}
Only one backend acts on the regime today. A near-field product handed to any other reconstructor is reconstructed with a far-field forward model, silently and without a warning, because the product carries correct sampling either way. See [Reconstructing With pty-chi](#reconstructing-with-pty-chi).
```

## Two Inputs And One Derived Distance

Everything below is in the beamline frame: **+z downstream, origin at the object**. The geometry needs two numbers from you; the third is computed.

| Symbol | Field | What it is |
| --- | --- | --- |
| `z_d` | `detector_distance_m`, `[Products] DetectorDistanceInMeters` | Object to detector. Positive downstream. An input. |
| `z_f` | `focus_object_distance_m`, `[Products] FocusObjectDistanceInMeters` | **Signed position of the beam focus.** An input. |
| `z` | `object_plane_propagation_distance_m` | `z_d / M`. Derived, never entered. The distance a reconstruction propagates. |

### The Sign Is The Geometry

`focus_object_distance_m` reads like an unsigned "distance from the focus to the object", but it is a *signed coordinate*. The ordinary magnifying setup — focus upstream of the sample, diverging cone — is **negative**.

This matters because a sign error produces a plausible number rather than an error. Writing `+0.001` for a focus 1 mm upstream describes a *converging* beam instead, giving `M = 999` where the truth is `1001`: no warning, and a reconstruction on a sampling that is wrong by 0.2%.

The cases are:

- `z_f < 0` — focus upstream, object in a diverging beam: `M = (|z_f| + z_d) / |z_f|`, magnifying.
- `z_f > 0` — focus downstream, object in a converging beam that crosses over before the detector: `M = (z_d - z_f) / z_f`.
- `z_f = 0` — no focusing optic, `M = 1` exactly. The sentinel cannot collide with a real value, since a focus in the object plane leaves the magnification undefined.

Both signed cases reduce to `M = abs((z_d - z_f) / z_f)`.

The **Magnification** and **Object-Plane Propagation Distance** rows of the product editor sit directly below **Focus-Object Distance**, so a wrong sign is visible as a magnification you did not expect rather than only as a pixel size three orders of magnitude from anything familiar.

### The Focus Is Stated Twice

A focusing probe builder states the same displacement in the opposite frame. `fresnel_zone_plate` and `kb_mirror` propagate `focal_length_m + defocus_distance_m` to reach the object plane, so their `defocus_distance_m` is how far *past* the focus the sample sits:

```text
defocus_distance_m == -focus_object_distance_m
```

A product that declares a focus wins over the builder's own value, and a disagreement is reported rather than resolved silently. A product that leaves the focus at zero declares no focusing optic, which cannot describe a focused probe, so the builder's value is used.

The aperture builders have no focus at all. `disk`, `rectangular` and `average_pattern` read `propagation_distance_m`, a free propagation distance applied after the field is formed.

On the command line the same identity is a default: `--near-field` alone derives the focus distance from `--fzp-defocus-m`, so an instrument whose users already enter the defocus need not enter it twice in two conventions.

```{note}
Attach a negative value with `=`. `--focus-object-distance-m=-1e-3` works and so does `--focus-object-distance-m -0.001`, but argparse reads a bare `-1e-3` as an option name rather than a number.
```

### Why `z` Is Not `z_d`

A cone beam magnifying by `M` images exactly like a parallel beam propagating `z_d / M` onto pixels `dx_det / M`. That is the Fresnel scaling theorem. A reconstruction runs in that equivalent parallel-beam geometry, which is why the propagation distance a backend is given is `z_d / M` rather than the lab-frame distance.

The figures below are illustrative, rounded from a detector of 512 pixels at 75 µm sitting 2 m from the sample at 10 keV — the rough scale of a zone-plate instrument, not a configuration to copy. Yours will differ; the identities will not.

| Case | `z_f` | `M` | `z = z_d / M` | Object pixel |
| --- | --- | --- | --- | --- |
| Parallel beam, no optic | `0` | `1` | 2 m, the detector distance | 75 µm, the detector pitch |
| Focus 1 mm **upstream**, diverging | **−1 mm** | ≈ 2000 | ≈ 1 mm | ≈ 37 nm |
| Focus 1 mm downstream, converging | +1 mm | ≈ 2000 | ≈ 1 mm | ≈ 37 nm |

The two signed cases land within a part in a thousand of each other, which is the point of the warning above: at this scale a sign error is a 0.1% error in the sampling, not an obvious one. The exact magnifications are 2001 and 1999.

Far field only cares about `z_d`. The magnification cancels there — the equivalent geometry scales pixel and distance by the same factor in `lambda z / (N dx)` — so the far-field branch is deliberately passed the lab-frame distance. The Fresnel number is *not* made invariant, and should not be: a focusing optic really does move the geometry toward near field.

`far_field` and `magnification` are independent. An optic says where the waist sits, not which regime the detector samples; far field with an optic and near field without one both occur. The reported **Fresnel Number** is the regime check, much greater than one being near field.

## Which Propagator Runs

The propagator is selected from the geometry. Ptychodus exposes no switch, and a near-field geometry lands on the angular spectrum in every realistic case, which is exact at every Fresnel number and preserves pitch. That is *why* the object pixel equals the detector pixel in the parallel-beam case, rather than it being a separate convention.

Two selections exist for two purposes, and conflating them is the easy mistake. Ptychodus chooses a propagator for probe simulation and the propagation-analysis panel; a reconstruction backend chooses its own, internally. The crossover is the same in both — exactly so since pty-chi 2.2.0, which brought its threshold into line with the one `choose_propagator` already applied — and the threshold is small enough that near field stays clear of it. [Reconstructing With pty-chi](#reconstructing-with-pty-chi) gives the measured numbers for that backend.

## Setting It Up In The GUI

You supply the quantities the instrument measures — detector distance, photon energy, the focus position, and the detector pixel size the file usually already carries. Ptychodus derives the rest: the magnification, the object pixel size and the propagation distance are computed from those and displayed, never entered.

In navigation-bar order:

Diffraction
: Load the dataset. The detector pixel size comes from the file as it does for a far-field scan, and the wizard's *Detector Pixel Size* group is there for a layout that records none. A reader that knows the regime also fills the **Propagation Regime** and **Focus-Object Distance** rows of the metadata page; check them to apply what the file declared.

: Near field raises the stakes on that pixel rather than changing who provides it. It *is* the reconstruction resolution here, so a file that records a raw sensor pitch where the beam actually saw an effective one — a scintillator and visible-light objective being the usual reason the two differ — puts every derived quantity out by that ratio. It shows up immediately in the Products rows below, which is the point of reading them.

Products
: Enter **Detector-Object Distance** and **Photon Energy**, and set the regime in the **Far Field** column. Double-click the product for the editor, where **Focus-Object Distance [mm]** lives; leave it at `0` for a parallel beam, or give the signed focus position for a cone beam.

: Then read the accounting back off the derived rows. **Magnification** should be `1` for a parallel beam and should match the optic for a cone beam; **Object Plane Pixel Width** should equal the detector pixel divided by that magnification; **Object-Plane Propagation Distance [mm]** should be the detector distance divided by it; and **Fresnel Number** should be large. Four numbers you did not type, each one a check on one you did.

Probe
: `average_pattern` is the regime-aware builder and the right cold start for flat bright-field illumination. A focusing-optic builder is not wrong in itself — a defocused zone plate is exactly what a cone-beam instrument needs — so choose by the illumination, not by the regime. `rectangular` and `disk` work as a flat cold start.

Object
: `random` is fine. Near field needs little padding, because one frame already covers the full field of view.

Processing
: Choose a library that implements a near-field forward model, and an algorithm it offers. See [Reconstructing With pty-chi](#reconstructing-with-pty-chi) for the one that does so today.

## From The Command Line

Every beamline driver takes the regime as a flag:

```sh
ptychodus-reconstruct-lamni --near-field \
  --diffraction-file scan.h5 --position-file scan.dat --output-directory out/
```

With no `--focus-object-distance-m`, the focus is derived from the driver's zone-plate defocus, which states the same displacement in the opposite frame. The batch drivers take the same two values in a settings file:

```yaml
metadata_overrides:
  far_field: false
  focus_object_distance_m: -0.001  # optional; defaults to the negated fzp_defocus_m
```

## Reconstructing With pty-chi

pty-chi is currently the only backend with a near-field forward model, so it is the only one that acts on the regime. The others receive a product whose sampling is correct and reconstruct it as though the detector were in the far field, which fails as a poor image rather than as an error.

Nothing below is configured by hand. A near-field product resolves it all on the way into the backend, and a value you set that contradicts the product is reported rather than silently honoured.

| Option | Near field | Why |
| --- | --- | --- |
| `free_space_propagation_distance_m` | `z_d / M` | The equivalent parallel-beam distance, finite rather than infinite. |
| `fft_shift` | cleared | The measurement is real-space intensity; there is no DC corner to roll. Far field sets it instead, for the plain FFT it compares against. |
| `forward_model_options.pad_for_shift` | raised from `0` to `50` | A bright-field probe wants a margin to shift into. Only the default is replaced, never a value you chose. |

### Algorithm

Pick **BH**, which reconstructs a hologram better than stock LSQML. The GUI default is the **Ptychodus** library's **Kickstart**, which estimates from far-field data only, so selecting PtyChi and BH is a manual step. The beamline drivers reconstruct with pty-chi and fall back to BH on their own for a near-field run given no `--ptychi-options` file.

Near-field scans are often few enough frames to run as a single batch, where **BH Method:** `CG` converges faster than the default `GD`.

A high-count hologram suits the Gaussian noise model, which is pty-chi's default. Poisson buys nothing here and costs conditioning.

### Propagator

Inside the reconstruction pty-chi picks `FourierPropagator` at infinite distance, and otherwise asks, per axis, whether the conjugate-plane pitch `lambda |z| / (N dx)` is coarser than the source pitch — taking the Fresnel transform when either axis says yes and the angular spectrum when neither does. For square pixels that reduces to the per-pixel Fresnel number `dx**2 / (lambda |z|)` against `1 / N`. The distance enters as a magnitude, so propagating backward selects the same propagator as propagating forward. For the illustrative instrument above:

| Geometry | Per-pixel `Fr` | Threshold `1 / N` | Selected |
| --- | --- | --- | --- |
| Parallel beam, 75 µm over 2 m | ≈ 23 | ≈ 2e-3 | angular spectrum |
| Cone beam, 37 nm over 1 mm | ≈ 0.011 | ≈ 2e-3 | angular spectrum |
| Far field | — | — | `FourierPropagator`, distance is infinite |

The parallel-beam case clears the threshold by four orders of magnitude. The cone beam clears it by about six times, which is comfortable but finite: the equivalent parallel-beam geometry divides the pixel and the distance by the same magnification, so `Fr` falls as `1 / M`, and a high enough magnification does reach the Fresnel transform — on this instrument near an `M` of `1.2e4`. That is an order of magnitude beyond a zone-plate cone beam, and well inside the regime where the conjugate grid is coarser than the source grid and the sampling would be in question anyway.

```{warning}
The **Forward Model:** combo in the Autodiff panel is not a propagator selector. It maps to pty-chi's `ForwardModels` enum, whose members are `base` and `planar_ptychography`, and it appears on that one algorithm's page only.
```

### Supplying Your Own Options

`--print-ptychi-options` names the fields ptychodus always derives from the data, which an options file must leave alone. `fft_shift` is among them: it states where the detector plane keeps its DC component, which the regime settles in both directions. `pad_for_shift` is not, because it is derived only for a near-field product; set it for a far-field run as you like.

## Known Limitations

Multi-distance holo-ptychography is not supported
: `detector_distance_m` is a scalar on the product, and the backends carry a single propagation distance to match. One distance per reconstruction; combining several detector planes would need a forward model that accepts more than one.

Affine position correction rejects a cone beam
: `transform_product` raises for a product whose magnification is not one.

Near-field *simulation* is exact only at `M == 1`
: The forward model does not normalize amplitudes by `1 / M**2`, so simulated counts scale with the magnified area, and `layer_spacing_m` is taken in the object frame rather than rescaled into the equivalent parallel-beam geometry.

Only one backend implements near field
: Any other reconstructor given a near-field product reconstructs it with a far-field forward model and reports nothing, because the product it receives is well-formed either way.

A near-field dataset left declared far field is not detected
: The regime-disagreement warning cannot catch it. The far-field branch derives a self-consistent object grid, so the Fresnel number it computes comes back small rather than large. The visible check is the **Object Plane Pixel Width** row: a value orders of magnitude below the detector pitch means the regime is wrong.
