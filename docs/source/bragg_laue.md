# Bragg And Laue Ptychography

Ptychodus reconstructs 2D Bragg (reflection) and Laue (transmission) ptychography with the same transmission reconstructors it uses for everything else. Off the forward direction, the detector records the Fourier transform of the sample projected along the **exit** beam, not the incident one. That is ordinary far-field ptychography in an affine-transformed frame, the **exit frame**, so pty-chi and the other backends reconstruct it unchanged once the probe positions and the probe are expressed there. Ptychodus does that bookkeeping; this page describes it from the api, which every front end shares.

## Scope

The model is kinematic and thin-sample:

- far field only;
- a single object slice;
- a sample thin enough that the probe does not change across its depth, roughly `thickness × sin 2θ ≪ probe size`;
- both beams crossing the sample surface at a finite angle.

Dynamical diffraction in thick crystals (Pendellösung), full 3D Bragg ptychography over a rocking curve, multislice through a tilted crystal and near-field Bragg geometry are out of scope. Ptychodus warns when a product falls outside the model (see [Warnings](#warnings)).

The reconstructed phase is `-G · u`: the lattice displacement `u` projected onto the scattering vector `G`, not a refractive-index phase. See [Interpreting The Result](#interpreting-the-result).

## Frames And Conventions

Three planes are involved:

| Frame | Plane | What lives there |
| --- | --- | --- |
| Beam (lab) | Across the incident beam | The scan positions, as the instrument records them; a model probe's cross section. |
| Exit | Across the exit beam, on the detector's axes | The 2D Fourier relation: the detector intensity is `|FFT(O · P)|²` on the ordinary far-field grid `λz / (NΔ)`. |
| Surface | The sample surface | The physical sample. |

A thin sample's surface maps onto both other planes by projection, along the incident and along the exit beam, so positions move from the beam frame to the exit frame by a 2×2 linear map `T`.

{py:class}`ptychodus.api.scattering.geometry.ScatteringGeometry` states the directions in the frame that probe positions and oriented detector frames share in forward transmission: z along the incident beam, x along detector columns, y along detector rows.

| Field | Meaning |
| --- | --- |
| `two_theta_deg` | The detector arm swings about +y, taking the exit beam from +z toward +x. |
| `detector_gamma_deg` | Then elevates the exit beam toward +y. The detector rides the arm, its columns staying in the arm's horizontal plane. |
| `incidence_angle_deg` | The angle between the incident beam and the surface plane, turning in the same sense as two-theta. 90° is normal incidence. |

A mirrored instrument, whose arm swings toward −x, negates both `two_theta_deg` and `incidence_angle_deg`.

The geometry is forward when both detector angles are zero, and `T` is then the identity whatever the incidence. Transmission, Bragg or Laue is derived, as the `regime` property, never stored, so it cannot contradict the angles:

- **Transmission:** the exit beam is forward.
- **Bragg:** off axis, with both beams on the same side of the surface. Every Bragg `T` has a negative determinant, because a reflection mirrors the scan; symmetric Bragg gives exactly `diag(-1, 1)`.
- **Laue:** off axis, with the exit beam leaving through the far side. At normal incidence `T = diag(cos 2θ, 1)`.

```{note}
These angles are not `tomography_angle_deg` or `tilt_angle_deg`. Those orient a sample within a transmission measurement, and change neither the detector's direction nor the frame the positions are in. `tilt_angle_deg` in particular is the sample tilt APS 4-ID POLAR records for an XMCD projection: ptychodus stores and persists it, and no computation reads it. In a Bragg or Laue geometry, the sample's orientation is `incidence_angle_deg`.
```

## A Product Lives In Its Exit Frame

{py:attr}`ptychodus.api.product.ProductMetadata.scattering_geometry` defaults to forward. A product's positions, probe and object are all in the exit frame of its geometry. Lab-frame data is carried into that frame once, where it enters a product, and never again. Reconstructor output and products read from file are already in it.

One function decides where the transform applies: {py:func}`ptychodus.api.scattering.frame.resolve_exit_frame_transform`. It returns `T`, which {py:func}`ptychodus.api.scattering.frame.compute_exit_frame_transform` builds, for an off-axis far-field geometry with neither beam grazing the surface, and the identity, with a logged warning, everywhere else. Every place ptychodus builds a product calls it, so they all agree.

## Preparing Instrument Data

Two inputs arrive in the beam frame: the scan positions, and a model probe such as a zone plate. Both are carried into the exit frame.

A probe estimated from the patterns, such as `generate_average_pattern_probe`, is already in the exit frame the patterns define. So is a warm-start probe from an earlier product. Neither may be transformed.

```python
from ptychodus.api.affine import transform_probe_positions
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.scattering.frame import (
    resolve_exit_frame_transform,
    transform_probe_to_exit_frame,
)
from ptychodus.api.scattering.geometry import ScatteringGeometry
from ptychodus.api.simulate.object import generate_uniform_object

geometry = ScatteringGeometry(
    two_theta_deg=33.6, detector_gamma_deg=0.5, incidence_angle_deg=17.2
)

# Positions as the scan recorded them, carried into the exit frame.
transform = resolve_exit_frame_transform(geometry, far_field=True)
positions = ProbePositionSequence(list(transform_probe_positions(lab_positions, transform)))

# A zone-plate probe describes the beam's cross section, so it is resampled too.
probes = transform_probe_to_exit_frame(
    ProbeEnsemble.from_probe(zone_plate_probe), geometry, far_field=True
)

product = Product(
    metadata=ProductMetadata(
        name='bragg',
        comments='',
        detector_distance_m=1.1,
        photon_energy_eV=10400.0,
        probe_photon_count=5e8,
        exposure_time_s=0.5,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
        scattering_geometry=geometry,
    ),
    probe_positions=positions,
    probes=probes,
    object_=generate_uniform_object(compute_object_geometry(positions, probes.get_geometry())),
    losses=[],
)
reconstruct_input = prepare_reconstruct_input(assembled_data, product)
```

`reconstruct_input` goes to any transmission reconstructor unchanged. The object canvas is sized from the exit-frame positions, and its pixel is the ordinary far-field pixel. The surface pixel it corresponds to is larger along the surface's in-plane axis, by `1 / |sin(θ_i - 2θ)|` when `γ = 0`.

To simulate test data, {py:func}`ptychodus.api.simulate.diffraction.generate_scattering_data` takes a product stated in the physical frames — the object on the surface, the probe and positions across the incident beam — and returns the patterns with the exit-frame truth, tagged with its geometry.

## Warnings

{py:func}`ptychodus.api.scattering.geometry.find_scattering_issues` lists the ways a geometry falls outside the model. A forward geometry has none.

| Issue | What ptychodus does |
| --- | --- |
| `NEAR_FIELD` | The transform is derived for far field, so positions and probe are left untransformed and the reconstruction treats the data as transmission. |
| `MULTISLICE` | Positions are transformed, but multislice propagation along the incident beam ignores the crystal's tilt. |
| `GRAZING_INCIDENCE`, `GRAZING_EXIT` | A beam lies in the surface plane, the projection is degenerate, and positions are left untransformed. |

{py:func}`ptychodus.api.reconstruct.prepare_reconstruct_input` logs each issue as a warning, so every reconstruction path reports them. The run still proceeds.

## Interpreting The Result

Positions come back in the exit frame. `transform.inverted()` takes them back to the scan's lab frame exactly.

The phase is that of the crystal's envelope near `G`. {py:meth}`ptychodus.api.scattering.geometry.ScatteringGeometry.get_scattering_vector_per_m` gives `G = k (k_f - z)`, and {py:func}`ptychodus.api.scattering.geometry.compute_displacement_m` converts a phase to the displacement along it, `u_G = -φ / |G|`. A test pins the sign: displacing the scatterers in a direct kinematic sum reproduces the envelope `exp(-i G · u)`, not `exp(+i G · u)`.

Three ambiguities carry over from the phase:

- a global phase is a constant displacement;
- a linear phase ramp is a homogeneous strain or rotation, relative to the reference lattice whose Bragg condition is met at the detector center;
- the displacement wraps every `2π / |G|`, the lattice-plane spacing.

## Checking The Angles

The angles and their sign conventions can be checked without reconstructing, from the bright-field parallax of a defocused probe. With the positions as recorded, the fitted parallax matrix is proportional to `Tᵀ U`, with `U` the detector orientation. It is the transpose, not the inverse, because the probe's curvature belongs to the incident beam.

{py:func}`ptychodus.api.parallax.compute_geometry_mismatch` compares a fit from {py:func}`ptychodus.api.parallax.estimate_parallax_transform` with the matrix a geometry predicts, scale free. `scripts/check_detector_orientation.py` reports it for given angles, next to the forward assumption's:

```sh
python scripts/check_detector_orientation.py --patterns-type APS_34IDE \
    --patterns scan_0412/scan_0412_000001.tif --positions-type APS_34IDE \
    --positions mda_files/34ide_000412.mda \
    --two-theta-deg 33.6 --detector-gamma-deg 0.5 --incidence-angle-deg 17.2
```

`scripts/check_scattering_round_trip.py` simulates a scan, reconstructs it with and without the transform, and compares both with the truth.

## Persistence

The geometry is part of the external file interface: three scalars beside the other metadata.

| Format | Keys |
| --- | --- |
| `product.h5` | Root attributes `two_theta_deg`, `detector_gamma_deg`, `incidence_angle_deg` |
| NPZ product | Arrays of the same names |
| CXI | `ptychodus/two_theta_deg`, `ptychodus/detector_gamma_deg`, `ptychodus/incidence_angle_deg` in the entry |

A file written before the geometry existed reads as forward (0, 0, 90), so it is a transmission product as before. The standard CXI detector and sample orientation fields are not changed, and `diffraction.h5` carries no geometry.

## GUI And Command Line

- **Settings:** `[Products]` `TwoThetaInDegrees`, `DetectorGammaInDegrees` and `IncidenceAngleInDegrees`.
- **Product editor:** rows for the three angles and a read-only **Scattering Geometry** row. Its tooltip lists any issues. Editing an angle remaps the positions exactly, and a zone-plate or other model probe is rebuilt in the new exit frame.
- **Diffraction wizard:** imports the angles when a reader records them.
- **Scan plot:** labels its axes as the exit frame when a product is off axis.
- **Command line:** `ptychodus-reconstruct-26idc` and `ptychodus-reconstruct-34ide` accept `--two-theta-deg`, `--detector-gamma-deg` and `--incidence-angle-deg`. With `--prepare-only` they write a prepared directory whose `product.h5` carries the geometry, which `ptychodus-reconstruct` then honors.
