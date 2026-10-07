"""Wavefield propagation models and associated parameter containers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
import math

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy
import numpy.typing

from .constants import TWO_PI_J, energy_eV_to_J, energy_eV_to_wavelength_m
from .typing import ComplexArrayType, RealArrayType
from .geometry import ImageExtent, PixelGeometry

_InexactT = TypeVar('_InexactT', bound=numpy.inexact[Any])


def intensity(wavefield: ComplexArrayType) -> RealArrayType:
    """Return the element-wise intensity (``|wavefield|²``) of a complex array."""
    return numpy.square(numpy.absolute(wavefield))


def _cut_central_row(
    array: numpy.typing.NDArray[_InexactT],
) -> numpy.typing.NDArray[_InexactT]:
    """Average the two central rows of a ``(num_steps, height, width)`` stack, then
    transpose to ``(width, num_steps)``.

    For an odd height the two indices coincide and nothing is averaged. The average is
    taken on whatever was passed in, so an intensity stack averages powers while a
    complex wavefield averages phasors.
    """
    size = array.shape[-2]
    cut_lower = array[:, (size - 1) // 2, :]
    cut_upper = array[:, size // 2, :]
    return numpy.transpose(numpy.add(cut_lower, cut_upper) / 2)


def _cut_central_column(
    array: numpy.typing.NDArray[_InexactT],
) -> numpy.typing.NDArray[_InexactT]:
    """Average the two central columns of a ``(num_steps, height, width)`` stack, then
    transpose to ``(height, num_steps)``. See :func:`_cut_central_row`."""
    size = array.shape[-1]
    cut_lower = array[:, :, (size - 1) // 2]
    cut_upper = array[:, :, size // 2]
    return numpy.transpose(numpy.add(cut_lower, cut_upper) / 2)


def compute_far_field_pixel_geometry(
    pixel_geometry: PixelGeometry,
    extent: ImageExtent,
    *,
    wavelength_m: float,
    propagation_distance_m: float,
) -> PixelGeometry:
    """Pixel geometry of the conjugate plane under the Fraunhofer relation
    ``dx_out = lambda |z| / (N dx_in)``.

    The single-FFT propagators (:class:`FresnelTransformPropagator`,
    :class:`FraunhoferPropagator`) map a plane of pitch ``dx_in`` onto a plane of this
    pitch. The relation is its own inverse: applying it twice returns *pixel_geometry*.

    Raises:
        ZeroDivisionError: when the input plane has zero extent along either axis.
            Callers that must degrade gracefully catch it rather than receiving an
            invented sentinel.
    """
    # Python-float arithmetic throughout: a numpy intermediate would yield inf instead
    # of the ZeroDivisionError that callers rely on.
    numerator_m2 = wavelength_m * abs(propagation_distance_m)
    return PixelGeometry(
        width_m=numerator_m2 / (extent.width_px * pixel_geometry.width_m),
        height_m=numerator_m2 / (extent.height_px * pixel_geometry.height_m),
    )


def compute_near_field_pixel_geometry(
    pixel_geometry: PixelGeometry, *, magnification: float
) -> PixelGeometry:
    """Pixel geometry of the object plane under the geometric projection ``dx_out = dx_in / M``.

    A cone beam of magnification ``M`` projects the detector pixels back onto the object
    demagnified by ``M``, so this is pure projection geometry: it involves no diffraction
    and therefore takes neither a wavelength nor a propagation distance, unlike
    :func:`compute_far_field_pixel_geometry`. Without a focusing optic ``M`` is 1 and the
    result is the input unchanged -- the defining property of the near-field regime, where
    the object grid and the detector grid coincide.

    Each axis is projected independently, so an anisotropic detector pixel stays
    anisotropic.

    Raises:
        ZeroDivisionError: at zero magnification, which places the detector at the focus
            where the projection is undefined. Callers that must degrade gracefully catch
            it rather than receiving an invented sentinel.
    """
    # Python-float arithmetic throughout, for the reason given in
    # compute_far_field_pixel_geometry: a numpy intermediate yields inf rather than
    # raising, and callers rely on the exception.
    return PixelGeometry(
        width_m=pixel_geometry.width_m / magnification,
        height_m=pixel_geometry.height_m / magnification,
    )


def compute_full_aperture_fresnel_number(
    pixel_geometry: PixelGeometry,
    extent: ImageExtent,
    *,
    wavelength_m: float,
    propagation_distance_m: float,
) -> float:
    """Full-aperture Fresnel number ``W H / (lambda z)`` of a plane of this pitch and extent.

    The propagation-regime indicator: much less than one is far field, near one is
    transitional, much greater than one is near field.

    Distinct from :attr:`PropagatorParameters.pixel_fresnel_number_x`, the signed
    per-pixel quantity ``dx^2 / (lambda z)`` taken along the width axis alone; the two
    differ by a factor of ``width_px * height_px * (dy / dx)``, the ``dy / dx`` arising
    precisely because that one is width-only. This number spans both axes instead,
    since an aperture has no preferred one.

    The distance is used as given rather than as a magnitude, so the two agree on sign
    convention. That convention carries less weight here: a regime indicator has no
    direction, and this is only ever evaluated at a positive distance, whereas the sign
    of the per-pixel number is load-bearing in the propagator phases.

    Raises:
        ZeroDivisionError: at zero wavelength or zero propagation distance, where the
            number is undefined.
    """
    width_m = extent.width_px * pixel_geometry.width_m
    height_m = extent.height_px * pixel_geometry.height_m
    return width_m * height_m / (wavelength_m * propagation_distance_m)


def compute_far_field_propagation_distance(
    pixel_geometry: PixelGeometry,
    extent: ImageExtent,
    *,
    wavelength_m: float,
    conjugate_pixel_width_m: float,
) -> float:
    """Propagation distance implied by a known conjugate-plane pitch.

    Solves ``dx_out = lambda |z| / (N dx_in)`` for ``|z|``, inverting
    :func:`compute_far_field_pixel_geometry` along the width axis. A format that
    records the sample-plane pixel size but not the sample-to-detector distance --
    which is how the fold_slice preprocessing step stores its geometry -- pins the
    distance this way, and the result reproduces `conjugate_pixel_width_m` when fed
    back through the forward relation.

    The width axis alone: the conjugate pitch is one number, so an anisotropic
    detector could not satisfy both axes at once, and the two agree wherever the
    detector is square.

    Raises:
        ZeroDivisionError: when the wavelength is zero, since every distance then
            maps to the same conjugate pitch and the inverse is undefined.
    """
    return conjugate_pixel_width_m * extent.width_px * pixel_geometry.width_m / wavelength_m


def compute_magnification(detector_distance_m: float, focus_object_distance_m: float) -> float:
    """Cone-beam magnification implied by the focus and detector positions.

    `focus_object_distance_m` is a signed coordinate in the beamline frame --
    downstream is +z with the origin at the object -- so its sign selects the
    geometry. Negative puts the focus upstream and the object in a diverging beam,
    giving ``(|z_f| + z_d) / |z_f|``; positive puts the focus downstream, so the
    object sits in a converging beam that crosses over before the detector, giving
    ``(z_d - z_f) / z_f``. Both reduce to ``|(z_d - z_f) / z_f|``.

    Zero means no focusing optic and yields 1.0, leaving parallel-beam geometry
    unchanged. The sentinel cannot collide with a real value: it would place the
    focus in the object plane, where the magnification is undefined.

    Sibling of :func:`compute_far_field_pixel_geometry`: both map detector-plane
    sampling onto the object plane, that one through the far-field reciprocal
    relation and this one through the geometric projection of a cone beam.
    """
    if focus_object_distance_m == 0.0:
        return 1.0

    return abs((detector_distance_m - focus_object_distance_m) / focus_object_distance_m)


@dataclass(frozen=True)
class ProductGeometry:
    """Quantities derived from a product's beam parameters and detector sampling.

    Produced by :func:`compute_product_geometry`. Every field is a pure function of
    stored product metadata and the detector geometry, so none of it needs persisting
    alongside a product -- recomputing is cheaper than keeping a second copy in step.

    Degenerate inputs report the true limit rather than a placeholder, since a product
    is routinely inspected before a diffraction dataset is bound to it. A quantity whose
    limit diverges is ``inf``; one whose degenerate form is genuinely indeterminate --
    ``0/0``, as for an unrecorded photon count over an unrecorded exposure -- is ``nan``.
    Two fields are deliberate exceptions to that rule for the reasons their own
    docstrings give: :attr:`object_plane_pixel_geometry` and :attr:`fresnel_number`.
    """

    probe_wavenumber_per_m: float
    """Reciprocal wavelength, ``1 / lambda``.

    Zero at zero photon energy. That is the limit rather than a guard: the wavenumber is
    proportional to the energy, so it vanishes with it.
    """
    probe_angular_wavenumber_rad_per_m: float
    """``2 pi`` times :attr:`probe_wavenumber_per_m`."""
    probe_photon_flux_per_s: float
    """Incident photons per second, the photon count over the exposure time.

    Infinite for a nonzero count over a zero exposure. ``nan`` when both are zero, which
    is the state of a product whose flux has not been recorded -- unmeasured is unknown,
    not zero.
    """
    probe_power_W: float  # noqa: N815
    """Beam power: the photon energy times :attr:`probe_photon_flux_per_s`.

    Inherits that field's ``inf`` and ``nan``, and is itself ``nan`` for an infinite flux
    at zero energy, where the product is indeterminate.
    """
    object_plane_propagation_distance_m: float
    """Propagation distance of the equivalent parallel-beam geometry, ``z_d / M``.

    A cone beam magnifying by ``M`` images like a parallel beam propagating this much
    shorter distance onto pixels this much smaller, which is the pairing
    :attr:`object_plane_pixel_geometry` applies. Equals the detector distance whenever
    there is no focusing optic.

    Infinite at zero magnification, which places the detector at the focus. The
    numerator cannot vanish alongside it, so the quotient genuinely diverges there.
    """
    object_plane_pixel_geometry: PixelGeometry
    """Sample-plane sampling implied by the detector and the declared regime.

    Far field samples the Fraunhofer reciprocal relation; near field is the geometric
    back-projection of the detector pixels through the cone, which without a focusing
    optic leaves them unchanged. The regime is taken as declared rather than inferred
    from the magnification: the two are independent, and a focusing optic constrains
    neither.

    Magnification-invariant in the far field: the equivalent parallel-beam geometry
    scales the pixel by the same factor as the distance, so the two cancel.

    **Exception to the limit rule.** A degenerate geometry gives ``PixelGeometry(0, 0)``
    rather than a divergent pitch, because zero on either axis is the sentinel
    :attr:`PixelGeometry.is_valid` tests for and callers branch on. An infinite pitch
    would read as valid and propagate into object geometries built from it.
    """
    fresnel_number: float
    """Full-aperture Fresnel number ``W H / (lambda z)`` at the **object** plane.

    The propagation-regime indicator: much less than one is far field, near one is
    transitional, much greater than one is near field. The detector-plane aperture
    number is its reciprocal up to the pixel count -- ``Fr_detector * Fr_object ==
    width_px * height_px`` exactly -- so reporting the detector plane would read large
    precisely when the geometry is deeply far field.

    ``z`` is :attr:`object_plane_propagation_distance_m`, so the indicator stays
    meaningful when a focusing optic magnifies the geometry. Unlike the pitch above
    this is *not* magnification-invariant, and should not be made so: the object extent
    is fixed while the equivalent distance shrinks, so a focusing optic really does move
    the geometry toward near field.

    **Exception to the limit rule.** A degenerate geometry gives ``0.0``. The case is a
    path-dependent ``0/0``, and along the far-field path it is a genuine limit: the
    object-plane width is ``lambda z / dx_d``, so ``W^2 / (lambda z) = lambda z / dx_d^2
    -> 0`` as ``z -> 0``.
    """
    detector_numerical_aperture: float
    """Collection half-angle the detector subtends at the sample.

    The geometric mean over the two axes, ``sqrt((W / 2 z_d) (H / 2 z_d))``, under the
    small-angle approximation ``sin theta ~ tan theta ~ theta``. Magnification-invariant,
    by the same cancellation as :attr:`object_plane_pixel_geometry`.

    This is the *collection* aperture. It is not the convergence aperture of a focusing
    optic, which is independent of it and which the optic models publish themselves as
    ``get_numerical_aperture``. Nor does it bound the achievable resolution on its own:
    ptychography reconstructs from the synthetic aperture the two combine into.
    """
    depth_of_field_m: float
    """Single-slice criterion ``lambda / NA^2``, over :attr:`detector_numerical_aperture`.

    The propagation depth across which the object may be treated as one thin slice; a
    sample thicker than this needs a multislice reconstruction.

    Infinite as the aperture vanishes, which is the limit rather than a guard, and
    ``nan`` when the photon energy is zero as well, where the ratio is indeterminate.
    """


def compute_product_geometry(
    *,
    probe_energy_eV: float,  # noqa: N803
    probe_photon_count: float,
    exposure_time_s: float,
    detector_distance_m: float,
    focus_object_distance_m: float = 0.0,
    far_field: bool = True,
    detector_extent: ImageExtent | None = None,
    detector_pixel_geometry: PixelGeometry | None = None,
) -> ProductGeometry:
    """Derive the beam and sampling quantities implied by a product's metadata.

    Takes the metadata fields individually rather than a product object, so that a
    caller holding them as settings parameters or as database columns need not
    assemble one first.

    The detector arguments describe the assembled patterns and are optional: omitting
    them stands for no bound diffraction dataset, and the fields that need a detector
    degrade as :class:`ProductGeometry` describes. This function owns that degradation
    policy -- the primitives it composes keep raising :exc:`ZeroDivisionError` rather
    than inventing sentinels.
    """
    # Degenerate inputs resolve to the true limit: inf where a quotient diverges, nan
    # where it is 0/0. The two exceptions are called out where they arise below.
    wavelength_m = energy_eV_to_wavelength_m(probe_energy_eV)
    extent = ImageExtent(width_px=0, height_px=0) if detector_extent is None else detector_extent
    pixel_geometry = (
        PixelGeometry(width_m=0.0, height_m=0.0)
        if detector_pixel_geometry is None
        else detector_pixel_geometry
    )

    try:
        wavenumber_per_m = 1.0 / wavelength_m
    except ZeroDivisionError:
        # Zero energy. The limit, not a guard: the wavenumber is proportional to the
        # energy, so it vanishes with it.
        wavenumber_per_m = 0.0

    try:
        photon_flux_per_s = probe_photon_count / exposure_time_s
    except ZeroDivisionError:
        # Indeterminate when nothing was recorded at all, which is the default product.
        # With a real count the flux genuinely diverges as the exposure vanishes.
        photon_flux_per_s = (
            math.nan if probe_photon_count == 0.0 else math.copysign(math.inf, probe_photon_count)
        )

    magnification = compute_magnification(detector_distance_m, focus_object_distance_m)

    try:
        propagation_distance_m = detector_distance_m / magnification
    except ZeroDivisionError:
        # Zero magnification places the detector at the focus. It requires the focus
        # distance to equal a nonzero detector distance, so the numerator cannot vanish
        # alongside the denominator and the quotient diverges.
        propagation_distance_m = math.copysign(math.inf, detector_distance_m)

    try:
        if far_field:
            # Fresnel scaling: a cone beam of magnification M images like a parallel
            # beam propagating z_d / M onto pixels dx_d / M. Both scale, so M cancels in
            # lambda z / (N dx) and the lab-frame distance is the one to pass here.
            # The Fresnel number below is deliberately not invariant -- see its docstring.
            object_plane_pixel_geometry = compute_far_field_pixel_geometry(
                pixel_geometry,
                extent,
                wavelength_m=wavelength_m,
                propagation_distance_m=detector_distance_m,
            )
        else:
            object_plane_pixel_geometry = compute_near_field_pixel_geometry(
                pixel_geometry, magnification=magnification
            )
    except ZeroDivisionError:
        # Exception to the limit rule: zero on either axis is the sentinel is_valid
        # tests for. See ProductGeometry.object_plane_pixel_geometry.
        object_plane_pixel_geometry = PixelGeometry(width_m=0.0, height_m=0.0)

    try:
        fresnel_number = compute_full_aperture_fresnel_number(
            object_plane_pixel_geometry,
            extent,
            wavelength_m=wavelength_m,
            propagation_distance_m=propagation_distance_m,
        )
    except ZeroDivisionError:
        # Exception to the limit rule: a path-dependent 0/0 that is a genuine limit
        # along the far-field path. See ProductGeometry.fresnel_number.
        fresnel_number = 0.0

    two_z_m = 2.0 * detector_distance_m
    detector_area_m2 = (extent.width_px * pixel_geometry.width_m) * (
        extent.height_px * pixel_geometry.height_m
    )

    try:
        numerical_aperture_sq = detector_area_m2 / (two_z_m * two_z_m)
    except ZeroDivisionError:
        # A detector in the sample plane subtends everything, so the aperture diverges;
        # with no detector bound the area vanishes too and the ratio is indeterminate.
        numerical_aperture_sq = math.nan if detector_area_m2 == 0.0 else math.inf

    try:
        depth_of_field_m = wavelength_m / numerical_aperture_sq
    except ZeroDivisionError:
        # Diverges as the aperture vanishes, except at zero energy, where the numerator
        # vanishes with it and the ratio is indeterminate.
        depth_of_field_m = math.nan if wavelength_m == 0.0 else math.inf

    return ProductGeometry(
        probe_wavenumber_per_m=wavenumber_per_m,
        probe_angular_wavenumber_rad_per_m=2.0 * numpy.pi * wavenumber_per_m,
        probe_photon_flux_per_s=photon_flux_per_s,
        probe_power_W=energy_eV_to_J(probe_energy_eV) * photon_flux_per_s,
        object_plane_propagation_distance_m=propagation_distance_m,
        object_plane_pixel_geometry=object_plane_pixel_geometry,
        fresnel_number=fresnel_number,
        detector_numerical_aperture=math.sqrt(numerical_aperture_sq),
        depth_of_field_m=depth_of_field_m,
    )


@dataclass(frozen=True)
class PropagatorParameters:
    """Geometric parameters for a wavefield propagator: wavelength, extent, pixel size, and distance."""

    wavelength_m: float
    """Illumination wavelength in meters."""
    width_px: int
    """Number of pixels in the x-direction."""
    height_px: int
    """Number of pixels in the y-direction."""
    pixel_width_m: float
    """Upstream-plane pixel width in meters.

    For the single-FFT propagators this is the plane at the smaller z: the *input*
    plane when ``propagation_distance_m`` is positive, and the *output* plane when it
    is negative. The opposite plane has pitch ``lambda |z| / (width_px * pixel_width_m)``
    -- see :func:`compute_far_field_pixel_geometry`. Both grids are the same for the
    pitch-preserving propagators (:class:`AngularSpectrumPropagator`,
    :class:`FresnelTransferFunctionPropagator`), so the distinction is moot there.
    """
    pixel_height_m: float
    """Upstream-plane pixel height in meters; see :attr:`pixel_width_m`."""
    propagation_distance_m: float
    """Propagation distance in meters. Negative propagates backward, which for the
    single-FFT propagators is the exact inverse of the forward operator built from the
    same parameters."""

    @property
    def dx(self) -> float:
        """Pixel width in wavelengths."""
        return self.pixel_width_m / self.wavelength_m

    @property
    def pixel_aspect_ratio(self) -> float:
        """Pixel aspect ratio (width / height)."""
        return self.pixel_width_m / self.pixel_height_m

    @property
    def z(self) -> float:
        """Propagation distance in wavelengths."""
        return self.propagation_distance_m / self.wavelength_m

    @property
    def pixel_fresnel_number_x(self) -> float:
        """Signed per-pixel Fresnel number ``dx^2 / (lambda z)`` along the width axis.

        Meaningful only paired with :attr:`pixel_aspect_ratio`, which is how the
        propagator algebra carries the height: every use of this number inside the
        single-FFT propagators is combined with a power of the aspect ratio that
        recovers the corresponding per-axis quantity. The amplitude prefactor, for
        one, computes the symmetric ``dx dy / (lambda z)`` as ``Fr / ar`` rather than
        storing it. Redefining this number to span both axes would therefore
        double-count the height -- and would do so silently, since every such error
        vanishes for square pixels.

        Signed because those phase terms must conjugate when the propagation
        direction reverses; take :func:`numpy.absolute` where only the magnitude is
        meant. Contrast :func:`compute_full_aperture_fresnel_number`, which shares the
        signed convention incidentally rather than load-bearingly.

        Distinct from that function in value as well: it is the full-aperture number
        ``W H / (lambda z)``, and the two differ by a factor of ``width_px *
        height_px * (dy / dx)``.
        """
        return numpy.square(self.dx) / self.z

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        JJ, II = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        XX = II - self.width_px // 2  # noqa: N806
        YY = JJ - self.height_px // 2  # noqa: N806
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = numpy.meshgrid(fy, fx, indexing='ij')  # noqa: N806
        return FY, FX


class Propagator(ABC):
    """Abstract interface for free-space wavefield propagators."""

    @abstractmethod
    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        pass


class AngularSpectrumPropagator(Propagator):
    """Exact propagator using the angular-spectrum transfer function; valid for all Fresnel numbers."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = TWO_PI_J * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)
        tf = numpy.exp(i2piz * numpy.sqrt(numpy.maximum(1 - ratio, 0.0)))

        # ifftshift matches the centered frequency grid to the unshifted spectrum in propagate
        self._transfer_function = ifftshift(numpy.where(ratio < 1, tf, 0))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(Propagator):
    """Fresnel propagator using a paraxial transfer function in the frequency domain."""

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = TWO_PI_J * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)

        # ifftshift matches the centered frequency grid to the unshifted spectrum in propagate
        self._transfer_function = ifftshift(numpy.exp(i2piz * (1 - ratio / 2)))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(Propagator):
    """Fresnel propagator using the direct Fresnel transform; changes pixel size between planes.

    The output plane has pitch ``lambda |z| / (N dx)`` -- see
    :func:`compute_far_field_pixel_geometry`. A negative propagation distance selects the
    exact inverse of the forward operator built from the same parameters, in which case
    :attr:`PropagatorParameters.pixel_width_m` describes the *output* plane.
    Retains the input quadratic phase that :class:`FraunhoferPropagator` drops, so it
    stays accurate outside the far field.

    The dropped/retained quadratic phase spans ``X_max = N / 2``, so the controlling
    quantity for far-field validity is ``N^2 * pixel_fresnel_number_x``, not the pixel
    Fresnel number alone. For N=256 that makes the honest condition ``Fr << 1.5e-5``.
    """

    def __init__(self, parameters: PropagatorParameters) -> None:
        if parameters.propagation_distance_m == 0.0:
            raise ValueError(
                'FresnelTransformPropagator requires a nonzero propagation distance; '
                'the output pixel size lambda*z/(N*dx) vanishes at z=0. Use '
                'AngularSpectrumPropagator, which is the identity there.'
            )

        ipi = 1j * numpy.pi

        # Signed: C2 and _B are pure phases that must conjugate when the direction
        # reverses, which is what makes the backward branch the forward branch's inverse.
        Fr = parameters.pixel_fresnel_number_x  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        # Magnitude only: a signed amplitude prefactor would flip the sign of the
        # recovered field on the backward branch.
        C0 = numpy.absolute(Fr) / (1j * ar)  # noqa: N806
        C1 = numpy.exp(TWO_PI_J * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0
        self._B = numpy.exp(ipi * Fr * (numpy.square(XX) + numpy.square(YY / ar)))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))
        else:
            return self._B * fftshift(ifft2(ifftshift(wavefield * self._A)))


class FraunhoferPropagator(Propagator):
    """Far-field (Fraunhofer) propagator: :class:`FresnelTransformPropagator` with the
    input quadratic phase ``exp(i pi Fr (X^2 + Y^2))`` dropped.

    The dropped/retained quadratic phase spans ``X_max = N / 2``, so the controlling
    quantity for far-field validity is ``N^2 * pixel_fresnel_number_x``, not the pixel
    Fresnel number alone. For N=256 that makes the honest condition ``Fr << 1.5e-5``.

    Shares the pitch and direction conventions of :class:`FresnelTransformPropagator`.
    """

    def __init__(self, parameters: PropagatorParameters) -> None:
        if parameters.propagation_distance_m == 0.0:
            raise ValueError(
                'FraunhoferPropagator requires a nonzero propagation distance; '
                'the output pixel size lambda*z/(N*dx) vanishes at z=0. Use '
                'AngularSpectrumPropagator, which is the identity there.'
            )

        ipi = 1j * numpy.pi

        # Signed phase, magnitude-only prefactor -- see FresnelTransformPropagator.
        Fr = parameters.pixel_fresnel_number_x  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = numpy.absolute(Fr) / (1j * ar)  # noqa: N806
        C1 = numpy.exp(TWO_PI_J * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield)))
        else:
            return fftshift(ifft2(ifftshift(wavefield * self._A)))


def choose_propagator(parameters: PropagatorParameters) -> tuple[Propagator, PixelGeometry]:
    """Select the correctly-sampled propagator and report the plane its output lives on.

    The single-FFT and transfer-function families sample opposite regimes, and each
    carries an implicit output grid, so the choice cannot be hidden inside either class:
    swapping silently would return a field on a plane the caller did not ask for. The
    returned :class:`PixelGeometry` is therefore part of the answer.

    Selection is per-axis with the conservative outcome, so an anisotropic geometry is
    never aliased on its narrow axis:

    - far-field pitch <= source pitch on *both* axes -> :class:`AngularSpectrumPropagator`
      on the source grid;
    - otherwise -> :class:`FresnelTransformPropagator` on the far-field grid.

    For square pixels this is the scalar condition ``|pixel_fresnel_number_x| <= 1 / N``.
    The two methods are numerically interchangeable at that crossover (measured
    agreement 2.5e-08); away from it they disagree by tens of percent, but that is grid
    disagreement rather than physics.

    At ``z = 0`` the far-field pitch is zero, so this selects angular spectrum on the
    source grid -- the identity for any geometry without evanescent modes. Zero distance
    is thus resolved by construction rather than by special case.

    Two propagators are never selected. :class:`FraunhoferPropagator` is
    :class:`FresnelTransformPropagator` with the input quadratic phase dropped, so the
    latter is strictly more accurate for one extra multiply.
    :class:`FresnelTransferFunctionPropagator` is the paraxial approximation of angular
    spectrum over the same grid.
    """
    source_pixel_geometry = PixelGeometry(
        width_m=parameters.pixel_width_m, height_m=parameters.pixel_height_m
    )
    extent = ImageExtent(width_px=parameters.width_px, height_px=parameters.height_px)
    far_field_pixel_geometry = compute_far_field_pixel_geometry(
        source_pixel_geometry,
        extent,
        wavelength_m=parameters.wavelength_m,
        propagation_distance_m=parameters.propagation_distance_m,
    )

    if (
        far_field_pixel_geometry.width_m <= source_pixel_geometry.width_m
        and far_field_pixel_geometry.height_m <= source_pixel_geometry.height_m
    ):
        return AngularSpectrumPropagator(parameters), source_pixel_geometry

    return FresnelTransformPropagator(parameters), far_field_pixel_geometry


@dataclass(frozen=True)
class PropagatedWavefield:
    """Stack of wavefields at evenly-spaced free-space propagation distances,
    produced by :func:`propagate_wavefield`.

    Stores the complex wavefield as ``(num_steps, num_incoherent_modes, height_px,
    width_px)``. Per-step intensity (incoherent-mode sum of ``|wf|^2``) and the three
    orthogonal planes used by the GUI are derived lazily.
    """

    wavefield: ComplexArrayType
    begin_coordinate_m: float
    end_coordinate_m: float
    pixel_geometry: PixelGeometry

    @property
    def num_steps(self) -> int:
        return self.wavefield.shape[0]

    @property
    def num_incoherent_modes(self) -> int:
        return self.wavefield.shape[1]

    @property
    def height_px(self) -> int:
        return self.wavefield.shape[2]

    @property
    def width_px(self) -> int:
        return self.wavefield.shape[3]

    @property
    def intensity(self) -> RealArrayType:
        """Per-step intensity image: ``sum_modes |wavefield|^2``,
        shape ``(num_steps, height_px, width_px)``. Recomputed on each access;
        cache in a local if calling repeatedly in a hot loop."""
        return numpy.sum(intensity(self.wavefield), axis=1)

    def get_xy_intensity(self, step: int) -> RealArrayType:
        return self.intensity[step]

    def get_zx_intensity(self) -> RealArrayType:
        return _cut_central_row(self.intensity)

    def get_zy_intensity(self) -> RealArrayType:
        return _cut_central_column(self.intensity)

    def get_xy_wavefield(self, step: int, mode: int) -> ComplexArrayType:
        """Complex wavefield of a single incoherent mode at one propagation step,
        shape ``(height_px, width_px)``.

        The mode-summed counterpart is :meth:`get_xy_intensity`. Summing over
        mutually incoherent modes is only meaningful in intensity, so the per-mode
        accessors are the only way to reach the phase.
        """
        return self.wavefield[step, mode]

    def get_zx_wavefield(self, mode: int) -> ComplexArrayType:
        """Complex ZX plane of a single incoherent mode, shape ``(width_px, num_steps)``."""
        return _cut_central_row(self.wavefield[:, mode])

    def get_zy_wavefield(self, mode: int) -> ComplexArrayType:
        """Complex ZY plane of a single incoherent mode, shape ``(height_px, num_steps)``."""
        return _cut_central_column(self.wavefield[:, mode])

    def save_npz(self, file_path: Path) -> None:
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            wavefield=self.wavefield,
            intensity=self.intensity,
            begin_coordinate_m=self.begin_coordinate_m,
            end_coordinate_m=self.end_coordinate_m,
            pixel_height_m=self.pixel_geometry.height_m,
            pixel_width_m=self.pixel_geometry.width_m,
        )


def propagate_wavefield(
    wavefield: ComplexArrayType,
    *,
    pixel_geometry: PixelGeometry,
    wavelength_m: float,
    begin_coordinate_m: float,
    end_coordinate_m: float,
    num_steps: int,
) -> PropagatedWavefield:
    """Propagate a multi-mode wavefield through a slab of free space using the
    angular-spectrum propagator at ``num_steps`` evenly-spaced distances in
    ``[begin_coordinate_m, end_coordinate_m]``.

    Args:
        wavefield: Complex source-plane wavefield, shape ``(num_modes, height_px,
            width_px)``. Each mode is propagated independently and the result preserves
            the mode axis.
        pixel_geometry: Source-plane pixel geometry (assumed constant across modes
            and propagation distances).
        wavelength_m: Illumination wavelength in meters.
        begin_coordinate_m: Smallest propagation distance in the output stack.
        end_coordinate_m: Largest propagation distance in the output stack.
        num_steps: Number of evenly-spaced steps along z.
    """
    if wavefield.ndim != 3:
        raise ValueError(
            f'wavefield must be 3-dimensional (modes, height, width); got ndim={wavefield.ndim}.'
        )

    num_modes, height_px, width_px = wavefield.shape

    propagated = numpy.zeros((num_steps, num_modes, height_px, width_px), dtype=wavefield.dtype)
    distance_m = numpy.linspace(begin_coordinate_m, end_coordinate_m, num_steps)

    for idx, z_m in enumerate(distance_m):
        params = PropagatorParameters(
            wavelength_m=wavelength_m,
            width_px=width_px,
            height_px=height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            propagation_distance_m=float(z_m),
        )
        propagator = AngularSpectrumPropagator(params)
        for mode in range(num_modes):
            propagated[idx, mode, :, :] = propagator.propagate(wavefield[mode, :, :])

    return PropagatedWavefield(
        wavefield=propagated,
        begin_coordinate_m=begin_coordinate_m,
        end_coordinate_m=end_coordinate_m,
        pixel_geometry=pixel_geometry,
    )
