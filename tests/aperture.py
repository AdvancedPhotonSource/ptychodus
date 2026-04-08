from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import fresnel, j0
import matplotlib.pyplot as plt
import numpy
import numpy.typing

from ptychodus.api.common import ComplexArrayType, RealArrayType


class SquareAperture:
    def __init__(self, width_m: float, wavelength_m: float) -> None:
        self._width_m = width_m
        self._wavelength_m = wavelength_m

    def get_fresnel_number(self, z_m: float) -> float:
        upper = (self._width_m / 2) ** 2
        lower = self._wavelength_m * z_m
        return upper / lower

    def _integral1d(self, r_m: RealArrayType, z_m: float) -> ComplexArrayType:
        sqrt2NF = numpy.sqrt(2 * self.get_fresnel_number(z_m))  # noqa: N806
        xi = 2 * r_m / self._width_m
        Sm, Cm = fresnel(sqrt2NF * (1 - xi))  # noqa: N806
        Sp, Cp = fresnel(sqrt2NF * (1 + xi))  # noqa: N806
        return (Cm + Cp + 1j * (Sm + Sp)) / numpy.sqrt(2)

    def diffract(self, x_m: RealArrayType, y_m: RealArrayType, z_m: float) -> ComplexArrayType:
        """Fresnel diffraction; see Goodman p.100"""
        assert x_m.shape == y_m.shape
        Ix = self._integral1d(x_m, z_m)  # noqa: N806
        Iy = self._integral1d(y_m, z_m)  # noqa: N806
        return Ix * Iy * numpy.exp(2j * numpy.pi * z_m / self._wavelength_m) / 1j


class CircularAperture:
    def __init__(self, diameter_m: float, wavelength_m: float) -> None:
        self._diameter_m = diameter_m
        self._wavelength_m = wavelength_m

    def get_fresnel_number(self, z_m: float) -> float:
        upper = (self._diameter_m / 2) ** 2
        lower = self._wavelength_m * z_m
        return upper / lower

    def diffract(self, x_m: RealArrayType, y_m: RealArrayType, z_m: float) -> ComplexArrayType:
        """Fresnel diffraction; see Goodman p.102"""
        assert x_m.shape == y_m.shape

        twopi = 2 * numpy.pi
        sqrtLZ = numpy.sqrt(self._wavelength_m * z_m)  # noqa: N806
        sqrtNF = numpy.sqrt(self.get_fresnel_number(z_m))  # noqa: N806

        rp = numpy.hypot(x_m, y_m) / sqrtLZ  # normalized radial coordinate at observation plane

        def real_integrand(rhop: float, rp_val: float) -> float:
            return rhop * numpy.cos(numpy.pi * rhop**2) * j0(twopi * rhop * rp_val)

        def imag_integrand(rhop: float, rp_val: float) -> float:
            return rhop * numpy.sin(numpy.pi * rhop**2) * j0(twopi * rhop * rp_val)

        result = numpy.empty_like(rp, dtype=complex)

        for idx in numpy.ndindex(rp.shape):
            rp_val = float(rp[idx])
            re, _ = quad(real_integrand, 0, sqrtNF, args=(rp_val,))
            im, _ = quad(imag_integrand, 0, sqrtNF, args=(rp_val,))
            result[idx] = re + 1j * im

        return twopi * result * numpy.exp(2j * numpy.pi * z_m / self._wavelength_m) / 1j


if __name__ == '__main__':
    wavelength_m = 500e-9  # 500 nm
    width_m = 1e-3  # 1 mm square aperture
    diameter_m = 1e-3  # 1 mm circular aperture (same size for fair comparison)
    z_m = 0.125  # 12.5 cm  →  Fresnel number = 4

    N = 256
    extent_m = 3e-3  # ±1.5 mm observation window (3× aperture width)
    coords = numpy.linspace(-extent_m / 2, extent_m / 2, N)
    xx, yy = numpy.meshgrid(coords, coords)

    square = SquareAperture(width_m, wavelength_m)
    circular = CircularAperture(diameter_m, wavelength_m)

    NF_sq = square.get_fresnel_number(z_m)
    NF_circ = circular.get_fresnel_number(z_m)
    print(f'Square aperture Fresnel number:   {NF_sq:.1f}')
    print(f'Circular aperture Fresnel number: {NF_circ:.1f}')

    print('Computing square aperture diffraction...')
    U_sq = square.diffract(xx, yy, z_m)

    # CircularAperture.diffract is radially symmetric, so compute on a 1D
    # radial grid and interpolate to 2D — much faster than N×N quad calls.
    print('Computing circular aperture diffraction (radial profile)...')
    r_max = numpy.hypot(xx, yy).max()
    r_1d = numpy.linspace(0.0, r_max, N)
    U_circ_1d = circular.diffract(r_1d, numpy.zeros_like(r_1d), z_m)

    r_obs = numpy.hypot(xx, yy)
    U_circ = interp1d(r_1d, U_circ_1d.real, kind='cubic')(r_obs) + 1j * interp1d(
        r_1d, U_circ_1d.imag, kind='cubic'
    )(r_obs)

    extent_mm = [-extent_m / 2 * 1e3, extent_m / 2 * 1e3] * 2  # for imshow
    kw = dict(origin='lower', extent=extent_mm, cmap='inferno')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    im0 = axes[0].imshow(numpy.abs(U_sq) ** 2, **kw)
    axes[0].set_title(
        f'Square aperture\nw = {width_m * 1e3:.1f} mm,  '
        f'N$_F$ = {NF_sq:.1f},  z = {z_m * 1e2:.1f} cm'
    )
    axes[0].set_xlabel('x (mm)')
    axes[0].set_ylabel('y (mm)')
    plt.colorbar(im0, ax=axes[0], label='Intensity (arb.)')

    im1 = axes[1].imshow(numpy.abs(U_circ) ** 2, **kw)
    axes[1].set_title(
        f'Circular aperture\nD = {diameter_m * 1e3:.1f} mm,  '
        f'N$_F$ = {NF_circ:.1f},  z = {z_m * 1e2:.1f} cm'
    )
    axes[1].set_xlabel('x (mm)')
    axes[1].set_ylabel('y (mm)')
    plt.colorbar(im1, ax=axes[1], label='Intensity (arb.)')

    fig.suptitle(
        f'Fresnel diffraction  (λ = {wavelength_m * 1e9:.0f} nm)',
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()
