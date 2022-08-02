import numpy

from ...api.probe import ProbeArrayType
from ..data import Detector
from .settings import ProbeSettings
from .sizer import ProbeSizer


def single_probe(probe_shape, lambda0, dx_dec, dis_defocus, dis_StoD, **kwargs):
    # return probe sorted by the spectrum
    # return scale is the wavelength dependent pixel scaling factor
    """
    Summary of this function goes here
    Parameters: probe_shape  -> the matrix size for probe
                lambda0      -> central wavelength
                dx_dec       -> pixel size on detector
                dis_defocus  -> defocus distance (sample to the focal plane)
                dis_StoD     -> sample to detector distance
                kwargs       -> setup: 'velo','2idd','lamni'
                             -> radius: zone plate radius
                             -> outmost: outmost zone width
                             -> beamstop: diameter of central beamstop
    """

    probe = numpy.zeros((probe_shape, probe_shape), dtype=numpy.complex)

    # pixel size on sample plane
    dx = lambda0 * dis_StoD / probe_shape / dx_dec

    # get zone plate parameter
    T, dx_fzp, FL0 = fzp_calculate(lambda0, dis_defocus, probe_shape, dx, **kwargs)

    nprobe = fresnel_propagation(T, dx_fzp, (FL0 + dis_defocus), lambda0)

    probe = nprobe / (numpy.sqrt(numpy.sum(numpy.abs(nprobe)**2)))

    return probe


def gaussian_spectrum(lambda0, bandwidth, energy):
    spectrum = numpy.zeros((energy, 2))
    sigma = lambda0 * bandwidth / 2.355
    d_lam = sigma * 4 / (energy - 1)
    spectrum[:, 0] = numpy.arange(-1 * numpy.floor(energy / 2), numpy.ceil(
        energy / 2)) * d_lam + lambda0
    spectrum[:, 1] = numpy.exp(-(spectrum[:, 0] - lambda0)**2 / sigma**2)
    return spectrum


def fzp_calculate(wavelength, dis_defocus, M, dx, **kwargs):
    """
    this function can calculate the transfer function of zone plate
    return the transfer function, and the pixel sizes
    """

    FZP_para = get_setup(**kwargs)

    FL = 2 * FZP_para['radius'] * FZP_para['outmost'] / wavelength

    # pixel size on FZP plane
    dx_fzp = wavelength * (FL + dis_defocus) / M / dx
    # coordinate on FZP plane
    lx_fzp = -dx_fzp * numpy.arange(-1 * numpy.floor(M / 2), numpy.ceil(M / 2))

    XX_FZP, YY_FZP = numpy.meshgrid(lx_fzp, lx_fzp)
    # transmission function of FZP
    T = numpy.exp(-1j * 2 * numpy.pi / wavelength * (XX_FZP**2 + YY_FZP**2) / 2 / FL)
    C = numpy.sqrt(XX_FZP**2 + YY_FZP**2) <= FZP_para['radius']
    H = numpy.sqrt(XX_FZP**2 + YY_FZP**2) >= FZP_para['CS'] / 2

    return T * C * H, dx_fzp, FL


def get_setup(**kwargs):

    if 'setup' in kwargs:
        setup = kwargs.get('setup')
    else:
        setup = 'custom'

    switcher = {
        'velo': {
            'radius': 90e-6,
            'outmost': 50e-9,
            'CS': 60e-6
        },
        '2idd': {
            'radius': 80e-6,
            'outmost': 70e-9,
            'CS': 60e-6
        },
        'lamni': {
            'radius': 114.8e-6 / 2,
            'outmost': 60e-9,
            'CS': 40e-6
        },
        'custom': {
            'radius': kwargs.get('radius'),
            'outmost': kwargs.get('outmost'),
            'CS': kwargs.get('beamstop')
        }
    }

    FZP_para = switcher.get(setup)
    return FZP_para


def fresnel_propagation(input, dxy, z, wavelength):
    """
    This is the python version code for fresnel propagation
    Summary of this function goes here
    Parameters:    dx,dy  -> the pixel pitch of the object
                z      -> the distance of the propagation
                lambda -> the wave length
                X,Y    -> meshgrid of coordinate
                input     -> input object
    """

    (M, N) = input.shape
    k = 2 * numpy.pi / wavelength
    # the coordinate grid
    M_grid = numpy.arange(-1 * numpy.floor(M / 2), numpy.ceil(M / 2))
    N_grid = numpy.arange(-1 * numpy.floor(N / 2), numpy.ceil(N / 2))
    lx = M_grid * dxy
    ly = N_grid * dxy

    XX, YY = numpy.meshgrid(lx, ly)

    # the coordinate grid on the output plane
    fc = 1 / dxy
    fu = wavelength * z * fc
    lu = M_grid * fu / M
    lv = N_grid * fu / N
    Fx, Fy = numpy.meshgrid(lu, lv)

    if z > 0:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
        kern = input * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.fft2(numpy.fft.fftshift(kern))
        OUT = numpy.fft.fftshift(cgh * numpy.fft.fftshift(pf))
    else:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.ifft2(
            numpy.fft.fftshift(input * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        OUT = numpy.fft.fftshift(cgh) * pf
    return OUT


class FresnelZonePlateProbeInitializer:

    def __init__(self, detector: Detector, probeSettings: ProbeSettings,
                 sizer: ProbeSizer) -> None:
        self._detector = detector
        self._probeSettings = probeSettings
        self._sizer = sizer

    def __call__(self) -> ProbeArrayType:
        shape = self._sizer.getProbeSize()
        lambda0 = self._sizer.getWavelengthInMeters()
        dx_dec = self._detector.getPixelSizeXInMeters()  # TODO non-square pixels are unsupported
        dis_defocus = self._probeSettings.defocusDistanceInMeters.value
        dis_StoD = self._detector.getDetectorDistanceInMeters()
        radius = self._probeSettings.zonePlateRadiusInMeters.value
        outmost = self._probeSettings.outermostZoneWidthInMeters.value
        beamstop = self._probeSettings.beamstopDiameterInMeters.value

        probe = single_probe(shape,
                             float(lambda0),
                             float(dx_dec),
                             float(dis_defocus),
                             float(dis_StoD),
                             radius=float(radius),
                             outmost=float(outmost),
                             beamstop=float(beamstop))
        return probe
