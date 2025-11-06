Available File Readers
======================

File readers are implemented using a Python namespace plugin system. We would
be happy to add file readers to support more ptychography instruments.

- Advanced Photon Source (APS)
    - 2-ID-D Bionanoprobe (BNP)
    - 2-ID-D Microprobe
    - 2-ID-E Microprobe
    - 4-ID-B,G,H Polarization Modulation Spectroscopy (Polar)
    - 9-ID-D Coherent Surface Scattering Imaging (CSSI)
    - 12-ID-E Ptycho-SAXS
    - 19-ID-E In-Situ Nanoprobe (ISN)
    - 26-ID-C CNM/APS Hard X-ray Nanoprobe (HXN)
    - 31-ID-E LYNX
    - 33-ID-C PtychoProbe
    - 33-ID-C Velociprobe
- Advanced Light Source (ALS)
    - Coherent X-ray Imaging (*.cxi)
- Linac Coherent Light Source (LCLS)
    - Hutch 1.3: X-ray Pump Probe (XPP)
- MAX IV
    - NanoMAX Diffraction Endstation (*.h5)
- National Synchrotron Light Source II (NSLS-II)
    - 3-ID Hard X-ray Nanoprobe (HXN)
- Swiss Light Source (SLS)
    - X12SA: Coherent Small-Angle X-ray Scattering (cSAXS)
- Common File Formats
    - Comma-Separated Values (*.csv)
    - EPICS Multi-Dimensional Archive (*.mda)
    - NumPy Binary Files (*.npy, *.npz)
    - PtychoShelves (*.mat, *.h5)
    - Ptychodus Diffraction Patterns (*.h5, *.npz)
    - Ptychodus Product (*.h5, *.npz)
    - Space-Separated Values (*.txt)
    - Tagged Image File Format (*.tif, *.tiff)

Good/Bad Pixel Masks
====================

Currently there are two numpy (NPY) file formats that can be used to indicate
detector pixels that are usable ("good pixels") or unusable ("bad pixels") for
processing. Both file types contain a 2-D boolean array with the same dimensions
as an unprocessed detector frame. For the "good pixels" format, True indicates a
usable pixel and False indicates an unusable pixel. For the "bad pixels" format,
True indicates an unusable pixel and False indicates a usable pixel. When one of
these files is provided, Ptychodus will zero bad pixels and provide the mask to
processing algorithms that support pixel masks. When one of these files is not
provided, Ptychodus assumes that all pixels should be used for processing.
