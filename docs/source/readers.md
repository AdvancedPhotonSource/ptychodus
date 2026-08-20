# Available File Readers

File readers are implemented using a Python namespace plugin system. We would be happy to add file readers to support more ptychography instruments.

- [Advanced Light Source (ALS)](https://als.lbl.gov/beamlines)
  - [7.0.1.2 COSMIC Imaging](https://als.lbl.gov/beamlines/7-0-1-2/) (`*.cxi`)
- [Advanced Photon Source (APS)](https://www.aps.anl.gov/Beamlines/Beamline-Directory)
  - [2-ID-D Microfluorescence and Bionanoprobe (BNP)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/5)
  - [2-ID-D Microprobe](https://www.aps.anl.gov/Beamlines/Beamline-Directory/5)
  - [2-ID-E Microprobe](https://www.aps.anl.gov/Beamlines/Beamline-Directory/61)
  - [4-ID-B,G,H POLAR: Polarization Modulation Spectroscopy](https://www.aps.anl.gov/Beamlines/Beamline-Directory/225)
  - [9-ID-D Coherent Surface Scattering Imaging (CSSI)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/221)
  - [12-ID-E High Resolution Small Angle X-ray Scattering (Ptycho-SAXS)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/228)
  - [19-ID-E In-situ Nanoprobe (ISN)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/232)
  - [26-ID-C CNM/APS Hard X-ray Nanoprobe (HXN)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/92)
  - [31-ID-E Ptychography-Laminography (LYNX)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/240)
  - [33-ID-C PtychoProbe, VelociProbe endstation](https://www.aps.anl.gov/Beamlines/Beamline-Directory/241)
  - [34-ID-C Microdiffraction, Coherent X-ray Scattering](https://www.aps.anl.gov/Beamlines/Beamline-Directory/242) — now listed as 34-ID-F Atomic
- [Linac Coherent Light Source (LCLS)](https://lcls.slac.stanford.edu/instruments)
  - [Hutch 1.3 XPP: X-ray Pump Probe](https://lcls.slac.stanford.edu/instruments/xpp)
  - SLAC NumPy Zipped Archive (`*.npz`)
- [MAX IV](https://www.maxiv.lu.se/beamlines-accelerators/beamlines/)
  - [NanoMAX](https://www.maxiv.lu.se/beamlines-accelerators/beamlines/nanomax/) (`*.h5`)
- [National Synchrotron Light Source II (NSLS-II)](https://www.bnl.gov/nsls2/beamlines)
  - [3-ID Hard X-ray Nanoprobe (HXN)](https://www.bnl.gov/nsls2/beamlines/beamline.php?r=3-ID)
- [Swiss Light Source (SLS)](https://www.psi.ch/en/sls/beamlines-at-sls)
  - [X12SA cSAXS: Coherent Small-Angle X-ray Scattering](https://www.psi.ch/en/node/508)
- Common File Formats
  - [Coherent X-ray Imaging](https://www.cxidb.org/cxi.html) (`*.cxi`)
  - Comma-Separated Values (`*.csv`)
  - EPICS Multi-Dimensional Archive (`*.mda`)
  - [fold_slice](https://github.com/yijiang1/fold_slice) (`*.mat`, `*.h5`)
  - NumPy Binary Files (`*.npy`, `*.npz`)
  - Ptychodus Diffraction Patterns (`*.h5`, `*.npz`)
  - Ptychodus Product (`*.h5`, `*.npz`)
  - Space-Separated Values (`*.txt`)
  - Tagged Image File Format (`*.tif`, `*.tiff`)

## Good/Bad Pixel Masks

Currently there are two numpy (NPY) file formats that can be used to indicate detector pixels that are usable ("good pixels") or unusable ("bad pixels") for processing. Both file types contain a 2-D boolean array with the same dimensions as an unprocessed detector frame. For the "good pixels" format, True indicates a usable pixel and False indicates an unusable pixel. For the "bad pixels" format, True indicates an unusable pixel and False indicates a usable pixel. When one of these files is provided, Ptychodus will zero bad pixels and provide the mask to processing algorithms that support pixel masks. When one of these files is not provided, Ptychodus assumes that all pixels should be used for processing.
