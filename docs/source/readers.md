# Available File Readers

File readers are implemented using a Python namespace plugin system. We would be happy to add file readers to support more ptychography instruments.

- [Advanced Light Source (ALS)](https://als.lbl.gov/beamlines)
  - [7.0.1.2 COSMIC Imaging](https://als.lbl.gov/beamlines/7-0-1-2/) (`*.cxi`, through the Coherent X-ray Imaging reader)
- [Advanced Photon Source (APS)](https://www.aps.anl.gov/Beamlines/Beamline-Directory)
  - [2-ID-D Microfluorescence and Bionanoprobe (BNP)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/5) (`*.mda`, `*.h5`, `*.hdf5`)
  - [2-ID-D Microprobe](https://www.aps.anl.gov/Beamlines/Beamline-Directory/5)
  - [2-ID-E Microprobe](https://www.aps.anl.gov/Beamlines/Beamline-Directory/61)
  - [4-ID-B,G,H POLAR: Polarization Modulation Spectroscopy](https://www.aps.anl.gov/Beamlines/Beamline-Directory/225)
  - [9-ID-D Coherent Surface Scattering Imaging (CSSI)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/221)
  - [12-ID-E High Resolution Small Angle X-ray Scattering (Ptycho-SAXS)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/228) (`*.dat`, `*.h5`, `*.hdf5`, `*.tif`, `*.tiff`)
  - [19-ID-E In-situ Nanoprobe (ISN)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/232)
  - [26-ID-C CNM/APS Hard X-ray Nanoprobe (HXN)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/92)
  - [31-ID-E Ptychography-Laminography (LamNI)](https://www.aps.anl.gov/Beamlines/Beamline-Directory/240) (`*.dat`, `*.h5`, `*.hdf5`)
  - [33-ID-C PtychoProbe, VelociProbe endstation](https://www.aps.anl.gov/Beamlines/Beamline-Directory/241)
  - [34-ID-E Microdiffraction](https://www.aps.anl.gov/Beamlines/Beamline-Directory/204) (`*.json`, `*.mda`, `*.tif`, `*.tiff`)
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
  - [Coherent X-ray Imaging](https://www.cxidb.org/cxi.html) (`*.cxi`), read and written; see [CXI Files](#cxi-files)
  - Comma-Separated Values (`*.csv`)
  - EPICS Multi-Dimensional Archive (`*.mda`)
  - [fold_slice](https://github.com/yijiang1/fold_slice) (`*.mat`, `*.h5`)
  - NumPy Binary Files (`*.npy`, `*.npz`)
  - Ptychodus Diffraction Patterns (`*.h5`, `*.npz`)
  - Ptychodus Product (`*.h5`, `*.npz`)
  - Space-Separated Values (`*.txt`)
  - Tagged Image File Format (`*.tif`, `*.tiff`)

## Good/Bad Pixel Masks

Several file formats can be used to indicate detector pixels that are usable ("good pixels") or unusable ("bad pixels") for processing. When one of these files is provided, Ptychodus will zero bad pixels and provide the mask to processing algorithms that support pixel masks. When one of these files is not provided, Ptychodus assumes that all pixels should be used for processing.

| Format | Contents |
| --- | --- |
| NumPy Bad Pixel Files (`*.npy`) | A 2-D boolean array with the same dimensions as an unprocessed detector frame, where True indicates an unusable pixel. |
| NumPy Good Pixel Files (`*.npy`) | The same array with the opposite sense, where True indicates a usable pixel. |
| APS 12-ID-E Valid Pixel Mask Files (`*.h5` `*.hdf5`) | A `valid_pixel_mask` dataset, where a nonzero value indicates a usable pixel. |
| APS 34-ID-E Microdiffraction Bad Pixel Files (`*.json`) | A `Bad pixels` array of `{"Pixel": [column, row], "Set": int}` objects, one per unusable pixel. The file carries no detector dimensions, so the reader supplies the extent of the EIGER2 CdTe 1M that writes it and rejects a coordinate outside it. |
| Coherent X-ray Imaging Files (`*.cxi`) | The `detector_1/mask` of a CXI file, which may hold nothing else. A pixel is bad when any bit other than "signal above background" (`0x1000`) is set. |

Masks are applied in raw detector coordinates. A preprocessed pixel is bad when any raw pixel it combines is bad, and pixels added by padding are always bad. A pixel holding a NaN or infinite value in any frame is zeroed and treated as bad for the whole dataset.

## Detector Background and Flat Field

A measured detector background and a flat field can accompany the patterns. The background is the signal recorded without the sample, in counts per frame; it must have the same exposure as the patterns, because it is used as given. The flat field is the detector's **response**: the per-pixel signal under uniform illumination, so that a measurement is about `flat_field × true + background`. A pattern is corrected as `(pattern − background) / flat_field`, clipped at zero.

By default both maps are carried with the patterns, unchanged, for reconstruction algorithms that model them. The `SubtractBackground` and `DivideFlatField` detector settings, or the `--subtract-background` and `--divide-flat-field` command options, apply them to the patterns instead; corrected patterns are floating point, and an applied map is no longer carried. A flat-field response at or below `FlatFieldFloor` marks its pixel bad when dividing. Correction happens after binning, padding and other preprocessing, with both maps carried through the same steps, so value filters act on measured counts.

| Format | Contents |
| --- | --- |
| NumPy Background Files (`*.npy`) | A 2-D array covering the full, unprocessed detector. |
| NumPy Flat-Field Response Files (`*.npy`) | A 2-D response map covering the full detector. Mean-one and absolute-efficiency maps are both accepted. |
| NumPy Flat-Field Correction Factor Files (`*.npy`) | Per-pixel factors to *multiply* a measurement by, the inverse convention; each is read as its reciprocal response, and a non-positive factor marks its pixel bad. |
| Coherent X-ray Imaging Files (`*.cxi`) | The background from `detector_1/data_dark`, averaged if it is a stack; the flat field from `ptychodus/flat_field` or `detector_1/qe_mask`. |

```{warning}
Some photon-counting detectors apply a flat-field correction in their electronics and record the correction factors alongside the data. Do not apply such a map again.
```

## CXI Files

Ptychodus reads and writes version 1.6 of the [Coherent X-ray Imaging format](https://www.cxidb.org/cxi.html). The reader recognizes the layout of a file by its contents, so it takes no options; a writer is chosen by layout.

| Layout | Written as |
| --- | --- |
| Coherent X-ray Imaging Files | Strict CXI 1.6. The object and probe are `image_1` and `image_2`, titled `object` and `probe`. |
| Coherent X-ray Imaging Files, SHARP layout | The object in `image_1/data` and the probe in `image_1/process_1/final_illumination` and `source_1/illumination`, as SHARP writes them. These hold one probe mode and one object layer. |
| Coherent X-ray Imaging Files, cdtools layout | Strict CXI with `source_1/wavelength`, `basis_vectors` stored transposed and the flat field in `detector_1/qe_mask`, as cdtools writes them. |

Translations are stored as CXI sample translations without a sign change. CXI defines a translation as the sample's offset from the beam, in a right-handed frame with y up and z downstream, and images follow CXI's default detector orientation, whose rows step towards −y and columns towards −x. Under that orientation the sample's offset equals the probe position in ptychodus coordinates.

Every layout also writes a `ptychodus` group in the entry, holding what CXI has no place for: scan indexes, pixel sizes, the object center, OPR modes and weights, losses, the flat field, and the remaining product metadata. Ptychodus reads its own files back exactly from any layout. In an exported reconstruction input, a pixel whose value was filled in rather than measured carries the mask bit `0x40000000` alongside `0x1` (invalid).
