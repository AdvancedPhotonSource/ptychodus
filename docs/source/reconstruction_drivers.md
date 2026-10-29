# Reconstruction Drivers

Ptychodus ships a reconstruction command per beamline. Each one reads that instrument's raw
diffraction and position files, builds a product, runs it through pty-chi, and writes the
result as a standard-layout directory. They need no GUI and no settings file, so they suit a
job scheduler, a data-management pipeline, or a terminal on a beamline workstation.

Every driver is installed as a console command by `pip install ptychodus`.

## Choosing a driver

Nine commands differ only in which instrument they read. The last four differ in shape: they
take their configuration from a file, or operate on a directory another driver prepared.

| Command | Instrument | Diffraction input | Position input |
| --- | --- | --- | --- |
| `ptychodus-reconstruct-2ide` | APS 2-ID-E XFM | `fly###_data_NNN.h5` series | EPICS MDA, `mda/2xfm_NNNN.mda` |
| `ptychodus-reconstruct-12ide` | APS 12-ID-E Ptycho-SAXS | `<sample>_<scan>_master.h5` or any frame file (2026_2 layout onward) | The same master; positions are read from it |
| `ptychodus-reconstruct-34ide` | APS 34-ID-E Microdiffraction | One scan's TIFF series | EPICS MDA, `mda_files/34ide_NNNNNN.mda` |
| `ptychodus-reconstruct-bnp` | APS 2-ID-D Bionanoprobe | `bnp_flyNNNN_NNNNNN.h5` series | EPICS MDA, `mda/bnp_flyNNNN.mda` |
| `ptychodus-reconstruct-isn` | APS 19-ID-E In-situ Nanoprobe | `PTYCHO/scan_NNNN_FFFFF.h5` series | `Processed/SOCKETSERVER/Scan_NNNN_position.h5` |
| `ptychodus-reconstruct-lamni` | APS 31-ID-E LamNI | Raw LamNI HDF5 | LamNI `.dat` (Orchestra or softGlueZynq) |
| `ptychodus-reconstruct-polar` | APS 4-ID-B,G,H POLAR | `scan_NNNNNN_master.hdf` | The same master file |
| `ptychodus-reconstruct-velociprobe` | APS 33-ID-C VelociProbe | `fly###_master.h5` | Encoder text file, `fly###_0.txt` |
| `ptychodus-reconstruct-foldslice` | Any, via fold_slice | Preprocessed `..._dp.hdf5` | Matching `..._para.hdf5` |
| `ptychodus-reconstruct-pear` | Any, via a PEAR parameter file | Named in the parameter file | Named in the parameter file |
| `ptychodus-reconstruct-batch-lamni` | APS 31-ID-E LamNI, a whole tomogram | Named in the settings file | Named in the settings file |
| `ptychodus-reconstruct` | Any, over a prepared directory | `diffraction.h5` | `product.h5` |
| `ptychodus-reconstruct-subprocess` | Any, over a prepared directory, for a launcher | `diffraction.h5` | `product.h5` |

A scan that has already been preprocessed by fold_slice goes through
`ptychodus-reconstruct-foldslice` whatever instrument recorded it, because the pattern file
carries its own geometry.

## Common arguments

The nine instrument commands share one argument set. Three paths are required:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054
```

Only one member of a multi-file detector series need be named; the reader globs the rest.

The remaining arguments fall into four groups.

`--crop-extent-px`, `--beam-center-x-px`, `--beam-center-y-px`
: Cut a square region out of each raw frame about the beam centre. The centre comes from the
  data file when it records one, and is estimated from the patterns otherwise. Cropping is
  usually what decides reconstruction time, since it sets the array size. `--crop-extent-px auto`
  (`crop_extent_px: auto` in a batch settings file) picks the smallest power-of-two side that
  holds the mean pattern's signal above its noise floor, capped at the largest that fits.

`--detector-distance-m`, `--photon-energy-eV`, `--detector-pixel-size-m`, `--probe-photon-count`, `--tomography-angle-deg`
: Override metadata the data file records wrongly or not at all. Each is used only when given.
  The last three are offered only by the commands whose file layout needs them.

`--min-total-counts`, `--max-valid-count`, `--bad-pixels-file`
: Drop patterns whose total counts fall below a floor, zero pixels at or above a ceiling, and
  apply a bad-pixel mask covering the full uncropped detector. The ceiling defaults to the
  pattern dtype's maximum, which is how Dectris detectors mark an invalid pixel.

`--background-file`, `--flat-field-file`, `--subtract-background`, `--divide-flat-field`
: Supply a measured detector background and flat-field response covering the full detector,
  overriding any the data file carries, and choose whether to correct the patterns with them
  or carry them alongside. `--background-file-type` and `--flat-field-file-type` name the
  readers. See [Detector Background and Flat Field](readers.md#detector-background-and-flat-field).

`--num-probe-modes`, `--num-opr-modes`, `--fzp-preset`, `--fzp-defocus-m`, `--object-padding-px`, `--seed`
: Shape the initial guesses. The zone-plate preset and defocus describe the optic that formed
  the probe.

Four more control the run itself: `--ptychi-options` chooses the reconstruction algorithm and
its settings, `--dry-run` reports the resolved configuration without reconstructing,
`--prepare-only` writes a prepared directory and stops (see
[Preparing and reconstructing separately](#preparing-and-reconstructing-separately)), and
`--log-level` takes a Python logging level as an integer (`10` for DEBUG, `20` for INFO).

```{note}
A profile can add or remove arguments. `ptychodus-reconstruct-foldslice` has no cropping,
bad-pixel, background or flat-field options, because it reads patterns that are already prepared.
`ptychodus-reconstruct-polar` adds statistical pattern filters. Run any command with `--help`
for the list it actually accepts.
```

## The output directory

Every driver writes one directory in the same layout, whatever instrument it read:

```text
fly054/
  diffraction.h5         # the assembled patterns, as reconstructed
  product.h5             # the final object, probe and positions
  product.000100.h5      # a snapshot every --num-sync-epochs epochs
  ptychi_options.yaml    # the options that actually ran
  ptychodus.log          # everything the run wrote to its console
```

`product.h5` is the result: it carries the object, the probe, the refined probe positions, and
the metadata the reconstruction used. It opens in the GUI and reads back through the Python
API. `ptychodus-enhance` takes it with a fluorescence dataset from the same scan, as
{doc}`fluorescence` describes.

`ptychi_options.yaml` records the options *after* the product's own values were folded in, so
it describes the run exactly rather than describing what was asked for. It can be passed
straight back to `--ptychi-options` to repeat the run.

`diffraction.h5` is written before the reconstruction starts, so an interrupted run still
leaves the one artifact that cannot be rebuilt without the raw files. It can run to several
gigabytes; `--no-save-diffraction` skips it.

`ptychodus-reconstruct-batch-lamni` writes one such directory per scan, and adds
`batch_settings.yaml` and `affine_fit.yaml` to each. The fit's `scale`, `asymmetry`,
`rotation_deg` and `shear_deg` follow pty-chi's convention for the same transform, in
`(x, y)`: a positive rotation turns x towards y, a positive asymmetry magnifies y relative to
x, and a positive shear moves x in proportion to y. `linear_transform` holds the matrix
itself.

A driver run with `--prepare-only` writes a smaller directory, which holds what a
reconstruction starts from rather than what it produced:

```text
fly054-prepared/
  diffraction.h5         # the assembled patterns
  product.h5             # the initial object, probe and positions
  ptychi_options.yaml    # the options the reconstruction will run with
```

## Choosing the algorithm

With no `--ptychi-options`, a driver runs stock LSQML. To run anything else, or to change any
reconstruction setting, pass a pty-chi options file.

Obtain one by asking a driver to print the defaults:

```sh
$ ptychodus-reconstruct-2ide --print-ptychi-options > lsqml.yaml    # LSQML, the default
$ ptychodus-reconstruct-2ide --print-ptychi-options DM > dm.yaml    # a different algorithm
```

The algorithm may be `DM`, `RAAR`, `PIE`, `ePIE`, `rPIE`, `LSQML`, `Autodiff` or `BH`. The flag
needs no other arguments, so any driver can produce the file.

Edit the result and pass it back:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054 \
    --ptychi-options ./lsqml.yaml
```

The file names its own algorithm, in an `options_class_name` field that appears throughout it.
That is what selects the reconstructor on the way back in, so change the algorithm by printing
a fresh file rather than by editing those names.

```{note}
Values describing the data — the wavelength, the object and probe pixel sizes, the slice
spacings, the propagation distance and the probe power — are written from the product when the
reconstruction starts, whatever the file says. The printed defaults list them. Editing them
changes nothing except to add a warning to the log.

A hand-written file must also spell scientific notation `1.0e-3` rather than `1e-3`: YAML
reads the shorter form as a string. Files produced by `--print-ptychi-options` are already
correct.
```

JSON is accepted as well as YAML, chosen by the file suffix, and `-` reads the options from
standard input.

## Reconstructing a whole tomogram

`ptychodus-reconstruct-batch-lamni` reconstructs a range of LamNI projections in one command.
It takes all of its settings from a YAML file, so a run is reproducible from something that can
be archived, and it copies that file into each scan's output directory.

```yaml
paths:
  experiment_folder: /data/31ide/31ide_2025-06-30/
  output_products_folder: /data/31ide/31ide_2025-06-30/analysis/batch_1/

batch:
  label: sample_A
  scan_start: 1000
  scan_end: 2400
  exclude_scans: [1140, 1150]

diffraction:
  crop_extent_px: 128

initial_object:
  object_padding_px: 512
```

```sh
$ ptychodus-reconstruct-batch-lamni --batch-settings batch.yaml --dry-run
$ ptychodus-reconstruct-batch-lamni --batch-settings batch.yaml --ptychi-options lsqml.yaml
```

Start with `--dry-run`: it resolves the scan table and reports every scan it would reconstruct,
with the files it would read and the directory it would write, without reading any data.

Scans run concurrently, one worker per visible CUDA device, so the job scheduler decides the
degree of parallelism through `CUDA_VISIBLE_DEVICES`. A scan whose `product.h5` already exists
is skipped, which is how an interrupted run resumes — re-run the same command.

### Driving it from Python

The command is a shim over a function that takes the same two things as objects: a
`BatchSettings` and a pty-chi `PtychographyTaskOptions`. A script that already holds those — one
sweeping a reconstruction parameter over a projection range, or building the settings from an
experiment database rather than a YAML file — calls it directly instead of writing a temporary
file and launching a subprocess.

```python
from ptychi.api import LSQMLOptions

from ptychodus.cli.reconstruct_batch_lamni import (
    Batch,
    BatchSettings,
    Diffraction,
    Paths,
    plan_batch_lamni,
    reconstruct_batch_lamni,
)

settings = BatchSettings(
    paths=Paths(
        experiment_folder='/data/31ide/31ide_2025-06-30/',
        output_products_folder='/data/31ide/31ide_2025-06-30/analysis/batch_1/',
    ),
    batch=Batch(label='sample_A', scan_start=1000, scan_end=1010),
    diffraction=Diffraction(crop_extent_px=128),
)

for record in plan_batch_lamni(settings):  # the dry run, in code: no data is read
    print(record.scan_no, record.encoder_angle_deg)

options = LSQMLOptions()
options.reconstructor_options.num_epochs = 200

result = reconstruct_batch_lamni(settings, options)
print(result.succeeded, result.skipped, result.failed)
```

The sections are the blocks of the YAML file, one class each, and every one of them is
importable from the same module. An omitted section takes the same defaults the file does, so
only `paths` is required.

Most runs start from a file that already exists, which makes a sweep two lines:

```python
from pathlib import Path

from ptychodus.cli.reconstruct_batch_lamni import load_batch_settings, reconstruct_batch_lamni

settings = load_batch_settings(Path('batch.yaml'))

for padding_px in (256, 512, 1024):
    settings.initial_object.object_padding_px = padding_px
    settings.paths.output_products_folder = Path(f'/tmp/sweep/pad{padding_px}')
    reconstruct_batch_lamni(settings)
```

The call blocks until the batch finishes and writes exactly what the command writes: one
standard-layout directory per scan, each with its own `batch_settings.yaml`. That archived copy
is rendered from the settings object the run was given — every default written out, and no
comments — because an object is all a Python caller has. The command passes its file's text
through instead, so a run from a file archives the file; a caller that wants the same can pass
the text as `settings_text`.

A scan that fails is reported rather than raised, because a tomogram with a projection missing
is a result its owner has to judge: `BatchResult` carries the scan numbers that succeeded, were
skipped, failed, or were cancelled, and `result.exit_code` is the status the console command
would have exited with. Anything that goes wrong before the first scan starts — an experiment
directory that is not there, a scan range that selects nothing, a crop that runs off the
detector — raises instead, since it describes every scan equally.

```{note}
The fan-out spawns one child process per scan, and a child spawned from a process that already
holds a CUDA context fails to initialize. Do not run torch on a GPU in the same interpreter
first: this is the same constraint that makes ptychodus enumerate devices in a throwaway child.
```

Neither function configures logging — nothing is printed until the caller sets logging up, for
example with `logging.basicConfig(level=logging.INFO)` — and neither installs signal handlers.
To make a run interruptible, pass a `CancellationToken`; it stops the batch at the first scan
boundary after the request and leaves the scans already written in place.

## Preparing and reconstructing separately

Reading and assembling the raw files and running pty-chi need not happen in one command. Give
any of the nine instrument commands `--prepare-only` and it stops after building the initial
product: it writes the prepared directory shown above and exits successfully without
reconstructing. `ptychodus-reconstruct` then runs the reconstruction from that directory:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054-prepared \
    --prepare-only
$ ptychodus-reconstruct -i ./fly054-prepared -o ./fly054
```

The split lets the preparation run where the raw data are, on a beamline workstation, and the
reconstruction run where the GPUs are. It also lets one scan be reconstructed several times
with different options without assembling the raw data again, which for a large scan is most
of the wall-clock time. Every argument that shapes the data or the initial guesses — cropping,
filtering, geometry, `--seed`, the probe options and `--ptychi-options` — takes effect at
preparation. `--num-sync-epochs` belongs to the reconstruction.

`ptychodus-reconstruct` reads no raw beamline files and knows nothing about any instrument. It
takes its options from the prepared directory's `ptychi_options.yaml` unless `--ptychi-options`
names another file, and falls back to stock LSQML, or BH for a near-field product, when there
is neither. The input directory is never modified, and the output must be a different directory,
since the final `product.h5` would otherwise overwrite the initial one. `--dry-run` loads and
pairs the inputs and reports what would run.

`ptychodus-reconstruct-subprocess` reconstructs the same input for a launching process. It
reports progress as newline-delimited JSON — one object per line, `started`, `epoch`,
`checkpoint`, `finished` — which makes it usable as a child process of a launcher that wants to
follow a reconstruction. Use `--event-file` to send that stream to a file and leave standard
output alone.

```{note}
`ptychodus-bdp` is deprecated and will be removed. Its prepared directory can still be
reconstructed with `ptychodus-reconstruct`; new pipelines should prepare data with
`--prepare-only` instead, which applies the instrument's own conditioning.
```

## Reconstructing from a PEAR parameter file

`ptychodus-reconstruct-pear` reads the flat JSON parameter file that PEAR's batch loop writes,
and reconstructs the single scan it describes. The parameter file names the instrument, the
input paths and every reconstruction setting, so the command takes almost no arguments of its
own and has no `--ptychi-options`: the algorithm settings come from the parameter file too.

```sh
$ ptychodus-reconstruct-pear --params ./scan0054_params.json --dry-run
```

The output directory is derived from the parameter file, following the same convention PEAR
uses, so results land where that workflow expects them.
