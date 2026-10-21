# Reconstruction Drivers

Ptychodus ships a reconstruction command per beamline. Each one reads that instrument's raw
diffraction and position files, builds a product, runs it through pty-chi, and writes the
result as a standard-layout directory. They need no GUI and no settings file, so they suit a
job scheduler, a data-management pipeline, or a terminal on a beamline workstation.

Every driver is installed as a console command by `pip install ptychodus`.

## Choosing a driver

Nine commands differ only in which instrument they read. The last three differ in shape: they
take their configuration from a file, or operate on a directory another driver prepared.

| Command | Instrument | Diffraction input | Position input |
| --- | --- | --- | --- |
| `ptychodus-reconstruct-2ide` | APS 2-ID-E XFM | `fly###_data_NNN.h5` series | EPICS MDA, `mda/2xfm_NNNN.mda` |
| `ptychodus-reconstruct-12ide` | APS 12-ID-E Ptycho-SAXS | `<scan>_<line>_<point>.h5` series | Position `.dat` file |
| `ptychodus-reconstruct-34ide` | APS 34-ID-E Microdiffraction | One scan's TIFF series | EPICS MDA, `mda_files/34ide_NNNNNN.mda` |
| `ptychodus-reconstruct-bnp` | APS 2-ID-D Bionanoprobe | `bnp_flyNNNN_NNNNNN.h5` series | EPICS MDA, `mda/bnp_flyNNNN.mda` |
| `ptychodus-reconstruct-isn` | APS 19-ID-E In-situ Nanoprobe | `PTYCHO/scan_NNNN_FFFFF.h5` series | `Processed/SOCKETSERVER/Scan_NNNN_position.h5` |
| `ptychodus-reconstruct-lamni` | APS 31-ID-E LamNI | Raw LamNI HDF5 | LamNI `.dat` (Orchestra or softGlueZynq) |
| `ptychodus-reconstruct-polar` | APS 4-ID-B,G,H POLAR | `scan_NNNNNN_master.hdf` | The same master file |
| `ptychodus-reconstruct-velociprobe` | APS 33-ID-C VelociProbe | `fly###_master.h5` | Encoder text file, `fly###_0.txt` |
| `ptychodus-reconstruct-foldslice` | Any, via fold_slice | Preprocessed `..._dp.hdf5` | Matching `..._para.hdf5` |
| `ptychodus-reconstruct-pear` | Any, via a PEAR parameter file | Named in the parameter file | Named in the parameter file |
| `ptychodus-reconstruct-batch-lamni` | APS 31-ID-E LamNI, a whole tomogram | Named in the settings file | Named in the settings file |
| `ptychodus-reconstruct-subprocess` | Any, over a prepared directory | `diffraction.h5` | `product.h5` |

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
  usually what decides reconstruction time, since it sets the array size.

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

Three more control the run itself: `--ptychi-options` chooses the reconstruction algorithm and
its settings, `--dry-run` reports the resolved configuration without reconstructing, and
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
API.

`ptychi_options.yaml` records the options *after* the product's own values were folded in, so
it describes the run exactly rather than describing what was asked for. It can be passed
straight back to `--ptychi-options` to repeat the run.

`diffraction.h5` is written before the reconstruction starts, so an interrupted run still
leaves the one artifact that cannot be rebuilt without the raw files. It can run to several
gigabytes; `--no-save-diffraction` skips it.

`ptychodus-reconstruct-batch-lamni` writes one such directory per scan, and adds
`batch_settings.yaml` and `affine_fit.yaml` to each.

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

## Reconstructing a prepared directory

`ptychodus-reconstruct-subprocess` runs one reconstruction over a directory that already holds
`diffraction.h5` and `product.h5` — a directory another driver wrote, or one staged for the
purpose. It reads no raw beamline files and knows nothing about any instrument.

```sh
$ ptychodus-reconstruct-subprocess -i ./fly054 -o ./fly054-rerun --num-sync-epochs 10
```

It is the way to re-run a reconstruction with different options without re-reading and
re-assembling the raw data, which for a large scan is most of the wall-clock time. It also
takes its options from the input directory's own `ptychi_options.yaml` when `--ptychi-options`
is not given, so the command above repeats the original run.

Unlike the other drivers it reports progress as newline-delimited JSON — one object per line,
`started`, `epoch`, `checkpoint`, `finished` — which makes it usable as a child process of a
launcher that wants to follow a reconstruction. Use `--event-file` to send that stream to a
file and leave standard output alone.

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
