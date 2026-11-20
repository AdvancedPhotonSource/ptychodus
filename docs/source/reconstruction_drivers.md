# Reconstruction Drivers

Ptychodus ships a reconstruction command per beamline. Each one reads that instrument's raw diffraction and position files, builds a product, runs it through pty-chi, and writes the result as a standard-layout directory. They need no GUI and no settings file, so they suit a job scheduler, a data-management pipeline, or a terminal on a beamline workstation.

Every driver is installed as a console command by `pip install ptychodus`.

## Choosing a driver

Ten commands differ only in which instrument they read. The last four differ in shape: they take their configuration from a file, or operate on a directory another driver prepared.

| Command | Instrument | Diffraction input | Position input |
| --- | --- | --- | --- |
| `ptychodus-reconstruct-2ide` | APS 2-ID-E XFM | `fly###_data_NNN.h5` series | EPICS MDA, `mda/2xfm_NNNN.mda` |
| `ptychodus-reconstruct-12ide` | APS 12-ID-E Ptycho-SAXS | `<sample>_<scan>_master.h5` or any frame file (2026_2 layout onward) | The same master; positions are read from it |
| `ptychodus-reconstruct-26idc` | CNM/APS 26-ID-C Hard X-ray Nanoprobe | Any `scan_N_img_F.tif`/`.tif.gz` frame, or `scan_N_F.h5` member | EPICS MDA, `26idbSOFT_NNNN.mda` |
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

A scan that has already been preprocessed by fold_slice goes through `ptychodus-reconstruct-foldslice` whatever instrument recorded it, because the pattern file carries its own geometry.

## Common arguments

The ten instrument commands share one argument set. Three paths are required:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054
```

Only one member of a multi-file detector series need be named; the reader globs the rest.

The remaining arguments fall into four groups.

`--crop-extent-px`, `--beam-center-x-px`, `--beam-center-y-px`
: Cut a square region out of each raw frame about the beam centre. The centre comes from the flags, then from the data file when it records one, and is estimated from the patterns otherwise. Cropping is usually what decides reconstruction time, since it sets the array size. Left out, the crop is the extent of `--probe-file` when one is given, so a warm start from an earlier scan matches it; otherwise it is `auto`, the smallest power-of-two side that holds the mean pattern's signal above its noise floor, capped at the largest that fits. When nothing rises above the noise floor, an omitted crop reads the whole frame with a warning, while an explicit `auto` is an error. Pass a side in pixels to choose one, or `full` to read the whole frame. `crop_extent_px` in a batch settings file takes the same values and behaves the same way, sized from `initial_probe.probe_from_file` when left out.

`--detector-distance-m`, `--photon-energy-eV`, `--detector-pixel-size-m`, `--probe-photon-count`, `--tomography-angle-deg`
: Override metadata the data file records wrongly or not at all. Each is used only when given. The last three are offered only by the commands whose file layout needs them.

`--min-detector-sum`, `--max-valid-count`, `--bad-pixels-file`
: Drop patterns whose detector sum falls below a floor, zero pixels at or above a ceiling, and apply a bad-pixel mask covering the full uncropped detector. The ceiling defaults to the detector's invalid-pixel marker: the pattern dtype's maximum, which is how Dectris detectors mark an invalid pixel, except where the instrument names a lower one for the dtype. `ptychodus-reconstruct-lamni` stores 12-bit Eiger frames as uint16 and marks them with 4095.

`--background-file`, `--flat-field-file`, `--subtract-background`, `--divide-flat-field`
: Supply a measured detector background and flat-field response covering the full detector, overriding any the data file carries, and choose whether to correct the patterns with them or carry them alongside. `--background-file-type` and `--flat-field-file-type` name the readers. See [Detector Background and Flat Field](readers.md#detector-background-and-flat-field).

`--num-probe-modes`, `--num-opr-modes`, `--fzp-preset`, `--object-padding-px`, `--seed`
: Shape the initial guesses. The zone-plate preset describes the optic that formed the probe.

`--focus-object-distance-m`
: The signed focus position, negative upstream of the object. Its negation is the defocus every model probe is built at, the average-pattern cold start included, and in near field it sets the magnification besides. Unset, it is read from the data file; failing that, far field estimates it from the data (below) and near field stops with an error, since nothing measures it there (0 states a parallel beam).

`--parallax-bin-size-px`, `--parallax-min-significance`, `--parallax-min-defocus-dof`
: When far-field data states no focus, the defocus is measured from the parallax between bright-field images, as the GUI's **Estimate Defocus** does (see [Measuring the Defocus](initial_guesses.md#measuring-the-defocus)), at the cost of one binned pass over the patterns. It is applied only when resolved by the significance and depth-of-focus thresholds; otherwise the focus is zero and a warning says so. Stating `--focus-object-distance-m`, 0 included, skips the measurement.

Five more control the run itself: `--options` chooses the reconstruction algorithm and its settings, `--num-sync-epochs` overrides how often the run checkpoints and reports progress, `--dry-run` reports the resolved configuration without reconstructing, `--prepare-only` writes a prepared directory and stops (see [Preparing and reconstructing separately](#preparing-and-reconstructing-separately)), and `--log-level` takes a Python logging level as an integer (`10` for DEBUG, `20` for INFO).

```{note}
A profile can add or remove arguments. `ptychodus-reconstruct-foldslice` has no cropping, bad-pixel, background or flat-field options, because it reads patterns that are already prepared. `ptychodus-reconstruct-polar` adds statistical pattern filters. Run any command with `--help` for the list it actually accepts.
```

## The output directory

Every driver writes one directory in the same layout, whatever instrument it read:

```text
fly054/
  diffraction.h5         # the assembled patterns, as reconstructed
  product.h5             # the final object, probe and positions
  product.000100.h5      # a checkpoint every num_sync_epochs epochs
  reconstruct_options.yaml  # the options the run was given
  reconstruct.log        # every log record the run wrote
```

`product.h5` is the reconstructed product: it carries the object, the probe, the refined probe positions, and the metadata the reconstruction used. It opens in the GUI and reads back through the Python API. `ptychodus-enhance-xrf` takes it with a fluorescence dataset from the same scan, as {doc}`fluorescence` describes.

`reconstruct_options.yaml` records the options the run was given, as a document naming the library and algorithm, and can be passed straight back to `--options` to repeat the run. The values pty-chi derives from the product are folded in when the reconstruction starts, and the options it actually ran with are written to `reconstruct.log`. A log is appended to rather than replaced, and each process of a multi-process launch writes its own, `reconstruct.rank01.log` and so on.

`diffraction.h5` is written before the reconstruction starts, so an interrupted run still leaves the one file that cannot be rebuilt without the raw files. It can run to several gigabytes; `--no-save-diffraction` skips it.

`ptychodus-reconstruct-batch-lamni` writes one such directory per scan, and adds `batch_settings.yaml` and `affine_fit.yaml` to each. The fit's `scale`, `asymmetry`, `rotation_deg` and `shear_deg` follow pty-chi's convention for the same transform, in `(x, y)`: a positive rotation turns x towards y, a positive asymmetry magnifies y relative to x, and a positive shear moves x in proportion to y. `linear_transform` holds the matrix itself.

A driver run with `--prepare-only` writes a smaller directory, which holds what a reconstruction starts from rather than what it produced:

```text
fly054-prepared/
  diffraction.h5         # the assembled patterns
  product.h5             # the initial object, probe and positions
  reconstruct_options.yaml  # the options the reconstruction will run with
  reconstruct.log        # every log record the preparation wrote
```

## Choosing the algorithm

With no `--options`, a driver runs stock LSQML, or BH for a near-field product. To run anything else, or to change any reconstruction setting, pass an options document.

Obtain one by asking a driver to print the defaults:

```sh
$ ptychodus-reconstruct-2ide --print-options > lsqml.yaml    # LSQML, the default
$ ptychodus-reconstruct-2ide --print-options DM > dm.yaml    # a different algorithm
```

The algorithm may be `DM`, `RAAR`, `PIE`, `ePIE`, `rPIE`, `LSQML`, `Autodiff` or `BH`. The flag needs no other arguments, so any driver can produce the file.

Edit the result and pass it back:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054 \
    --options ./lsqml.yaml
```

The document names its library and algorithm at the top, beside the options themselves: the pty-chi task options under `task`, and ptychodus's own `num_sync_epochs` and `compact_mode_update_clustering`. Fields the document omits keep their defaults, and a field the algorithm does not declare is refused. The task also carries pty-chi's `options_class_name` stamps, which must agree with the algorithm, so change the algorithm by printing a fresh document rather than by editing those names.

```{note}
Values describing the data — the wavelength, the object and probe pixel sizes, the slice spacings, the propagation distance and the probe power — are written from the product when the reconstruction starts, whatever the document says. The comment above `task` in a printed document lists them. Editing them changes nothing except to add a warning to the log.

A hand-written file must also spell scientific notation `1.0e-3` rather than `1e-3`: YAML reads the shorter form as a string. Documents produced by `--print-options` are already correct.
```

`-` reads the document from standard input.

## Reconstructing a whole tomogram

`ptychodus-reconstruct-batch-lamni` reconstructs a range of LamNI projections in one command. It takes all of its settings from a YAML file, so a run is reproducible from something that can be archived, and it copies that file into each scan's output directory.

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

The experiment folder holds `dat-files`, `scan_positions` and `eiger_4` either under a `data` subdirectory or directly at its top level; when both are present, `data` wins.

```sh
$ ptychodus-reconstruct-batch-lamni --batch-settings batch.yaml --dry-run
$ ptychodus-reconstruct-batch-lamni --batch-settings batch.yaml --options lsqml.yaml
```

Start with `--dry-run`: it resolves the scan table and reports every scan it would reconstruct, with the files it would read and the directory it would write, without reading any data.

Scans run concurrently, one worker per visible CUDA device, so the job scheduler decides the degree of parallelism through `CUDA_VISIBLE_DEVICES`. A scan whose `product.h5` already exists is skipped, which is how an interrupted run resumes — re-run the same command.

### Driving it from Python

The command is a shim over a function that takes the same two things as objects: a `BatchSettings` and pty-chi options, a `PtyChiOptions` such as `default_options` returns. A script that already holds those — one sweeping a reconstruction parameter over a projection range, or building the settings from an experiment database rather than a YAML file — calls it directly instead of writing a temporary file and launching a subprocess.

```python
from ptychi.api import LSQMLOptions

from ptychodus.plugins.ptychi.options import default_options
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

options = default_options(LSQMLOptions)
options.task.reconstructor_options.num_epochs = 200

result = reconstruct_batch_lamni(settings, options)
print(result.succeeded, result.skipped, result.failed)
```

The sections are the blocks of the YAML file, one class each, and every one of them is importable from the same module. An omitted section takes the same defaults the file does, so only `paths` is required.

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

The call blocks until the batch finishes and writes exactly what the command writes: one standard-layout directory per scan, each with its own `batch_settings.yaml`. That archived copy is rendered from the settings object the run was given — every default written out, and no comments — because an object is all a Python caller has. The command passes its file's text through instead, so a run from a file archives the file; a caller that wants the same can pass the text as `settings_text`.

A scan that fails is reported rather than raised, because a tomogram with a projection missing is a result its owner has to judge: `BatchResult` carries the scan numbers that succeeded, were skipped, failed, or were cancelled, and `result.exit_code` is the status the console command would have exited with. Anything that goes wrong before the first scan starts — an experiment directory that is not there, a scan range that selects nothing, a crop that runs off the detector — raises instead, since it describes every scan equally.

```{note}
The fan-out spawns one child process per scan, and a child spawned from a process that already holds a CUDA context fails to initialize. Do not run torch on a GPU in the same interpreter first: this is the same constraint that makes ptychodus enumerate devices in a throwaway child.
```

Neither function configures logging — nothing is printed until the caller sets logging up, for example with `logging.basicConfig(level=logging.INFO)` — and neither installs signal handlers. To make a run interruptible, pass a `CancellationToken`; it stops the batch at the first scan boundary after the request and leaves the scans already written in place.

## Preparing and reconstructing separately

Reading and assembling the raw files and running pty-chi need not happen in one command. Give any of the ten instrument commands `--prepare-only` and it stops after building the initial product: it writes the prepared directory shown above and exits successfully without reconstructing. `ptychodus-reconstruct` then runs the reconstruction from that directory:

```sh
$ ptychodus-reconstruct-2ide \
    --diffraction-file /data/2ide/ptycho/fly054_data_001.h5 \
    --position-file /data/2ide/mda/2xfm_0054.mda \
    --output-directory ./fly054-prepared \
    --prepare-only
$ ptychodus-reconstruct -i ./fly054-prepared -o ./fly054
```

The split lets the preparation run where the raw data are, on a beamline workstation, and the reconstruction run where the GPUs are. It also lets one scan be reconstructed several times with different options without assembling the raw data again, which for a large scan is most of the wall-clock time. Every argument that shapes the data or the initial guesses — cropping, filtering, geometry, `--seed`, the probe options, `--options` and `--num-sync-epochs` — takes effect at preparation, and the options are carried into the prepared directory.

`ptychodus-reconstruct` reads no raw beamline files and knows nothing about any instrument. It takes its options from the prepared directory's `reconstruct_options.yaml` unless `--options` names another document, and falls back to stock pty-chi LSQML, or BH for a near-field product, when there is neither. The document names the library, so the same prepared directory reconstructs with any installed library — `--print-options cdtools` writes cdtools' defaults, for one — and a trainable reconstructor loads the `model` file the directory holds. The input directory is never modified, and the output must be a different directory, since the final `product.h5` would otherwise overwrite the initial one. `--dry-run` loads and pairs the inputs and reports what would run.

`ptychodus-reconstruct-subprocess` reconstructs the same input for a launching process. It reports progress as newline-delimited JSON — one object per line: `started`, naming the `library` and `algorithm`; `epoch`, for each output with its `num_epochs` goal; `checkpoint`; `finished` — which makes it usable as a child process of a launcher that wants to follow a reconstruction. Use `--event-file` to send that stream to a file and leave standard output alone.

## Reconstructing from a PEAR parameter file

`ptychodus-reconstruct-pear` reads the flat JSON parameter file that PEAR's batch loop writes, and reconstructs the single scan it describes. The parameter file names the instrument, the input paths and every reconstruction setting, so the command takes almost no arguments of its own and has no `--options`: the algorithm settings come from the parameter file too.

```sh
$ ptychodus-reconstruct-pear --params ./scan0054_params.json --dry-run
```

The output directory is derived from the parameter file, following the same convention PEAR uses, so results land where that workflow expects them.

## Training a model

`ptychodus-train` trains a learned reconstructor such as PtychoPINN from a directory of training data in the library's own format, as the GUI's export or `ptychodus-generate-training-data` writes it — every PtychoPINN archive (`*.npz`) in the directory, merged into one object bank when there are several, or PtychoFM's `_dp`/`_para` pairs. It reads no raw beamline files and no settings file, and leaves the input directory untouched. [Training Data](training.md) shows how to write such a directory from measured scans and synthetic objects.

Its options come from a YAML document, as `ptychodus-enhance-xrf`'s do. `--print-options` writes one at the defaults, naming the library and the algorithm beside every field, and `--options` passes an edited copy back. Fields it omits keep their defaults; a misspelled field, or one the library fixes, is refused rather than ignored.

```sh
$ ptychodus-train --print-options PtychoPINN/Unsupervised > pinn.yaml
$ ptychodus-train -i ./training-data -o ./trained --options pinn.yaml
```

The library writes what it trains into the output directory, and the command adds the trained model there as `model` with the library's extension, such as `model.zip` for PtychoPINN, beside `train_options.yaml`, the options it trained with, and `train.log`. `--dry-run` checks the inputs, options and launch without training.

To train on several GPUs of one node, launch it with `torchrun`:

```sh
$ torchrun --standalone --nproc-per-node=4 -m ptychodus.cli.train -i ./training-data -o ./trained --options pinn.yaml
```

Each rank runs the library in a child process that joins one process group, and only the first rank logs progress and saves the model; add `--local-ranks-filter 0` to see one rank's console. Choose the GPUs with `CUDA_VISIBLE_DEVICES`: every rank must see the same devices, as many as `--nproc-per-node`. PtychoPINN trains this way. A library that trains on one device only is refused under a launch of more than one process, and any launch across more than one node is refused.
