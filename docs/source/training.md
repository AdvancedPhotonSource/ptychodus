# Training Data

A learned reconstructor such as PtychoPINN trains from a directory of training data. `ptychodus-generate-training-data` writes that directory, from measured scans, from objects synthesized under a scan's probe, or from both, and `ptychodus-train` trains from it. This page walks through both commands; the [reconstruction drivers](reconstruction_drivers.md#training-a-model) page covers the rest of `ptychodus-train`.

## Writing training data

`ptychodus-generate-training-data` reads its input as `ptychodus-reconstruct` does: a prepared directory with `-i`, as a driver run with `--prepare-only` writes it, or a scan in a storage root with `--product UUID` (and `--storage-root`, or `$PTYCHODUS_WEBAPP_STORAGE_ROOT`). `--file-type` names the training-data format; the PtychoPINN and PtychoFM formats are the ones `ptychodus-train` reads.

```sh
ptychodus-generate-training-data -i ./scan -o ./training-data --file-type PtychoPINN
```

By default the command exports the measured scan: its patterns, paired with the product's positions by scan index exactly as a reconstruction pairs them, together with the product's probe and object. `--index-filter even` or `odd` exports half of the scan, as it does for `ptychodus-reconstruct`. A PtychoPINN export stores the product's object as `objectGuess`, which supervised training takes as the ground truth, so export a product holding a trusted reconstruction when training that way.

### Numbering and overwriting

Every file is named `<stem>_NNNN`, after the highest index already in the output directory; `--stem` defaults to `train_data`. A PtychoPINN file is `train_data_0000.npz`, and a PtychoFM training pair is `train_data_0000_dp.hdf5` with `train_data_0000_para.hdf5`. Running the command again adds files rather than replacing any, so exports of several scans, synthetic sets, or a mix of both can share one training directory.

Nothing is overwritten, even by runs into the same directory at the same time, such as the tasks of a job array. A run reserves an index by creating a hidden directory `.staging-<stem>_NNNN`, which only one run can create, has the writer write there, and links each file into place with a call that fails rather than replaces an existing name. The reservation is removed when the file is in place. One left behind by a killed run is harmless to numbering, but PtychoFM's training searches subdirectories, so delete stale `.staging-*` directories; the command warns about any it finds.

## Synthetic training data

`--synthetic STRATEGY` keeps the product's probe, positions and metadata, but replaces the object with one drawn from an object model, and simulates the patterns with the product's forward model: Poisson noise at the product's photon scale, and an ideal detector with no bad pixels, background or flat field. Only the product is read, so `diffraction.h5` is not needed. Any object model that needs no measured data is a strategy; the [initial guesses](initial_guesses.md#object-generators) page lists them with their parameters.

```sh
ptychodus-generate-training-data --print-options dead_leaves > leaves.yaml
ptychodus-generate-training-data -i ./scan -o ./training-data --file-type PtychoPINN \
    --synthetic dead_leaves --count 100 --options leaves.yaml --seed 1
```

`--print-options` writes a strategy's options at their defaults, and `--options` passes an edited copy back; fields it omits keep their defaults, and a document written for another strategy is refused. `--count` draws that many independent objects, one file each. Each file's random stream comes from `--seed` and its own index, so any file can be reproduced alone, and a later run with the same seed adds new data under new indexes. Without `--seed` the command draws one and logs it.

`--position-jitter-m` adds an independent Gaussian offset with that standard deviation, in meters, to each position along each axis. The files record the perturbed positions, the ones the patterns were simulated at. The object canvas widens about its center as far as the perturbed positions need, so no probe footprint is clipped.

### The dead-leaves model

`dead_leaves` gives objects with the statistics of natural images: scale-invariant structure, sharp occluding edges, and flat regions of every size. Leaves are disks dropped by a Poisson process, each hiding whatever lies beneath it.

- Radii have density proportional to `r ** -leaf_radius_power_law_exponent` between `leaf_radius_lower_px` and `leaf_radius_upper_px`, in pixel widths. An exponent of three makes the field scale invariant, and leaves are circular in physical space, also on non-square pixels.
- Each leaf's transmission has a phase uniform between `leaf_phase_lower_rad` and `leaf_phase_upper_rad`, and an amplitude whose square, the intensity transmission, is uniform between the squares of `leaf_amplitude_lower` and `leaf_amplitude_upper`.
- Each pixel is the mean complex transmission over `num_subsamples_per_axis` squared jittered points, the box-filtered sample of the continuous field. A complex mean treats every global phase alike, so edges carry no cue to the absolute phase, which phase retrieval cannot determine.

The sample is exact rather than an approximation that improves with more leaves. Leaves are painted front to back, so what the window shows is final once the window is covered, and when few points remain uncovered only leaves that cover one are drawn.

## Training PtychoPINN

`ptychodus-train` reads every PtychoPINN training archive (`*.npz`) directly in its input directory; other files are skipped with a warning.

```sh
ptychodus-train --print-options PtychoPINN/Unsupervised > pinn.yaml
ptychodus-train -i ./training-data -o ./trained --options pinn.yaml
```

A directory holding one archive is trained from as it is. PtychoPINN trains from one archive, so several are first merged into one object bank, the form PtychoPINN's own synthetic pipeline uses for several objects:

- each archive becomes one object, with its patterns tagged by `object_index`, so neighbor groups never mix objects;
- supervised training cuts each frame's ground-truth patch from that archive's `objectGuess` with PtychoPINN's own patch extraction, while unsupervised training needs none;
- the merged bank is held in memory, once per process under `torchrun`, in a temporary directory under the output directory that is removed afterwards.

The bank stores one probe, so a directory must hold data sharing one probe: a measured export and the synthetic sets made from the same product qualify. Archives that disagree on the probe, the pattern size or the declared measurement scale are refused rather than mixed; train scans with different probes from separate directories.

PtychoFM reads its own directory of `_dp`/`_para` training pairs, so the same workflow with `--file-type PtychoFM` and `ptychodus-train --print-options PtychoFM` trains it.
