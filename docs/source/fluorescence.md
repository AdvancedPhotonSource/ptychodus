# Fluorescence Enhancement

Ptychodus can sharpen an X-ray fluorescence dataset measured alongside a ptychography scan. The fluorescence maps are recorded at the scan resolution, which is limited by the probe size, while the ptychographic reconstruction resolves the probe itself and the object on a much finer grid. An enhancer combines the two to estimate each element map on the object grid.

## Running an enhancement

`ptychodus-enhance` enhances a fluorescence dataset against a product file, such as the `product.h5` that `ptychodus-reconstruct` writes:

```sh
$ ptychodus-enhance --product out/product.h5 --fluorescence 2xfm_0123.mda.h5 \
    --algorithm vspi -o out/fluorescence.h5
```

The enhanced dataset is written in the format the measured one was read in (`--file-type`, `XRF-Maps` by default), and the output must differ from both inputs. Nothing is written unless the enhancer runs to completion, so an interrupted run leaves no partially enhanced file. `--dry-run` reads and checks the inputs and options without enhancing.

An enhancer runs at its default options unless an options file is given. `--print-options` writes one, naming the enhancer and its fields by their declared names:

```sh
$ ptychodus-enhance --print-options vspi > vspi.yaml
$ ptychodus-enhance --product out/product.h5 --fluorescence 2xfm_0123.mda.h5 \
    --options vspi.yaml -o out/fluorescence.h5
```

```yaml
# vspi fluorescence enhancer options, at their defaults. Save this,
# edit it, and pass it back with --options.
algorithm: vspi
options:
  # Tikhonov regularization of the least-squares solve; zero applies none.
  damping_factor: 0.0
  # Upper bound on LSMR iterations per element map.
  max_iterations: 100
```

The file names its own enhancer, so `--algorithm` is not needed beside it; passing a different one is an error. Fields the file omits keep their defaults, and a field the enhancer does not declare is an error rather than being ignored.

## Enhancers and their settings

`[Fluorescence]` holds what to enhance and which enhancer to run:

| Key | Meaning |
| --- | --- |
| `FilePath` | the measured fluorescence dataset |
| `FileType` | the reader that dataset is opened with |
| `Algorithm` | the enhancer to run |

Each enhancer declares its own options, and they persist in a group of that enhancer's own, so no two enhancers share a key. The enhancers available depend on what is installed; these ship with ptychodus.

### `two_step` — `[FluorescenceTwoStep]`

Each element map is spread from the scan positions onto the object grid, and the probe intensity is then deconvolved from it.

| Key | Default | Meaning |
| --- | --- | --- |
| `UpscalingMethod` | `linear` | `identity`, `nearest`, `linear`, `cubic`, or a radial basis function: `linear_rbf`, `thin_plate_spline_rbf`, `cubic_rbf`, `quintic_rbf` |
| `DeconvolutionMethod` | `richardson_lucy` | `identity`, `richardson_lucy`, `wiener`, `unsupervised_wiener` |
| `WienerBalance` | `0.05` | regularization of the Wiener filter, trading noise against sharpness |

### `vspi` — `[FluorescenceVSPI]`

Virtual single pixel imaging: each element map is solved for on the object grid by least squares (LSMR) against a forward model that weights the object by the probe intensity at every scan position.

| Key | Default | Meaning |
| --- | --- | --- |
| `DampingFactor` | `0.0` | Tikhonov regularization; zero applies none |
| `MaxIterations` | `100` | upper bound on LSMR iterations per element map |

### `ptychozoon` — `[FluorescencePtychozoon]`

The same virtual single pixel imaging model as implemented by [ptychozoon](https://github.com/AdvancedPhotonSource/ptychozoon), a separately maintained package that runs the solve on a GPU and reports intermediate results as it goes. It is offered only where ptychozoon is installed (the `ptychozoon` extra), and its keys follow ptychozoon's own names.

| Key | Default | Meaning |
| --- | --- | --- |
| `DampingFactor` | `0.0` | Tikhonov regularization; zero applies none |
| `MaxIter` | `100` | total number of LSMR iterations |
| `Atol` | `1e-06` | LSMR stops once the relative residual falls below this |
| `Btol` | `1e-06` | LSMR stops once the relative residual of the normal equations falls below this |
| `CheckpointInterval` | `5` | report the solution every this many iterations |

```{note}
ptychozoon runs in a separate process on the devices selected by `[Processing] Devices`, the same selection the reconstructors use. It needs a CUDA device: selecting only the CPU leaves it nothing to run on, so use `vspi` to enhance on the CPU.
```
