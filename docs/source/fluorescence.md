# Fluorescence Enhancement

Ptychodus can sharpen an X-ray fluorescence dataset measured alongside a ptychography scan. The fluorescence maps are recorded at the scan resolution, which is limited by the probe size, while the ptychographic reconstruction resolves the probe itself and the object on a much finer grid. An enhancer combines the two to estimate each element map on the object grid.

## Running an enhancement

From the workflow API, open a product and enhance a fluorescence file against it:

```python
from pathlib import Path

from ptychodus.model import ModelCore

with ModelCore(Path('settings.ini')) as model:
    product_api = model.workflow_api.load_product(Path('product.h5'), file_type='HDF5')
    product_api.enhance_fluorescence_local(
        Path('fluorescence.h5'),
        Path('fluorescence_enhanced.h5'),
        algorithm='vspi',
        algorithm_parameters={'max_iterations': 50},
        block=True,
    )
```

`algorithm` selects an enhancer by the name it is registered under, and `None` runs the one the settings select. `algorithm_parameters` overrides that enhancer's options by field name for this run only: the settings are left as they were, and a name the enhancer does not declare is logged and ignored.

In batch mode, an input directory holding `fluorescence.h5` is enhanced with the enhancer the settings select once the reconstruction finishes.

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
