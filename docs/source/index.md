# Ptychodus documentation

```{image} ptychodus.svg
:alt: Ptychodus Logo
:align: center
:width: 10em
```

[Ptychodus](https://github.com/AdvancedPhotonSource/ptychodus) is a ptychography data analysis application that reads instrument data, prepares the data for processing, and supports calling several reconstruction libraries for phase retrieval. Ptychodus can be used interactively or integrated into a data pipeline.

```{toctree}
:maxdepth: 2
:caption: Contents:

getting_started
initial_guesses
fluorescence
xmcd
near_field
bragg_laue
readers
reconstruction_drivers
training
api
globus
stream
```

## Python API Example

Reconstruct a prepared directory, as a driver's `--prepare-only` writes one, with any installed reconstructor:

```python
from pathlib import Path

from ptychodus.api.io import load_diffraction_data, load_product, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.reconstruct import prepare_reconstruct_input

registry = PluginRegistry.load_plugins()
library = registry.reconstructor_libraries.get_strategy_by_name('PtyChi')
reconstructor = next(r for r in library if r.name == 'LSQML')
options = reconstructor.get_default_options()

assembled_data = load_diffraction_data(Path('prepared/diffraction.h5'))
product = load_product(Path('prepared/product.h5'))
reconstruct_input = prepare_reconstruct_input(assembled_data, product)

for output in reconstructor.reconstruct(reconstruct_input, options=options):
    print(f'{output.progress}/{output.progress_goal}')

save_product(Path('product.h5'), output.product)
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
