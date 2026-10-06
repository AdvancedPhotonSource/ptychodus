# Ptychodus

[Ptychodus](https://github.com/AdvancedPhotonSource/ptychodus) is a ptychography data analysis application that extracts, loads, and transforms instrument data for processing. It integrates several reconstruction libraries for phase retrieval. Ptychodus can be used interactively or integrated into beamline data pipelines.

## Standard Installation

To install ptychodus from PyPI with the most common optional dependencies:

```sh
$ python -m pip install ptychodus[globus,gui,ptychi]
```

Instructions for installing in containers, uv, and from conda-forge are provided in the `docs` directory.

## Developer Installation

For a developer installation:

```sh
$ git clone https://github.com/AdvancedPhotonSource/ptychodus.git
$ cd ptychodus
$ uv sync --extra globus --extra gui --extra ptychi
```

The `ptychi` extra installs a **CPU** build of PyTorch. For GPU reconstruction, swap it for the extra matching your driver's CUDA ceiling — `nvidia-smi` reports that ceiling on its banner line:

```sh
$ uv sync --extra globus --extra gui --extra ptychi-cuda128   # or -cuda130, -cuda132
```

A build newer than the driver leaves `torch.cuda.is_available()` false and reconstruction silently falls back to the CPU; `uv run ptychodus-system-check` reports the driver ceiling, the PyTorch build and whether the two agree. These extras pin PyTorch to the matching index through `[tool.uv.sources]`, which only `uv` reads — a `pip install` of the same extra takes whatever PyTorch PyPI offers.

Launch `ptychodus`:

```sh
$ uv run ptychodus
```

## Reporting Bugs

Open a bug at <https://github.com/AdvancedPhotonSource/ptychodus/issues>.
