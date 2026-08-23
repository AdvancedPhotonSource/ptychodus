# Installation Instructions

## Python Package Index (PyPI)

To install ptychodus with the most common optional dependencies:

```sh
$ python -m pip install ptychodus[globus,gui,ptychi]
```

## uv

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Install ptychodus with the most common optional dependencies:

   ```sh
   $ uv tool install ptychodus[globus,gui,ptychi]
   ```

3. Launch ptychodus:

   ```sh
   $ ptychodus
   ```

4. To upgrade ptychodus, use uv tool upgrade:

   ```sh
   $ uv tool upgrade ptychodus[globus,gui,ptychi]
   ```

## Conda-Forge

1. Install [miniforge](https://github.com/conda-forge/miniforge).

2. Create the `ptychodus` environment

   - To install `ptychodus` with the GUI and all optional packages:

     ```sh
     $ conda create -n ptychodus ptychodus-all
     ```

   - To install `ptychodus` with the GUI and no optional packages:

     ```sh
     $ conda create -n ptychodus ptychodus
     ```

   - To install `ptychodus` without the GUI or optional packages:

     ```sh
     $ conda create -n ptychodus ptychodus-core
     ```

3. Activate the `ptychodus` environment

   ```sh
   $ conda activate ptychodus
   $ ptychodus
   ```

## Container image variants

The repository ships one Dockerfile per accelerator family. Pick the variant that matches your hardware and select an explicit file with `-f`:

| Dockerfile | Use it for |
| --- | --- |
| `Dockerfile.cpu` | CPU-only hosts (no GPU; ptychi runs on CPU torch) |
| `Dockerfile.cuda` | NVIDIA GPUs (e.g. ALCF Polaris, NERSC Perlmutter); CUDA minor version is a build ARG |
| `Dockerfile.rocm` | AMD GPUs (e.g. OLCF Frontier); ROCm is a build ARG |
| `Dockerfile.xpu` | Intel XPU (e.g. ALCF Aurora); base tag is a build ARG |

The GPU files default to recent versions and expose `--build-arg` knobs to switch:

- `Dockerfile.cuda`: `CUDA_VERSION` (default `13.0`), `PYTORCH_VERSION`, `CUDNN_VERSION`. The base image is `pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel`; override any args if a given combination isn't published upstream.
- `Dockerfile.rocm`: `ROCM_VERSION` (default `7.2.4`), `UBUNTU_VERSION`, `PYTHON_VERSION`, `PYTORCH_VERSION`. Base image is `rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PYTHON_VERSION}_pytorch_release_${PYTORCH_VERSION}`.
- `Dockerfile.xpu`: `BASE_TAG` (default `latest`). Base image is `intel/intel-optimized-pytorch:${BASE_TAG}`; pin to a dated tag for reproducibility.

## Podman

The repository ships [scripts/podman/build](../../scripts/podman/build), a helper that builds the CPU image plus the three CUDA variants in one shot and threads the checkout's PEP 440 version (from `setuptools_scm`) in as `--build-arg PTYCHODUS_VERSION=…` so `ptychodus --version` inside the container reports the real value instead of the `setuptools_scm` fallback.

```sh
$ scripts/podman/build                       # cpu + cuda12.8 + cuda13.0 + cuda13.2
$ scripts/podman/build cpu cuda13.0          # subset
$ PTYCHODUS_VERSION=1.5.1 scripts/podman/build cuda13.0    # explicit version
$ CONTAINER_ENGINE=docker scripts/podman/build cpu         # use docker instead
```

To build a variant not covered by the helper (ROCm, XPU, or a CUDA/PyTorch pair not in the matrix), invoke podman directly and pass the version yourself:

```sh
$ VER=$(uv run --with setuptools-scm python -m setuptools_scm)
$ podman build --build-arg PTYCHODUS_VERSION="$VER" -f Dockerfile.rocm -t ptychodus:rocm .
$ podman build --build-arg PTYCHODUS_VERSION="$VER" -f Dockerfile.xpu  -t ptychodus:xpu  .
```

Run container

```{note}
GPU access requires CDI (Container Device Interface) to be configured on the host. Run `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` once before using `--device nvidia.com/gpu=all`.
```

```sh
$ xhost +local:podman
$ podman run -it --rm --env DISPLAY --security-opt label=type:container_runtime_t --network host \
    --device nvidia.com/gpu=all ptychodus:cuda13.0
$ xhost -local:podman
```

## Docker

Build Docker image with the helper (works with either engine):

```sh
$ CONTAINER_ENGINE=docker scripts/podman/build cuda13.0
```

or directly:

```sh
$ VER=$(uv run --with setuptools-scm python -m setuptools_scm)
$ docker build --build-arg PTYCHODUS_VERSION="$VER" -f Dockerfile.cuda -t ptychodus:cuda13.0 .
```

Run container

```{note}
GPU access requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to be installed on the host before using `--gpus all`.
```

```sh
$ xhost +local:docker
$ docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
      --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ptychodus:cuda13.0
$ xhost -local:docker
```

## Apptainer / Singularity

The images above are OCI-compliant and can be converted to SIF for HPC sites (NERSC, OLCF, ALCF) that prefer Apptainer. After building an OCI image locally, convert it:

```sh
$ apptainer build ptychodus-cuda13.0.sif docker-daemon://localhost/ptychodus:cuda13.0
```

If the image lives in a registry, pull directly with `docker://` instead.

## Wrapper Script (Beamline Workstations)

The repository ships `scripts/podman/ptychodus`, a small bash wrapper that makes the containerized application behave like a native `ptychodus` command. It auto-selects the image variant for the host's GPU, forwards X11 to the user's desktop, bind-mounts `$HOME`, `$PWD`, and the beamline data roots (`/local`, `/gdata`) into the container, and runs rootless so that files written to those mounts are owned by the invoking user. All command line arguments are passed through to `ptychodus` unchanged.

### Prerequisites

The wrapper assumes facility IT has already provisioned the host:

- Rootless podman is installed and works for the beamline account (`podman info` succeeds without `sudo`). The wrapper itself does **not** install podman.
- For NVIDIA hosts, `/etc/cdi/nvidia.yaml` exists (generated by `nvidia-ctk cdi generate` — a one-time root step; see the note above).
- For AMD hosts, the beamline account is a member of the `video` and `render` groups.
- The image variant for the host's GPU has been built into the user's rootless podman storage using the build commands in the **Podman** section above. No root is required for `podman build`.

### Install (userspace, no sudo)

Run from the repository root as the beamline account:

```sh
$ install -Dm 0755 scripts/podman/ptychodus ~/.local/bin/ptychodus
$ which ptychodus            # should print ~/.local/bin/ptychodus
```

If `which` resolves to a conda environment's launcher instead (for example `~/miniconda3/envs/ptychodus/bin/ptychodus`), prepend `~/.local/bin` to `PATH` in the account's shell rc:

```sh
$ export PATH="$HOME/.local/bin:$PATH"
```

For a site-wide install, copy the script to any directory the shared beamline account can write to (for example `/opt/beamline/bin` if pre-provisioned by IT) and have each user's `PATH` include it. The wrapper does not care about its install location.

Sanity check:

```sh
$ ptychodus --version
```

stderr will show a one-line notice such as `ptychodus: using image ptychodus:cuda13.0 (detected: nvidia)`; stdout prints the version reported by the in-container `ptychodus`.

### Usage

```sh
$ ptychodus                                          # GUI, auto-detect GPU
$ ptychodus -s settings.ini                          # GUI with settings
$ ptychodus -b reconstruct -i ./input -o ./output    # headless batch
$ PTYCHODUS_IMAGE=ptychodus:cpu ptychodus            # force CPU image
$ PTYCHODUS_QUIET=1 ptychodus -v                     # silent, version only
```

File paths passed via `-i`, `-o`, `-s`, or as positional arguments resolve naturally as long as they live under `$HOME`, the current directory, or one of the auto-mounted beamline roots (`/local`, `/gdata`).

### Environment overrides

| Variable | Effect |
| --- | --- |
| `PTYCHODUS_IMAGE` | Skip GPU auto-detection; use this image tag verbatim. |
| `PTYCHODUS_QUIET` | Suppress the stderr "using image" notice. |

The image tag map and the list of beamline mounts live near the top of `scripts/podman/ptychodus` and can be edited in place to retag or add sites (for example `/data`, `/nsls2`).

### Troubleshooting

- **"image … not found in rootless podman storage"** — run the `podman build` command printed in the error.
- **"cannot open display"** — confirm `$DISPLAY` is set in the shell, then run `xhost +local:` once per X session. Wayland desktops work via XWayland as long as `$DISPLAY` is set (the default on GNOME and KDE).
- **"podman: command not found"** — ask facility IT to install rootless podman.
- **Wrong GPU family detected** — override with `PTYCHODUS_IMAGE=ptychodus:cpu` (or any other tag).
- **Files written by the container are owned by root** — rootless `--userns=keep-id` is not in effect; check `podman info` for `rootless: true`.
- **"permission denied" on a path argument** — the path is outside the bind-mounted set. Run from under `$HOME` or one of the beamline roots, or add the path to `BEAMLINE_MOUNTS` at the top of the wrapper.

## VS Code Dev Container

The repository includes a [Dev Container](https://containers.dev/) configuration at `.devcontainer/devcontainer.json` that builds from `Dockerfile.cpu` — the safe default for contributors without a local GPU. GPU users can edit the `dockerfile:` field to point at `Dockerfile.cuda` (or `Dockerfile.rocm`) before reopening in the container.

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension for VS Code.

2. Open the repository folder in VS Code, then choose **Dev Containers: Reopen in Container** from the Command Palette ({kbd}`Ctrl+Shift+P`).

VS Code will build the image and reopen the workspace inside the container.

## For Maintainers: Publishing Releases

The version is derived from the git tag by `setuptools-scm` — there is no version string to edit anywhere in the tree. Tag first, then build.

The `ptychodus-store` web UI is compiled TypeScript that is not tracked in git. Both the `sdist` and `build_py` steps compile it, so the release sdist and wheel each carry `ptychodus_store/ui/dist/`, and consumers installing from PyPI never need Node. Setting `PTYCHODUS_STORE_REQUIRE_UI_BUILD=1` turns a missing or stale UI into a build failure instead of a warning — always set it when publishing, so a release can never ship a stale or empty UI.

### Prepare

1. Confirm the release tag is on `HEAD` and no tracked file is modified:

   ```sh
   $ git describe --tags --dirty --match 'v[0-9]*'
   ```

   A `-dirty` suffix or a commit distance means the artifacts will carry a development version, not the release version.

2. Remove stale build artifacts. `setuptools` reuses `build/lib/`, so a leftover tree from an earlier build can ship outdated modules:

   ```sh
   $ rm -rf build dist src/*.egg-info
   ```

3. Put `tsc` on `PATH`. On a host with no Node.js, bootstrap one with `nodeenv`:

   ```sh
   $ uv tool install nodeenv
   $ nodeenv --node=lts --prebuilt ~/.local/node-lts
   $ export PATH="$HOME/.local/node-lts/bin:$PATH"
   $ npm install -g typescript
   ```

   See the "Rebuild the frontend" section of `src/ptychodus_store/README.md` for the UI development workflow.

### Build and verify

1. Build the sdist and wheel into `./dist/`. `--no-sources` is required: `[tool.uv.sources]` points `ptychopinn` at a local checkout, which must not leak into the published metadata.

   ```sh
   $ PTYCHODUS_STORE_REQUIRE_UI_BUILD=1 uv build --no-sources
   ```

   The equivalent with `build` is `PTYCHODUS_STORE_REQUIRE_UI_BUILD=1 python -m build`.

2. Confirm the compiled UI is in **both** artifacts. Each count must equal the number of TypeScript sources:

   ```sh
   $ tar -tzf dist/ptychodus-*.tar.gz | grep -c 'ui/dist/.*\.js$'
   $ unzip -Z1 dist/ptychodus-*.whl | grep -c 'ui/dist/.*\.js$'
   $ find src/ptychodus_store/ui/src -name '*.ts' | wc -l
   ```

3. Confirm the metadata renders on PyPI:

   ```sh
   $ uvx twine check dist/*
   ```

### Publish

Uploading is irreversible — PyPI does not allow reusing a filename, even after a release is deleted.

```sh
$ uv publish
```

`uv publish` reads a PyPI API token from `UV_PUBLISH_TOKEN`, or accepts `--token`. The equivalent with `twine` is:

```sh
$ python -m twine upload --verbose dist/*
```
