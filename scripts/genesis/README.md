# Genesis Facility Scripts

14 April 2025

These scripts are not packaged; run them from a checkout with `ptychodus` installed:

```sh
python scripts/genesis/ptychodus_iri_tokens.py --help       # IRI facility API tokens
python scripts/genesis/ptychodus_transfer_tokens.py --help  # Globus transfer tokens
python scripts/genesis/<facility>/submit_job.py             # per-facility job submission
python scripts/genesis/olcf/check_token.py                  # introspect an OLCF token, query Odo status
```

`claude_token.py` is an importable helper module of Globus token getters — ALCF and NERSC IRI tokens, and ALCF, NERSC and OLCF transfer tokens; `olcf/example.sh` is a raw-curl walkthrough of the OLCF S3M API. The adapters that consume these tokens live in [facility_adapters.py](../../src/ptychodus/model/genesis/facility_adapters.py), which is the source of truth for each facility's Globus collection UUID and staging root. The IRI API that all three facilities speak is documented at <https://api.iri.nersc.gov/#/docs>.

## Facility Comparison

| | ALCF | NERSC | OLCF |
| --- | --- | --- | --- |
| System | Polaris | Perlmutter | Odo |
| Accelerator | NVIDIA A100 | NVIDIA A100 | AMD MI250X |
| Image variant | `containers/Dockerfile.cuda` | `containers/Dockerfile.cuda` | `containers/Dockerfile.rocm` |
| Container runtime | stock podman, or Apptainer | `podman-hpc` wrapper | stock podman, then Apptainer |
| Runs the container directly | yes | yes | no — convert to SIF first |
| Scheduler | PBS | Slurm | Slurm |
| IRI API endpoint | `https://api.alcf.anl.gov` | `https://api.iri.nersc.gov` | `https://amsc-open.s3m.olcf.ornl.gov` |
| Compute environment | conda env `ptychodus` | conda env `ptychodus` | conda prefix `/ccsopen/proj/<account>/ptychodus-env` |

Containers are for building and running by hand. Job submission still goes through the conda environment described under each facility below, because the adapters do not yet populate the container fields of their job specifications. The launcher `scripts/podman/ptychodus` is for beamline workstations — it forwards X11 and mounts `$HOME`, `/local` and `/gdata` — and is not usable at any of these facilities.

## Token Files

Tokens are stored under `~/.ptychodus`:

| File | Written by | Read by |
| --- | --- | --- |
| `iri_tokens.json` | `ptychodus_iri_tokens.py` | the facility adapters |
| `genesis_transfer_tokens.json` | `ptychodus_transfer_tokens.py` | the Globus transfer providers |

Both files must be **owner read/write only**. Ptychodus refuses to load a token file with looser permissions, and an unloadable token file **disables the facility adapters rather than stopping startup** — the facility list simply comes up empty, with the reason only in the log. After copying a token file between machines, restore the mode:

```sh
chmod 600 ~/.ptychodus/iri_tokens.json
```

## ALCF

### Get Tokens

Get [ALCF IRI API](https://api.alcf.anl.gov) access tokens using the instructions and scripts at <https://github.com/argonne-lcf/alcf-facility-api-token>.

The `scripts/genesis/alcf/globus_access_token.py` script helps you authenticate with Globus and obtain access tokens for filesystem operations. Authenticate with your ALCF account:

```sh
python scripts/genesis/alcf/globus_access_token.py authenticate
```

You can view your access token with:

```sh
python scripts/genesis/alcf/globus_access_token.py get_access_token
```

### Compute Environment

Submitted jobs activate a conda environment **named exactly `ptychodus`** — `ALCFFacilityAdapter` hardcodes `conda activate ptychodus`, so an environment under any other name fails at launch:

```sh
module use /soft/modulefiles
module load conda
conda create -n ptychodus python==3.11 pytorch torchvision
conda activate ptychodus
pip install -e ./ptychodus[ptychi]
```

PtychoPINN is optional and is published on no index; append `-e ./PtychoPINN` to the last command only if you have a checkout of it.

### Access Polaris

```sh
ssh USERNAME@polaris.alcf.anl.gov
```

Use passcode only.

### Containers

Polaris has NVIDIA A100 GPUs, so the matching image is [containers/Dockerfile.cuda](../../containers/Dockerfile.cuda); see [docs/source/getting_started.md](../../docs/source/getting_started.md) for how to build it. Every ptychodus Dockerfile copies the build context (`COPY . /src`), so building on Polaris needs a ptychodus checkout on that filesystem — otherwise build the image elsewhere and bring it over.

Stock podman — the podman that ships with the Linux distribution — is supported on Polaris and runs containers directly, with no conversion step. ALCF has no page covering that route, which is why the reference below describes only Apptainer.

Apptainer is the route ALCF documents: <https://docs.alcf.anl.gov/polaris/containers/containers/>. The constraints below belong to that route, not to containers on Polaris generally.

- Apptainer builds and runs on compute nodes only; request one with `qsub -I` and `-l singularity_fakeroot=true`, then pass `--fakeroot` to `apptainer build`.
- Load the modules with `ml use /soft/modulefiles`, `ml spack-pe-base` and `ml apptainer`.
- Point `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` at `/local/scratch/`; the default `~/.apptainer/cache` fills the home quota.
- Pulling a `docker://` image from a compute node needs `HTTP_PROXY` and `HTTPS_PROXY` set to `http://proxy.alcf.anl.gov:3128`; network virtualization is unavailable.
- Request the filesystems the job touches with `-l filesystems=home:eagle`, matching what `ALCFFacilityAdapter` already sets.
- Launch under PBS with `mpiexec … apptainer exec …`, not `srun`.

## NERSC

### Get Tokens

The script (`scripts/genesis/nersc/get_globus_token.py`) and instructions ([get_globus_token.md](nersc/get_globus_token.md)) are from <https://github.com/NERSC/iri-api-get-globus-token>.

Run the script and follow instructions to input the auth code:

```sh
python scripts/genesis/nersc/get_globus_token.py
```

Token JSON is saved to `~/.globus/auth_tokens.json`.

### Allocation

Use account "amsc013". qos name for GPU "express_amsc_g" and for CPU "express_amsc".

### Compute Environment

Submitted jobs activate a conda environment **named exactly `ptychodus`** — `NERSCFacilityAdapter` hardcodes `conda activate ptychodus`, so an environment under any other name fails at launch:

```sh
module load conda
conda create -n ptychodus python==3.11 pytorch torchvision
conda activate ptychodus
pip install -e ./ptychodus[ptychi]
```

PtychoPINN is optional and is published on no index; append `-e ./PtychoPINN` to the last command only if you have a checkout of it.

### Access Perlmutter

```sh
ssh USERNAME@perlmutter.nersc.gov
```

Use password + passcode.

### Containers

Perlmutter has NVIDIA A100 GPUs, so the matching image is [containers/Dockerfile.cuda](../../containers/Dockerfile.cuda); see [docs/source/getting_started.md](../../docs/source/getting_started.md) for how to build it. References: <https://docs.nersc.gov/development/containers/> and, for the runtime itself, <https://docs.nersc.gov/development/containers/podman-hpc/overview/>.

`podman-hpc` is NERSC's HPC-adapted wrapper, **not** the stock podman that ships with the distribution: it adds subcommands (`migrate`, `shared-run`, `infohpc`) and keeps images in its own squashfs store, so the OLCF commands below do not carry over. Shifter is the older path and still works for existing pipelines.

- `podman-hpc build` leaves an image the scheduler cannot see; **`podman-hpc migrate <tag>`** converts it to squashfs under `$SCRATCH/storage`. This is the step that is easy to miss.
- The build cache is local to the login node that built it, so another login node rebuilds from scratch.
- Nothing is bound into the container by default — `--gpu` is required for GPU access.
- `--network=host` works alongside most capability flags but **not** with `--gpu`.
- To share one image across a project, pull or migrate with `--squash-dir` and have consumers set `PODMANHPC_ADDITIONAL_STORES`; use the `/dvs_ro/cfs/…` spelling on compute nodes.
- Do not set `PYTHONPATH`; the wrapper is itself a Python program.

```sh
podman-hpc build -f containers/Dockerfile.cuda -t ptychodus:cuda .
podman-hpc migrate ptychodus:cuda
srun -n 1 -G 1 podman-hpc run --rm --gpu ptychodus:cuda python3 -m ptychodus --version
```

## OLCF

### Get Tokens

Generate a token at <https://docs.olcf.ornl.gov/services_and_applications/s3m/overview.html#generate-a-token>.

Use open enclave account along with the CSC682 project when you generate the token.

### Compute Environment

Submitted jobs activate a conda environment at the prefix `/ccsopen/proj/<account>/ptychodus-env`, where `<account>` is the Genesis account setting — `OLCFFacilityAdapter` builds the prefix from it, so the environment has to sit exactly there:

```sh
module load miniforge3
conda create --prefix /ccsopen/proj/csc682/ptychodus-env python==3.11
conda activate /ccsopen/proj/csc682/ptychodus-env
pip install -e ptychodus[ptychi]
```

### Access Odo

Instructions: <https://docs.olcf.ornl.gov/systems/odo_user_guide.html>

```sh
ssh USERNAME@login1.odo.olcf.ornl.gov
```

Use password only.

### Containers

Odo has AMD MI250X GPUs, so the matching image is [containers/Dockerfile.rocm](../../containers/Dockerfile.rocm); see [docs/source/getting_started.md](../../docs/source/getting_started.md) for how to build it. Reference: <https://docs.olcf.ornl.gov/software/containers_on_frontier.html> — written for Frontier, and applicable here because Odo shares Frontier's architecture.

Unlike ALCF and NERSC, OLCF does not support running podman directly: an image has to be converted to SIF and run under Apptainer.

- Podman needs subuid/subgid mappings enabled on your account. Request them by emailing <help@olcf.ornl.gov> before the first build, which fails without them.
- Create `$HOME/.config/containers/storage.conf` before building; the OLCF page above gives the snippet.
- `podman build` needs `--network host` on a login node.
- Images are stored under `/tmp` on the login node that built them, so they are tied to that node and can be removed at any time. Save anything worth keeping to `/ccsopen/proj/<projid>` or `/gpfs/wolf2/olcf/<projid>/proj-shared`.
- Load `olcf-container-tools` (`apptainer-enable-mpi`, `apptainer-enable-gpu`) only when running. Loading it before `apptainer build` breaks the build with a `mount /opt/cray->/opt/cray` error.

```sh
podman save -o ptychodus.tar localhost/ptychodus:<version>-rocm7.2.4
apptainer build ptychodus.sif docker-archive://ptychodus.tar
```

## AmSC Data Transfer API

The [Demo Data Transfer APIs for AmSC website](https://amsc-data-api.nersc.gov/docs) links to a [script (generate_token.py)](https://gist.github.com/tylern4/924b19e58d75046e593e0db2d87f6c5c) that gets a Globus bearer token for testing:

```sh
python scripts/genesis/generate_token.py login \
    --mapped-collections 05d2c76a-e867-4f67-aa57-76edeb0beda0 \
    --mapped-collections 9d6d994a-6d04-11e5-ba46-22000b92c6ec
```

## Ptychodus Installation

Install uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python

```sh
uv python install 3.11
```

Install Ptychodus

```sh
uv tool install ptychodus[globus,gui,ptychi]
```

A container is the alternative to installing into a conda or uv environment; see the **Containers** section for each facility above.
