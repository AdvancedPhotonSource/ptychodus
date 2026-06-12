Installation Instructions
=========================

Python Package Index (PyPI)
---------------------------

To install ptychodus with the most common optional dependencies:

.. code-block:: shell

    $ python -m pip install ptychodus[globus,gui,ptychi]


uv
--

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package and project manager.

#. Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_.

#. Install ptychodus with the most common optional dependencies:

   .. code-block:: shell

       $ uv tool install ptychodus[globus,gui,ptychi]

#. Launch ptychodus:

   .. code-block:: shell

       $ ptychodus

#. To upgrade ptychodus, use uv tool upgrade:

   .. code-block:: shell

       $ uv tool upgrade ptychodus[globus,gui,ptychi]


Conda-Forge
-----------

#. Install `miniforge <https://github.com/conda-forge/miniforge>`_.

#. Create the ``ptychodus`` environment

   * To install ``ptychodus`` with the GUI and all optional packages:

     .. code-block:: shell

           $ conda create -n ptychodus ptychodus-all

   * To install ``ptychodus`` with the GUI and no optional packages:

     .. code-block:: shell

           $ conda create -n ptychodus ptychodus

   * To install ``ptychodus`` without the GUI or optional packages:

     .. code-block:: shell

           $ conda create -n ptychodus ptychodus-core

#. Activate the ``ptychodus`` environment

   .. code-block:: shell

       $ conda activate ptychodus
       $ ptychodus


Container image variants
------------------------

The repository ships one Dockerfile per accelerator family. Pick the variant
that matches your hardware and select an explicit file with ``-f``:

================================  ==========================================================================================
Dockerfile                        Use it for
================================  ==========================================================================================
``Dockerfile.cpu``                CPU-only hosts (no GPU; ptychi runs on CPU torch)
``Dockerfile.cuda``               NVIDIA GPUs (e.g. ALCF Polaris, NERSC Perlmutter); CUDA minor version is a build ARG
``Dockerfile.rocm``               AMD GPUs (e.g. OLCF Frontier); ROCm is a build ARG
``Dockerfile.xpu``                Intel XPU (e.g. ALCF Aurora); base tag is a build ARG
================================  ==========================================================================================

The GPU files default to recent versions and expose ``--build-arg`` knobs to
switch:

* ``Dockerfile.cuda``: ``CUDA_VERSION`` (default ``13.0``), ``PYTORCH_VERSION``,
  ``CUDNN_VERSION``. The base image is
  ``pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel``;
  override any args if a given combination isn't published upstream.
* ``Dockerfile.rocm``: ``ROCM_VERSION`` (default ``7.2.4``), ``UBUNTU_VERSION``,
  ``PYTHON_VERSION``, ``PYTORCH_VERSION``. Base image is
  ``rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py${PYTHON_VERSION}_pytorch_release_${PYTORCH_VERSION}``.
* ``Dockerfile.xpu``: ``BASE_TAG`` (default ``latest``). Base image is
  ``intel/intel-optimized-pytorch:${BASE_TAG}``; pin to a dated tag for
  reproducibility.

Podman
------

Build Podman image

.. code-block:: shell

    $ podman build -f Dockerfile.cpu                                -t ptychodus:cpu       .
    $ podman build -f Dockerfile.cuda                               -t ptychodus:cuda13.0  .
    $ podman build -f Dockerfile.cuda --build-arg CUDA_VERSION=12.6 -t ptychodus:cuda12.6  .
    $ podman build -f Dockerfile.cuda --build-arg CUDA_VERSION=13.2 -t ptychodus:cuda13.2  .
    $ podman build -f Dockerfile.rocm                               -t ptychodus:rocm      .
    $ podman build -f Dockerfile.xpu                                -t ptychodus:xpu       .

Run container

.. note::

   GPU access requires CDI (Container Device Interface) to be configured on the host.
   Run ``sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`` once before
   using ``--device nvidia.com/gpu=all``.

.. code-block:: shell

   $ xhost +local:podman
   $ podman run -it --rm --env DISPLAY --security-opt label=type:container_runtime_t --network host \
       --device nvidia.com/gpu=all ptychodus:cuda13.0
   $ xhost -local:podman


Docker
------

Build Docker image

.. code-block:: shell

   $ docker build -f Dockerfile.cuda -t ptychodus:cuda13.0 .

(Substitute any variant file and tag as in the Podman section above.)


Run container

.. note::

   GPU access requires `nvidia-container-toolkit
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
   to be installed on the host before using ``--gpus all``.

.. code-block:: shell

   $ xhost +local:docker
   $ docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
         --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ptychodus:cuda13.0
   $ xhost -local:docker


Apptainer / Singularity
-----------------------

The images above are OCI-compliant and can be converted to SIF for HPC sites
(NERSC, OLCF, ALCF) that prefer Apptainer. After building an OCI image
locally, convert it:

.. code-block:: shell

   $ apptainer build ptychodus-cuda13.0.sif docker-daemon://localhost/ptychodus:cuda13.0

If the image lives in a registry, pull directly with ``docker://`` instead.


Wrapper Script (Beamline Workstations)
--------------------------------------

The repository ships ``scripts/podman/ptychodus``, a small bash wrapper that
makes the containerized application behave like a native ``ptychodus``
command. It auto-selects the image variant for the host's GPU, forwards X11
to the user's desktop, bind-mounts ``$HOME``, ``$PWD``, and the beamline data
roots (``/local``, ``/gdata``) into the container, and runs rootless so that
files written to those mounts are owned by the invoking user. All command
line arguments are passed through to ``ptychodus`` unchanged.

Prerequisites
^^^^^^^^^^^^^

The wrapper assumes facility IT has already provisioned the host:

* Rootless podman is installed and works for the beamline account
  (``podman info`` succeeds without ``sudo``). The wrapper itself does
  **not** install podman.
* For NVIDIA hosts, ``/etc/cdi/nvidia.yaml`` exists (generated by
  ``nvidia-ctk cdi generate`` — a one-time root step; see the note above).
* For AMD hosts, the beamline account is a member of the ``video`` and
  ``render`` groups.
* The image variant for the host's GPU has been built into the user's
  rootless podman storage using the build commands in the **Podman**
  section above. No root is required for ``podman build``.

Install (userspace, no sudo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run from the repository root as the beamline account:

.. code-block:: shell

   $ install -Dm 0755 scripts/podman/ptychodus ~/.local/bin/ptychodus
   $ which ptychodus            # should print ~/.local/bin/ptychodus

If ``which`` resolves to a conda environment's launcher instead (for example
``~/miniconda3/envs/ptychodus/bin/ptychodus``), prepend ``~/.local/bin`` to
``PATH`` in the account's shell rc:

.. code-block:: shell

   $ export PATH="$HOME/.local/bin:$PATH"

For a site-wide install, copy the script to any directory the shared
beamline account can write to (for example ``/opt/beamline/bin`` if
pre-provisioned by IT) and have each user's ``PATH`` include it. The
wrapper does not care about its install location.

Sanity check:

.. code-block:: shell

   $ ptychodus --version

stderr will show a one-line notice such as
``ptychodus: using image ptychodus:cuda13.0 (detected: nvidia)``; stdout
prints the version reported by the in-container ``ptychodus``.

Usage
^^^^^

.. code-block:: shell

   $ ptychodus                                          # GUI, auto-detect GPU
   $ ptychodus -s settings.ini                          # GUI with settings
   $ ptychodus -b reconstruct -i ./input -o ./output    # headless batch
   $ PTYCHODUS_IMAGE=ptychodus:cpu ptychodus            # force CPU image
   $ PTYCHODUS_QUIET=1 ptychodus -v                     # silent, version only

File paths passed via ``-i``, ``-o``, ``-s``, or as positional arguments
resolve naturally as long as they live under ``$HOME``, the current
directory, or one of the auto-mounted beamline roots (``/local``,
``/gdata``).

Environment overrides
^^^^^^^^^^^^^^^^^^^^^

================================  ===============================================================
Variable                          Effect
================================  ===============================================================
``PTYCHODUS_IMAGE``               Skip GPU auto-detection; use this image tag verbatim.
``PTYCHODUS_QUIET``               Suppress the stderr "using image" notice.
================================  ===============================================================

The image tag map and the list of beamline mounts live near the top of
``scripts/podman/ptychodus`` and can be edited in place to retag or add
sites (for example ``/data``, ``/nsls2``).

Troubleshooting
^^^^^^^^^^^^^^^

* **"image … not found in rootless podman storage"** — run the
  ``podman build`` command printed in the error.
* **"cannot open display"** — confirm ``$DISPLAY`` is set in the shell, then
  run ``xhost +local:`` once per X session. Wayland desktops work via
  XWayland as long as ``$DISPLAY`` is set (the default on GNOME and KDE).
* **"podman: command not found"** — ask facility IT to install rootless
  podman.
* **Wrong GPU family detected** — override with
  ``PTYCHODUS_IMAGE=ptychodus:cpu`` (or any other tag).
* **Files written by the container are owned by root** — rootless
  ``--userns=keep-id`` is not in effect; check ``podman info`` for
  ``rootless: true``.
* **"permission denied" on a path argument** — the path is outside the
  bind-mounted set. Run from under ``$HOME`` or one of the beamline roots,
  or add the path to ``BEAMLINE_MOUNTS`` at the top of the wrapper.


VS Code Dev Container
---------------------

The repository includes a `Dev Container <https://containers.dev/>`_
configuration at ``.devcontainer/devcontainer.json`` that builds from
``Dockerfile.cpu`` — the safe default for contributors without a local GPU.
GPU users can edit the ``dockerfile:`` field to point at ``Dockerfile.cuda``
(or ``Dockerfile.rocm``) before reopening in the container.

#. Install the `Dev Containers
   <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_
   extension for VS Code.

#. Open the repository folder in VS Code, then choose
   **Dev Containers: Reopen in Container** from the Command Palette
   (:kbd:`Ctrl+Shift+P`).

VS Code will build the image and reopen the workspace inside the container.


For Maintainers: Publishing Releases
-------------------------------------

Via pip / build + twine
^^^^^^^^^^^^^^^^^^^^^^^^

From the directory that contains ``pyproject.toml``, create a wheel in ``./dist/``:

.. code-block:: shell

   $ python -m build

Upload to PyPI:

.. code-block:: shell

   $ python -m twine upload --verbose dist/*


Via uv
^^^^^^

From the directory that contains ``pyproject.toml``, create a wheel in ``./dist/``:

.. code-block:: shell

   $ uv build --no-sources

Upload to PyPI:

.. code-block:: shell

   $ uv publish
