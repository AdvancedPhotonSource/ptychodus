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
