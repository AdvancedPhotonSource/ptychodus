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


Podman
------

Build Podman image

.. code-block:: shell

    $ podman build -t ptychodus:latest .

Run container

.. note::

   GPU access requires CDI (Container Device Interface) to be configured on the host.
   Run ``sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`` once before
   using ``--device nvidia.com/gpu=all``.

.. code-block:: shell

   $ xhost +local:podman
   $ podman run -it --rm --env DISPLAY --security-opt label=type:container_runtime_t --network host \
       --device nvidia.com/gpu=all ptychodus:latest
   $ xhost -local:podman


Docker
------

Build Docker image

.. code-block:: shell

   $ docker build -t ptychodus:latest .


Run container

.. note::

   GPU access requires `nvidia-container-toolkit
   <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
   to be installed on the host before using ``--gpus all``.

.. code-block:: shell

   $ xhost +local:docker
   $ docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
         --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ptychodus:latest
   $ xhost -local:docker


VS Code Dev Container
---------------------

The repository includes a `Dev Container <https://containers.dev/>`_ configuration
that builds from the project ``Dockerfile``, giving a ready-to-use environment
inside VS Code.

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
