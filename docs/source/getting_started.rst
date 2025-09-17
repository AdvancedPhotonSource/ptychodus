Installation Instructions
=========================

Python Package Index (PyPI)
---------------------------

To install ptychodus with the most common optional dependencies:

.. code-block:: shell

    $ python -m pip install ptychodus[globus,gui,ptychi]


For Developers: Distributing Wheels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the directory that contains ``pyproject.toml``, create wheel in ``./dist/``

.. code-block:: shell

   $ python -m build

Upload to PyPI

.. code-block:: shell

   $ python -m twine upload --verbose dist/*


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

.. code-block:: shell

   $ xhost +local:docker
   $ docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
         --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ptychodus:latest
   $ xhost -local:docker
