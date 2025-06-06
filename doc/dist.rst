Distribution Instructions
=========================

Python Package Index (PyPI)
---------------------------

From the directory that contains pyproject.toml, create wheel in ./dist/

.. code-block:: shell

   $ python -m build

Upload to PyPI

.. code-block:: shell

   $ python3 -m twine upload --verbose dist/*


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
