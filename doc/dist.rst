Distribution Instructions
=========================

Python Package Index (PyPI)
---------------------------

From the ptychodus directory, create wheel in ./dist/

.. code-block:: shell

   $ python -m build .

Upload to PyPI

.. code-block:: shell

   $ twine upload dist/*

Docker
------

Build Docker image

.. code-block:: shell

   $ podman build -t ptychodus:latest .


Run container

.. code-block:: shell

   $ xhost +local:podman
   $ podman run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
         --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 python-ptychodus
   $ xhost -local:podman
