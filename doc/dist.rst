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

   $ time docker build -t python-ptychodus .

Run container

.. code-block:: shell

   $ xhost +local:docker
   $ docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host \
         --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 python-ptychodus
   $ xhost -local:docker

Check container status

.. code-block:: shell

   $ docker ps -a

Clean up images

.. code-block:: shell

   $ sudo docker system prune -a
