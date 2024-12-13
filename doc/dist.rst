Distribution Instructions
=========================

Python Package Index (PyPI)
---------------------------

From the ptychodus directory

.. code-block:: shell

   $ python -m build .

Docker Image
------------

Build image

.. code-block:: shell

   $ time docker build -t python-ptychodus .

Run container

.. code-block:: shell

   $ xhost +local:docker
   # docker run -it --rm  -e "DISPLAY=$DISPLAY" -v "$HOME/.Xauthority:/root/.Xauthority:ro" --network host python-ptychodus
   $ xhost -local:docker

Check container status

.. code-block:: shell

   $ docker ps -a

Clean up images

.. code-block:: shell

   $ sudo docker system prune -a
