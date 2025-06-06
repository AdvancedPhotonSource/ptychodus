Ptychodus
=========

`Ptychodus <https://github.com/AdvancedPhotonSource/ptychodus>`_
is a ptychography data analysis application that reads instrument data,
prepares the data for processing, and supports calling several reconstruction
libraries for phase retrieval. Ptychodus can be used interactively or
integrated into a data pipeline.


Standard Installation
---------------------

To install ptychodus from PyPI with the most common optional dependencies:

.. code-block:: shell

    $ python -m pip install ptychodus[globus,gui,ptychi]

Instructions for installing in containers and from conda-forge are provided in
the ``docs`` directory.


Developer Installation
----------------------

- For a developer installation:

.. code-block:: shell

   $ git clone https://github.com/AdvancedPhotonSource/ptychodus.git
   $ conda create -n ptychodus --file ptychodus/requirements-dev.txt
   $ conda activate ptychodus
   $ pip install -e ./ptychodus

- To install `pty-chi <https://github.com/AdvancedPhotonSource/pty-chi>`_

.. code-block:: shell

   $ pip install ptychi

- To install `PtychoNN <https://github.com/mcherukara/PtychoNN>`_

.. code-block:: shell

   $ conda install -n ptychodus ptychonn

- Launch `ptychodus`:

.. code-block:: shell

   $ conda activate ptychodus
   $ ptychodus


Reporting Bugs
--------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.
