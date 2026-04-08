Ptychodus
=========

`Ptychodus <https://github.com/AdvancedPhotonSource/ptychodus>`_ is a
ptychography data analysis application that extracts, loads, and transforms
instrument data for processing. It integrates several reconstruction libraries
for phase retrieval. Ptychodus can be used interactively or integrated into
beamline data pipelines.

Standard Installation
---------------------

To install ptychodus from PyPI with the most common optional dependencies:

.. code-block:: shell

    $ python -m pip install ptychodus[globus,gui,ptychi]

Instructions for installing in containers, uv, and from conda-forge are provided in
the ``docs`` directory.


Developer Installation
----------------------

- For a developer installation:

.. code-block:: shell

   $ git clone https://github.com/AdvancedPhotonSource/ptychodus.git
   $ cd ptychodus
   $ uv sync --extra globus --extra gui --extra ptychi

- Launch `ptychodus`:

.. code-block:: shell

   $ uv run ptychodus


Reporting Bugs
--------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.
