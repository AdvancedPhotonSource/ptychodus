Ptychodus
=========

`ptychodus`_ is a ptychography analysis application that supports multiple reconstruction libraries. Current reconstructor status:

* `tike`_ is working
* `PtychoNN`_ is working
* `ptychopy`_ is under development

Standard Installation
---------------------

1. Install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

2. Install ptychodus.

   * To install `ptychodus` with the GUI and no optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus

   * To install `ptychodus` with the GUI and all optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus-all

   * To install `ptychodus` without the GUI or optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus-core

3. Activate the `ptychodus` conda environment to run ptychodus.

   .. code-block:: shell
       $ conda activate ptychodus
       $ ptychodus -h
       usage: ptychodus [-h] [-b RESULTS_FILE] [-f PREFIX] [-p PORT] [-r RESTART_FILE] [-s SETTINGS_FILE] [-v]

       ptychodus is a ptychography analysis application

       options:
         -h, --help            show this help message and exit
         -b RESULTS_FILE, --batch RESULTS_FILE
                               run reconstruction non-interactively
         -f PREFIX, --file-prefix PREFIX
                               replace file path prefix
         -p PORT, --port PORT  remote process communication port number
         -r RESTART_FILE, --restart RESTART_FILE
                               use restart data from file
         -s SETTINGS_FILE, --settings SETTINGS_FILE
                               use settings from file
         -v, --version         show program's version number and exit
       $ ptychodus


Developer Installation
----------------------

* For a developer installation:

.. code-block:: shell

   $ git clone https://github.com/AdvancedPhotonSource/ptychodus.git
   $ conda create -c conda-forge -n ptychodus --file ptychodus/requirements-dev.txt
   $ conda activate ptychodus
   $ pip install -e ./ptychodus

* To install the `tike` backend:

.. code-block:: shell

   $ conda install -n ptychodus -c conda-forge tike

* To install the `PtychoNN` backend:

.. code-block:: shell

   $ conda install -n ptychodus -c conda-forge ptychonn pytorch-gpu

* To launch the `ptychodus` GUI (with the "ptychodus" conda environment activated):

.. code-block:: shell

   $ ptychodus

Tips
----

* This project is experimenting with `type hints <https://docs.python.org/3/library/typing.html>`_ which can be checked using `mypy <http://mypy-lang.org>`_.

.. code-block:: shell

  $ mypy ptychodus

* Stubs to support PyQt5 type hinting can be installed within the conda environment.

.. code-block:: shell

   $ pip install PyQt5-stubs

Reporting bugs
-------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.

.. _`ptychodus`: https://github.com/AdvancedPhotonSource/ptychodus
.. _`tike`: https://github.com/tomography/tike
.. _`ptychopy`: https://github.com/AdvancedPhotonSource/ptychopy
.. _`PtychoNN`: https://github.com/mcherukara/PtychoNN
.. _`PvaPy`: https://github.com/epics-base/pvaPy
