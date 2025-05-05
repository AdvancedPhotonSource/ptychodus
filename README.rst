Ptychodus
=========

`ptychodus`_ is a ptychography analysis application that supports multiple reconstruction libraries.


Standard Installation
---------------------

1. Install `miniforge <https://github.com/conda-forge/miniforge>`_.

2. Install ptychodus.

   * To install `ptychodus` with the GUI and all optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus-all

   * To install `ptychodus` with the GUI and no optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus

   * To install `ptychodus` without the GUI or optional packages:

     .. code-block:: shell

           $ conda create -c conda-forge -n ptychodus ptychodus-core

3. Activate the `ptychodus` conda environment to run ptychodus.

   .. code-block:: shell

       $ conda activate ptychodus
       $ ptychodus -h

       usage: ptychodus [-h] [-b {reconstruct,train}] [-f FILE_PREFIX] [-s SETTINGS_FILE] [-v] [-w OUTPUT_DIR]

       ptychodus is a ptychography analysis application

       options:
         -h, --help            show this help message and exit
         -b {reconstruct,train}, --batch {reconstruct,train}
                               run action non-interactively
         -f FILE_PREFIX, --file-prefix FILE_PREFIX
                               replace file path prefix in settings
         -s SETTINGS_FILE, --settings SETTINGS_FILE
                               use settings from file
         -v, --version         show program's version number and exit
         -w OUTPUT_DIR, --write OUTPUT_DIR
                               stage reconstruction inputs to directory

       $ ptychodus


Developer Installation
----------------------

* For a developer installation:

.. code-block:: shell

   $ git clone https://github.com/AdvancedPhotonSource/ptychodus.git
   $ conda create -c conda-forge -n ptychodus --file ptychodus/requirements-dev.txt
   $ conda activate ptychodus
   $ pip install -e ./ptychodus

* To install the `pty-chi`_ backend:

.. code-block:: shell

   $ pip install ptychi

* To install the `PtychoNN`_ backend:

.. code-block:: shell

   $ conda install -n ptychodus -c conda-forge ptychonn

* Launch `ptychodus`:

.. code-block:: shell

   $ conda activate ptychodus
   $ ptychodus


Reporting Bugs
--------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.

.. _`ptychodus`: https://github.com/AdvancedPhotonSource/ptychodus
.. _`pty-chi`: https://github.com/AdvancedPhotonSource/pty-chi
.. _`PtychoNN`: https://github.com/mcherukara/PtychoNN
.. _`PvaPy`: https://github.com/epics-base/pvaPy
