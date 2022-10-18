Ptychodus
=========

`ptychodus`_ is a ptychography analysis application that supports multiple reconstruction libraries. Current reconstructor status:

* `tike`_ is working
* `ptychopy`_ is under development
* `PtychoNN`_ is under development

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
       usage: ptychodus [-h] [-b] [-d] [-f PREFIX] [-p PORT] [-s SETTINGS] [-v]

       ptychodus is a ptychography analysis application

       optional arguments:
         -h, --help            show this help message and exit
         -b, --batch           run reconstruction non-interactively
         -d, --dev             run in developer mode
         -f PREFIX, --file-prefix PREFIX
                               replace file path prefix
         -p PORT, --port PORT  remote process communication port number
         -s SETTINGS, --settings SETTINGS
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

   $ conda install -n ptychodus -c conda-forge pytorch

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

Basic RPC Demonstration
-----------------------

* Generate a batch mode reconstruction result file (named `results.npz` in this example)

.. code-block:: shell

   $ ptychodus -b -s /path/to/settings.ini

* Launch the GUI in one terminal and navigate to the "Monitor" view

.. code-block:: shell

   $ ptychodus -p 9999

* Send a RPC message (JSON format) to instruct the GUI to display the reconstruction result

.. code-block:: shell

   $ ptychodus-rpc -p 9999 -m '{"procedure": "LoadResults", "filePath": "/path/to/results.npz"}'


Streaming Demonstration
-----------------------

Terminal 1

pvapy-hpc-consumer \
    --input-channel pvapy:image \
    --control-channel consumer:*:control \
    --status-channel consumer:*:status \
    --output-channel consumer:*:output \
    --processor-class ptychodus.PtychodusAdImageProcessor \
    --processor-args '{ "settingsFilePath": "/home/beams/SHENKE/Ptychography/ptychodus/ptychodus.ini", "reconstructFrameId": 10300 }' \
    --report-period 10 \
    --log-level debug

Terminal 2

# application status
pvget consumer:1:status

# configure application
pvput consumer:1:control '{"command" : "configure", "args" : "{\"nPatternsTotal\":10400}"}'

# get last command status
pvget consumer:1:control

# $ pvapy-ad-sim-server -cn pvapy:image -nx 128 -ny 128 -dt uint8 -rt 60 -fps 10
pvapy-ad-sim-server -cn pvapy:image -if /home/beams/SHENKE/Ptychography/ptychodus/fly001.npy -rt 120 -fps 1000

# shutdown consumer process
pvput consumer:1:control '{"command" : "stop"}'


Reporting bugs
--------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.

.. _`ptychodus`: https://github.com/AdvancedPhotonSource/ptychodus
.. _`tike`: https://github.com/tomography/tike
.. _`ptychopy`: https://github.com/AdvancedPhotonSource/ptychopy
.. _`PtychoNN`: https://github.com/mcherukara/PtychoNN
