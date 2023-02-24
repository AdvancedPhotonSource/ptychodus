Globus Compute Demonstration
============================

Perform steps 1-4 on the local and remote computers.

#. Install Anaconda

.. code-block:: shell

   $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   $ sh Miniconda3-latest-Linux-x86_64.sh

#. Make conda available in your current bash environment

.. code-block:: shell

   $ eval "$(~/miniconda3/bin/conda shell.bash hook)"

#. Create the "ptychodus" conda environment

.. code-block:: shell
   $ conda create -c conda-forge -n ptychodus ptychodus-all

#. Activate the "ptychodus" conda environment

.. code-block:: shell
   $ conda activate ptychodus

#. On the remote computer, install the funcX endpoint PyPI package into the "ptychodus" conda environment

.. code-block:: shell
   $ python -m pip install funcx-endpoint

#. Configure and start the funcX endpoint

.. code-block:: shell
   $ cp ptychodus/ptychodusFuncXPolarisConfig.py ~/.funcx/default/config.py
   $ vi ~/.funcx/default/config.py
   $ funcx-endpoint configure
   $ funcx-endpoint start

#. For data transfer, use a guest collection on Globus Connect Server 5 (GCS5) or use Globus Connect Personal endpoint. See https://docs.globus.org/how-to/guest-collection-share-and-access/ for help setting up a guest collection on GCS5.
#. On the local computer, launch the reconstruction tasks from the "Remote" view.
#. On the remote computer, watch the queue (qstat) and funcX endpoint logs

.. code-block:: shell
   $ tail -f ~/.funcx/default/endpoint.log

#. When the demo is done, stop the funcX endpoint on the remote computer

.. code-block:: shell
   $ funcx-endpoint stop
