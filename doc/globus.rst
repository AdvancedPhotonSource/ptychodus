Globus Compute Demonstration
============================

Perform steps 1-4 on the local and remote computers.

1. Install Anaconda

.. code-block:: shell

   $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   $ sh Miniconda3-latest-Linux-x86_64.sh

2. Make conda available in your current bash environment

.. code-block:: shell

   $ eval "$(~/miniconda3/bin/conda shell.bash hook)"

3. Create the "ptychodus" conda environment

.. code-block:: shell
   $ conda create -c conda-forge -n ptychodus ptychodus-all

4. Activate the "ptychodus" conda environment

.. code-block:: shell
   $ conda activate ptychodus

5. On the remote computer, install the funcX endpoint PyPI package into the "ptychodus" conda environment

.. code-block:: shell
   $ python -m pip install funcx-endpoint

6. Configure and start the funcX endpoint

.. code-block:: shell
   $ cp ptychodus/ptychodusFuncXPolarisConfig.py ~/.funcx/default/config.py
   $ vi ~/.funcx/default/config.py
   $ funcx-endpoint configure
   $ funcx-endpoint start

7. For data transfer, use a guest collection on Globus Connect Server 5 (GCS5) or use Globus Connect Personal endpoint. See https://docs.globus.org/how-to/guest-collection-share-and-access/ for help setting up a guest collection on GCS5.
8. On the local computer, launch the reconstruction tasks from the "Remote" view.
9. On the remote computer, watch the queue (qstat) and funcX endpoint logs

.. code-block:: shell
   $ tail -f ~/.funcx/default/endpoint.log

10. When the demo is done, stop the funcX endpoint on the remote computer

.. code-block:: shell
   $ funcx-endpoint stop
