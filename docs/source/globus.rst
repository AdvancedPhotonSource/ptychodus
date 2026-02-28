Globus Compute Workflow
=======================

#####

export TERM=xterm
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
time conda create -y -n ptychodus tike ptychonn gladier gladier-tools globus-compute-endpoint --file ptychodus/requirements-dev.txt
time conda create -y -c conda-forge -n ptychodus tike ptychonn gladier gladier-tools --file ptychodus/requirements-dev.txt
conda activate ptychodus
pip install -e ptychodus --no-deps
ptychodus -h

globus-compute-endpoint configure
mv ~/polaris.yaml ~/.globus_compute/default/config.yaml
cat ~/.globus_compute/default/config.yaml
globus-compute-endpoint start

#####

qstat | grep shenke

#####

https://globus-compute.readthedocs.io/en/stable/endpoints/endpoint_examples.html#polaris-alcf

mkdir ptychodus-compute-endpoint
cd ptychodus-compute-endpoint/
uv venv --python 3.13
source .venv/bin/activate
uv pip install ptychodus[globus,ptychi] globus-compute-endpoint
OPENBLAS_NUM_THREADS=1 ptychodus --version
cat ~/.globus_compute/default/config.yaml
source ~/ptychodus-compute-endpoint/.venv/bin/activate
globus-compute-endpoint start

#####

Perform steps 1-3 on the local and remote computers.

#. Install `miniforge <https://github.com/conda-forge/miniforge>`_.

#. Make conda available in your current shell environment

.. code-block:: shell

   $ eval "$(~/miniforge3/bin/conda shell.bash hook)"

#. Create and activate the ``ptychodus`` conda environment

.. code-block:: shell

   $ conda create -c conda-forge -n ptychodus ptychodus-all
   $ conda activate ptychodus

#. Install a `Globus compute endpoint <https://globus-compute.readthedocs.io/en/stable/quickstart.html#deploying-an-endpoint>`_
   into the ``ptychodus`` environment on the remote computer and configure it. An
   example configuration file for ALCF Polaris is bundled with the Ptychodus
   source distribution.

.. code-block:: shell

   $ python -m pip install globus-compute-endpoint
   $ globus-compute-endpoint configure

#. Start the Globus compute endpoint

.. code-block:: shell

   $ globus-compute-endpoint start <ENDPOINT_NAME>

#. For data transfer, use a `guest collection <https://docs.globus.org/how-to/guest-collection-share-and-access>`_
   on a Globus Connect Server or use a `Globus Connect Personal endpoint <https://www.globus.org/globus-connect-personal>`_ on your local computer.
#. On the local computer, launch the reconstruction tasks from the "Workflow" view.
#. On the remote computer, watch the queue (use qstat on Polaris) and Globus compute endpoint logs

.. code-block:: shell

   $ tail -f ~/.globus-compute/default/endpoint.log

#. When the demo is done, stop the Globus compute endpoint on the remote computer

.. code-block:: shell

   $ globus-compute-endpoint stop


**Example Globus Compute Endpoint Configuration**

Here is an example Globus compute endpoint ``config.yaml`` for ALCF Polaris:

.. literalinclude:: polaris.yaml
   :language: yaml
