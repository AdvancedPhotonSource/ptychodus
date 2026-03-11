Globus Compute Workflow
=======================

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
