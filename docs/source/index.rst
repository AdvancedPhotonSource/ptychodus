Ptychodus documentation
=======================

.. image:: ptychodus.svg
   :alt: Ptychodus Logo
   :align: center
   :width: 10em


`Ptychodus <https://github.com/AdvancedPhotonSource/ptychodus>`_
is a ptychography data analysis application that reads instrument data,
prepares the data for processing, and supports calling several reconstruction
libraries for phase retrieval. Ptychodus can be used interactively or
integrated into a data pipeline.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   readers
   api
   globus
   pvapy


Python API Example
------------------

.. code-block:: python

    from pathlib import Path
    from ptychodus.model import ModelCore

    def main() -> int:
        settings_file = Path("path/to/settings.ini")

        with ModelCore(settings_file) as model:
            input_product_api = model.workflow_api.create_product("new_product_name")
            output_product_api = input_product_api.reconstruct_local()
            output_product_api.save_product("/path/to/file.h5", file_type="HDF5")


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
