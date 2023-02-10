RPC Demonstration
=================

* Generate a batch mode reconstruction result file

.. code-block:: shell

   $ ptychodus -b /path/to/results.npz -s /path/to/settings.ini

* Launch the GUI in one terminal and navigate to the "Monitor" view

.. code-block:: shell

   $ ptychodus -p 9999

* Send a RPC message (JSON format) to instruct the GUI to display the reconstruction result

.. code-block:: shell

   $ ptychodus-rpc -p 9999 -m '{"procedure": "LoadResults", "filePath": "/path/to/results.npz"}'
