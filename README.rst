Ptychodus
=========

`ptychodus`_ is a ptychography analysis front-end that supports multiple reconstruction back-ends. Current reconstructor status:

* `tike`_ is working
* `ptychopy`_ is under development
* `PtychoNN`_ is under development

Installation
------------

* To install `ptychodus`, install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ then:

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

Reporting bugs
--------------

Open a bug at https://github.com/AdvancedPhotonSource/ptychodus/issues.

.. _`ptychodus`: https://github.com/AdvancedPhotonSource/ptychodus
.. _`tike`: https://github.com/tomography/tike
.. _`ptychopy`: https://github.com/AdvancedPhotonSource/ptychopy
.. _`PtychoNN`: https://github.com/mcherukara/PtychoNN

