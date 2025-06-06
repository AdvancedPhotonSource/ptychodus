PvaPy Streaming Workflow
========================

* To install the `PvaPy <https://github.com/epics-base/pvaPy>`_

.. code-block:: shell

   $ conda install -n ptychodus -c apsu pvapy

* In Terminal 1:

.. code-block:: shell

   $ pvapy-hpc-consumer \
       --input-channel pvapy:image \
       --control-channel consumer:*:control \
       --status-channel consumer:*:status \
       --output-channel consumer:*:output \
       --processor-class ptychodus.PtychodusAdImageProcessor \
       --processor-args '{ "settingsFilePath": "/path/to/ptychodus.ini", "reconstructFrameId": 1000 }' \
       --report-period 10 \
       --log-level debug

* In Terminal 2:

.. code-block:: shell

   # application status
   $ pvget consumer:1:status

   # configure application
   $ pvput consumer:1:control '{"command" : "configure", "args" : "{\"nPatternsTotal\": 1000}"}'

   # get last command status
   $ pvget consumer:1:control

   # start area detector sim server
   $ pvapy-ad-sim-server -cn pvapy:image -if /path/to/fly001.npy -rt 120 -fps 1000

* At the end of the demo,

.. code-block:: shell

   # shutdown consumer process
   pvput consumer:1:control '{"command" : "stop"}'
