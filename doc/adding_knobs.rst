Adding UI knobs
===============

Adding to an existing parameter group
-------------------------------------

**Add entries in settings**

Add the entries in the setting group of the right ``Settings`` object. 
For example: ``ptychodus/model/ptychonn/settings.py:PtychoNNPositionPredictionSettings``.


**Add getters and setters in presenter**

Add the getters and setters that read/write the relevant entries
in the settings object. For example:
``ptychodus/model/ptychonn/core.py:PtychoNNPositionPredictionPresenter``. 


**Add UI elements in view**

Add the relevant UI elements in the view class. For example:
``ptychodus/view/ptychonn.py:PtychoNNPositionPredictionParametersView``.


**Connect UI elements to presenter setters and getters**

Connect UI elements in the view object to setters and getters functions in the presenter
in the controller class. 

Usually, setters are connected to UI events that involve a value change. For example:

::

   view.numberNeighborsCollectiveSpinbox.valueChanged.connect(
      presenter.setNumberNeighborsCollective)

while getters are used in the ``_syncModelToView`` method for updating the values
displayed in the UI elements:

::

   self._view.reconstructorImagePathLineEdit.setText(
      str(self._presenter.getReconstructorImageFilePath())
   )

See this module for more examples: ``ptychodus/controller/ptychonn/position.py:PtychoNNPositionPredictionParametersController``.
