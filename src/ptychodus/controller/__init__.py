"""Controllers mediating between the Qt views and the model.

Deliberately empty: re-exporting ControllerCore here would make every import under
``ptychodus.controller`` pull in PyQt5, which would put the Qt-free array rendering
in ``controller/rendering/`` out of reach of a bare install. Import the composition
root from ``ptychodus.controller.core`` instead.
"""
