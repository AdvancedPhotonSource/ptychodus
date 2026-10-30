"""Controllers for the diffraction panel and the open-dataset wizard.

Deliberately empty. ``diffraction/core.py`` and ``product/core.py`` depend on each
other's table models, so re-exporting DiffractionController here would make a plain
``from ..diffraction.dataset import ...`` in product/core.py pull diffraction/core.py
in mid-initialization. Import DiffractionController from
``ptychodus.controller.diffraction.core`` instead.
"""
