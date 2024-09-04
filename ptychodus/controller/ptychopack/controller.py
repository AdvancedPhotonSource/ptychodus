from __future__ import annotations

from ...model.ptychopack import PtychoPackReconstructorLibrary
from ...view.ptychopack import PtychoPackParametersView


class PtychoPackParametersController:

    def __init__(self, model: PtychoPackReconstructorLibrary,
                 view: PtychoPackParametersView) -> None:
        self._model = model
        self._view = view
