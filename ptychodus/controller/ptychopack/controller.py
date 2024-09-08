from __future__ import annotations

from ...model.ptychopack import PtychoPackReconstructorLibrary
from ...view.ptychopack import PtychoPackView


class PtychoPackController:

    def __init__(self, model: PtychoPackReconstructorLibrary, view: PtychoPackView) -> None:
        self._model = model
        self._view = view
