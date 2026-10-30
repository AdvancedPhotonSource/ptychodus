from __future__ import annotations

from ...api.reconstruct import (
    ReconstructorLibrary,
    TrainableReconstructor,
)


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):
    """What PtychoPINN offers, holding nothing but the names it is built from."""

    def __init__(self) -> None:
        super().__init__('ptychopinn', 'ptychopinn', requires=('ptycho_torch', 'lightning'))

    def _build(self) -> list[TrainableReconstructor]:
        # Imported lazily so this module's own import cost stays small.
        from .reconstructor import build_reconstructor

        return [build_reconstructor(mode) for mode in ('Unsupervised', 'Supervised')]
