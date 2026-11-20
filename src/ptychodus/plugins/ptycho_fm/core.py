from __future__ import annotations

from ...api.reconstruct import (
    ReconstructorLibrary,
    TrainableReconstructor,
)


class PtychoFMReconstructorLibrary(ReconstructorLibrary):
    """What PtychoFM offers, holding nothing but the names it is built from.

    ``ptychi`` is required alongside PtychoFM's own packages because inference
    borrows ``ptychi.image_proc.ObjectPatchSampler`` to stitch
    predicted patches. Only the reconstruct path needs it, but a reconstructor
    that can train and not reconstruct is not worth advertising, so a missing
    pty-chi disables the library here rather than failing mid-reconstruction.
    """

    def __init__(self) -> None:
        super().__init__('ptycho-fm', 'ptycho-fm', requires=('ptycho_fm', 'torch', 'ptychi'))

    def _build(self) -> list[TrainableReconstructor]:
        # Deferred so that importing this module costs nothing: the reconstructor
        # module reaches for numpy, and its reconstruct and train for torch.
        from .reconstructor import PtychoFMReconstructor

        return [PtychoFMReconstructor(mode) for mode in ('Unsupervised', 'Supervised')]
