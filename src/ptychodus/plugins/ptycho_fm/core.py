from __future__ import annotations

from ...api.reconstruct import (
    ReconstructorLibrary,
    TrainableReconstructor,
)


class PtychoFMReconstructorLibrary(ReconstructorLibrary):
    """What PtychoFM offers, holding nothing but the names it is built from.

    ``ptychi`` is required alongside PtychoFM's own packages because the child
    borrows ``ptychi.image_proc.place_patches_fourier_shift`` to stitch
    predicted patches. Only the reconstruct path needs it, but a reconstructor
    that can train and not reconstruct is not worth advertising, so a missing
    pty-chi disables the backend here rather than failing inside a spawned
    child.
    """

    def __init__(self) -> None:
        super().__init__('ptycho-fm', 'ptycho-fm', requires=('ptycho_fm', 'torch', 'ptychi'))

    @property
    def name(self) -> str:
        return 'PtychoFM'

    def _build(self) -> list[TrainableReconstructor]:
        # Deferred so that importing this module costs nothing: the factory
        # reaches for h5py and numpy, and the child it configures reaches for
        # torch.
        from .reconstructor import build_reconstructor

        return [build_reconstructor(mode) for mode in ('Unsupervised', 'Supervised')]
