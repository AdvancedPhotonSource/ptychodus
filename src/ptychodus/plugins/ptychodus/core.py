from __future__ import annotations

from ...api.reconstruct import Reconstructor, ReconstructorLibrary


class PtychodusReconstructorLibrary(ReconstructorLibrary):
    """The reconstructors ptychodus implements itself: Kickstart, a coarse probe and object
    from the diffraction moments to start another backend from.

    Needs nothing beyond ptychodus, so it is always available.
    """

    def __init__(self) -> None:
        super().__init__('ptychodus.plugins.ptychodus', 'ptychodus')

    def _build(self) -> list[Reconstructor]:
        from .reconstructor import build_reconstructor

        return [build_reconstructor()]
