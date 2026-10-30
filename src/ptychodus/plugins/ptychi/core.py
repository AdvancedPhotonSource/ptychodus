from __future__ import annotations

from ptychodus.api.reconstruct import Reconstructor, ReconstructorLibrary


class PtyChiReconstructorLibrary(ReconstructorLibrary):
    """What pty-chi offers, holding nothing but the names it is built from."""

    def __init__(self) -> None:
        super().__init__('ptychi', 'ptychi', requires=('ptychi', 'torch'))

    def _build(self) -> list[Reconstructor]:
        # Deferred so that importing this module costs nothing: the algorithm
        # table reaches ``ptychi.api``, which pulls torch in for its type
        # annotations.
        from .algorithms import build_reconstructors

        return list(build_reconstructors())
