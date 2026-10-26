from __future__ import annotations

from ...api.reconstruct import Reconstructor, ReconstructorLibrary


class CDToolsReconstructorLibrary(ReconstructorLibrary):
    """What cdtools offers, holding nothing but the names it is built from.

    Only FancyPtycho, cdtools' far-field model with incoherent probe modes, position
    refinement and a detector background and flat field. Its multislice model places
    every slice after the first by rounding the window corner where the first slice
    floors it, so about half the positions read later slices one pixel off; it stays
    out until that is fixed upstream.

    Installs with ``uv sync --extra cdtools``, which takes PyPI's default PyTorch. To
    choose the PyTorch build, add the ``ptychi`` extra matching the driver alongside it,
    whose index pinning then applies to both.
    """

    def __init__(self) -> None:
        super().__init__('cdtools', 'cdtools-py', requires=('cdtools', 'torch'))

    def _build(self) -> list[Reconstructor]:
        # Deferred so that importing this module costs nothing: the factory reaches for
        # numpy, and the child it configures reaches for torch.
        from .reconstructor import build_reconstructor

        return [build_reconstructor()]
