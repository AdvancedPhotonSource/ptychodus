from __future__ import annotations

import logging
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

from ...api.reconstruct import (
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)

logger = logging.getLogger(__name__)


def _ptycho_fm_available() -> bool:
    """Return True iff every package the child imports is present, without importing them.

    ``ptychi`` is in the list because the child borrows
    ``ptychi.image_proc.place_patches_fourier_shift`` to stitch predicted
    patches. Only the reconstruct path needs it, but a reconstructor that can
    train and not reconstruct is not worth advertising, so a missing pty-chi
    disables the backend here rather than failing inside a spawned child.
    """
    return all(find_spec(mod) is not None for mod in ('ptycho_fm', 'torch', 'ptychi'))


class PtychoFMReconstructorLibrary(ReconstructorLibrary):
    """What PtychoFM offers, holding nothing but the answer to whether it is installed."""

    def __init__(self) -> None:
        super().__init__('ptycho-fm')
        self._reconstructors: list[TrainableReconstructor] | None = None

    @property
    def name(self) -> str:
        return 'PtychoFM'

    def __iter__(self) -> Iterator[Reconstructor]:
        if self._reconstructors is None:
            self._reconstructors = self._build()

        return iter(self._reconstructors)

    def _build(self) -> list[TrainableReconstructor]:
        """Enumerate the reconstructors, on first use rather than at registration.

        Registration happens in every process that loads plugins, including
        ones that never reconstruct, so the cost of deciding what a backend
        offers belongs to the first caller that asks.
        """
        if not _ptycho_fm_available():
            logger.info('PtychoFM (ptycho-fm) not found.')
            return []

        try:
            ptycho_fm_version = version('ptycho-fm')
        except PackageNotFoundError:
            ptycho_fm_version = 'unknown'

        logger.info(f'PtychoFM (ptycho-fm) {ptycho_fm_version}')

        # Deferred so that importing this module costs nothing: the factory
        # reaches for h5py and numpy, and the child it configures reaches for
        # torch.
        from .reconstructor import build_reconstructor

        return [build_reconstructor(mode) for mode in ('Unsupervised', 'Supervised')]
