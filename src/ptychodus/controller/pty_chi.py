from PyQt5.QtWidgets import QWidget

from ..model.pty_chi import PtyChiReconstructorLibrary
from .reconstructor import ReconstructorViewControllerFactory

__all__ = ['PtyChiViewControllerFactory']


class PtyChiViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtyChiReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backendName(self) -> str:
        return 'pty-chi'

    def createViewController(self, reconstructorName: str) -> QWidget:
        return QWidget()
