from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...model.scan import ScanRepositoryItemPresenter
from ...view.scan import TabularScanView, ScanEditorDialog
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class TabularScanController:

    def __init__(self, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        self._item = presenter.item
        self._view = TabularScanView.createInstance()
        self._dialog = ScanEditorDialog.createInstance(presenter.name, self._view, parent)
        self._transformController = ScanTransformController.createInstance(
            presenter.item, self._dialog.transformView)

    @classmethod
    def editParameters(cls, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._dialog.open()
