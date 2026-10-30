from __future__ import annotations
from collections.abc import Callable

from PyQt5.QtWidgets import QPushButton, QWidget

from .parameters import ParameterViewController


class MakeEditableViewController(ParameterViewController):
    """A button that lets a reconstruction result be conditioned, then reopens its editor.

    A result's builder deliberately ignores the editor's settings, so the dialog offers
    nothing else until it is pressed; the reopened dialog then shows the controls that
    now apply.
    """

    def __init__(
        self, make_editable: Callable[[], None], reopen: Callable[[], None], tool_tip: str
    ) -> None:
        super().__init__()
        self._make_editable = make_editable
        self._reopen = reopen
        self._widget = QPushButton('Make Editable')
        self._widget.setToolTip(tool_tip)
        self._widget.clicked.connect(self._on_clicked)

    def get_widget(self) -> QWidget:
        return self._widget

    def _on_clicked(self) -> None:
        self._make_editable()
        self._widget.window().close()
        self._reopen()
