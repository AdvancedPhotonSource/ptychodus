from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLayout,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CheckableGroupBox(QGroupBox):
    """A checkable group box that can collapse to its title row while unchecked.

    Qt greys out an unchecked group box's children but still reserves their space. When
    ``collapse_when_unchecked`` is set, the body is hidden instead, so a long stack of
    group boxes reduces to a list of title rows. The title and its checkbox stay visible,
    since that checkbox is the only way to bring the body back.
    """

    def __init__(
        self,
        title: str,
        *,
        collapse_when_unchecked: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self._collapse_when_unchecked = collapse_when_unchecked
        self._contents = QWidget()

        # Hiding individual fields of a QFormLayout would strand their labels, so the
        # body lives in one wrapper widget that is hidden or shown as a unit.
        self._layout = QVBoxLayout()
        self._layout.addWidget(self._contents)
        self.setLayout(self._layout)

        # A hidden widget contributes no height, but the margins around it would still
        # leave an empty strip under the title, so they are dropped while collapsed.
        self._expanded_margins = self._layout.contentsMargins()

        self.toggled.connect(self._apply_contents_visibility)
        self._apply_contents_visibility(self.isChecked())

    def set_contents_layout(self, layout: QLayout) -> None:
        """Place a layout inside the collapsible body."""
        # The outer layout keeps the style's margins, which is what reserves clearance
        # for the title, so the inner one adds none of its own.
        layout.setContentsMargins(0, 0, 0, 0)
        self._contents.setLayout(layout)

    def _apply_contents_visibility(self, checked: bool) -> None:
        visible = checked or not self._collapse_when_unchecked
        self._contents.setVisible(visible)

        if visible:
            self._layout.setContentsMargins(self._expanded_margins)
        else:
            self._layout.setContentsMargins(0, 0, 0, 0)


class GroupBoxWithPresets(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)

        self.presets_menu = QMenu()
        self._presets_button = QToolButton()
        self._presets_button.setText('Presets  ')
        self._presets_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._presets_button.setMenu(self.presets_menu)
        self._presets_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.contents = QWidget()
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.contents)

        self._frame = QFrame()
        self._frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._frame.setFrameShadow(QFrame.Shadow.Plain)
        self._frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._frame.setLayout(frame_layout)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title_label, 0, 0)
        layout.addWidget(self._presets_button, 0, 1, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._frame, 1, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)
