from __future__ import annotations

from PyQt5.QtWidgets import QFormLayout, QLabel, QLineEdit, QWidget

from ptychodus.view.widgets import CheckableGroupBox


def _make(qapp, *, collapse_when_unchecked: bool) -> CheckableGroupBox:
    del qapp  # fixture presence ensures QApplication exists
    group_box = CheckableGroupBox('Title', collapse_when_unchecked=collapse_when_unchecked)

    layout = QFormLayout()
    layout.addRow('Field:', QLineEdit())
    group_box.set_contents_layout(layout)

    return group_box


def _contents(group_box: CheckableGroupBox) -> QWidget:
    layout = group_box.layout()
    assert layout is not None
    item = layout.itemAt(0)
    assert item is not None
    widget = item.widget()
    assert widget is not None
    return widget


class TestCollapsing:
    def test_contents_hidden_while_unchecked(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=True)
        group_box.setChecked(False)
        assert _contents(group_box).isHidden()

    def test_contents_shown_while_checked(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=True)
        group_box.setChecked(True)
        assert not _contents(group_box).isHidden()

    def test_toggling_flips_contents_in_both_directions(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=True)
        contents = _contents(group_box)

        group_box.setChecked(False)
        assert contents.isHidden()

        group_box.setChecked(True)
        assert not contents.isHidden()

        group_box.setChecked(False)
        assert contents.isHidden()

    def test_unchecked_box_is_shorter_than_a_checked_one(self, qapp) -> None:
        checked = _make(qapp, collapse_when_unchecked=True)
        checked.setChecked(True)

        unchecked = _make(qapp, collapse_when_unchecked=True)
        unchecked.setChecked(False)

        assert unchecked.sizeHint().height() < checked.sizeHint().height()


class TestWithoutCollapsing:
    def test_contents_stay_visible_when_unchecked(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=False)
        group_box.setChecked(False)
        assert not _contents(group_box).isHidden()

    def test_contents_stay_visible_when_checked(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=False)
        group_box.setChecked(True)
        assert not _contents(group_box).isHidden()


class TestContentsLayout:
    def test_layout_goes_in_the_wrapper_not_the_group_box(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=True)
        contents = _contents(group_box)
        assert contents.layout() is not None
        assert group_box.layout() is not contents.layout()

    def test_wrapper_adds_no_margins_of_its_own(self, qapp) -> None:
        group_box = _make(qapp, collapse_when_unchecked=True)
        layout = _contents(group_box).layout()
        assert layout is not None
        assert layout.getContentsMargins() == (0, 0, 0, 0)

    def test_widgets_are_reparented_into_the_wrapper(self, qapp) -> None:
        del qapp  # fixture presence ensures QApplication exists
        group_box = CheckableGroupBox('Title', collapse_when_unchecked=True)
        label = QLabel('Field')

        layout = QFormLayout()
        layout.addRow(label)
        group_box.set_contents_layout(layout)

        assert label.parentWidget() is _contents(group_box)
