from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QTreeView, QVBoxLayout, QWidget


class DataParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Diffraction Dataset', parent)
        self.treeView = QTreeView()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> DataParametersView:
        view = cls(parent)

        # FIXME data size/rate stats live view; assembled size on memory/disk
        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        view.setLayout(layout)

        return view
