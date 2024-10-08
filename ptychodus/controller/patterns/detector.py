from __future__ import annotations

from PyQt5.QtWidgets import QFormLayout


from ...model.patterns import Detector
from ...view.patterns import DetectorView
from ..parametric import LengthWidgetParameterViewController, SpinBoxParameterViewController


class DetectorController:
    def __init__(self, detector: Detector, view: DetectorView) -> None:
        self._widthInPixelsViewController = SpinBoxParameterViewController(detector.widthInPixels)
        self._heightInPixelsViewController = SpinBoxParameterViewController(detector.heightInPixels)
        self._pixelWidthViewController = LengthWidgetParameterViewController(
            detector.pixelWidthInMeters
        )
        self._pixelHeightViewController = LengthWidgetParameterViewController(
            detector.pixelHeightInMeters
        )
        self._bitDepthViewController = SpinBoxParameterViewController(detector.bitDepth)

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._widthInPixelsViewController.getWidget())
        layout.addRow('Detector Height [px]:', self._heightInPixelsViewController.getWidget())
        layout.addRow('Pixel Width:', self._pixelWidthViewController.getWidget())
        layout.addRow('Pixel Height:', self._pixelHeightViewController.getWidget())
        layout.addRow('Bit Depth:', self._bitDepthViewController.getWidget())
        view.setLayout(layout)
