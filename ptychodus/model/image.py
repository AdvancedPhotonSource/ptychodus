from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import matplotlib
import numpy
import numpy.typing

from ..api.plugins import PluginChooser, PluginEntry


@dataclass(frozen=True)
class ImageExtent:
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    def __add__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width + other.width
            h = self.height + other.height
            return ImageExtent(width=w, height=h)

    def __sub__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width - other.width
            h = self.height - other.height
            return ImageExtent(width=w, height=h)

    def __mul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width * other
            h = self.height * other
            return ImageExtent(width=w, height=h)

    def __rmul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = other * self.width
            h = other * self.height
            return ImageExtent(width=w, height=h)

    def __floordiv__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width // other
            h = self.height // other
            return ImageExtent(width=w, height=h)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.width}, {self.height})'


class ImagePresenter(Observable):
    @staticmethod
    def _createColorMapEntry(cmap: str) -> PluginEntry[matplotlib.colors.Colormap]:
        return PluginEntry[matplotlib.colors.Colormap](simpleName=cmap, displayName=cmap, strategy=matplotlib.cm.get_cmap(cmap))

    def __init__(self,
            scalarTransformationChooser: PluginChooser[ScalarTransformation],
            complexToRealStrategyChooser: PluginChooser[ComplexToRealStrategy]) -> None:
        super().__init__()
        self._colorMapChooser = colorMapChooser
        self._scalarTransformationChooser = scalarTransformationChooser
        self._complexToRealStrategyChooser = complexToRealStrategyChooser
        self._array: Optional[numpy.typing.NDArray] = None
        self._vmin = 0.
        self._vmax = 1.

        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        self._cyclicColorMapList = ['twilight', 'twilight_shifted', 'hsv']
        self._cyclicColorMap = 'hsv'
        self._acyclicColorMapList = [cm for cm in matplotlib.pyplot.colormaps()
                if cm not in self._cyclicColorMaps]
        self._acyclicColorMap = 'viridis'

# FIXME acyclic vs cyclic
    def getColorMapList(self) -> list[str]:
        return self._colorMapChooser.getDisplayNameList()

    def getColorMap(self) -> str:
        return self_colorMapChooser.getCurrentDisplayName()

    def setColorMap(self, name: str) -> None:
        self._colorMapChooser.setFromDisplayName(name)
# FIXME

    def getScalarTransformationList(self) -> list[str]:
        return self._scalarTransformationChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self_scalarTransformationChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._scalarTransformationChooser.setFromDisplayName(name)

    def getComplexToRealStrategyList(self) -> list[str]:
        return self._complexToRealStrategyChooser.getDisplayNameList()

    def getComplexToRealStrategy(self) -> str:
        return self_complexToRealStrategyChooser.getCurrentDisplayName()

    def setComplexToRealStrategy(self, name: str) -> None:
        self._complexToRealStrategyChooser.setFromDisplayName(name)

    # FIXME auto vmin/vmax

    def getVMin(self) -> float:
        return min(self._vmin, self._vmax)

    def setVMin(self, vmin: float) -> None:
        self._vmin = vmin

    def getVMax(self) -> float:
        return max(self._vmin, self._vmax)

    def setVMax(self, vmax: float) -> None:
        self._vmax = vmax

    def setArray(self, array: numpy.typing.NDArray) -> None:
        self._array = array

    def getImage(self) -> numpy.typing.NDArray:
# FIXME BEGIN
        image_data = self._image_data

        if image_data is None:
            return

        currentComplexComponentStrategy = self._view.imageRibbon.complexComponentComboBox.currentData(
        )

        if numpy.iscomplexobj(image_data):
            self._view.imageRibbon.complexComponentComboBox.setVisible(True)
            self._view.imageRibbon.colorMapComboBox.setModel(
                self._cyclicColorMapModel if currentComplexComponentStrategy.isCyclic else self.
                _acyclicColorMapModel)
            image_data = currentComplexComponentStrategy.complexToRealFunction(image_data)
        else:
            self._view.imageRibbon.complexComponentComboBox.setVisible(False)
            self._view.imageRibbon.colorMapComboBox.setModel(self._acyclicColorMapModel)

        currentScalarTransformStrategy = self._view.imageRibbon.scalarTransformComboBox.currentData(
        )
        image_data = currentScalarTransformStrategy.transformFunction(image_data)

        if self._view.imageRibbon.vminAutoCheckBox.isChecked():
            self._vmin = image_data.min()
            self._updateVminLineEditText()

        if self._view.imageRibbon.vmaxAutoCheckBox.isChecked():
            self._vmax = image_data.max()
            self._updateVmaxLineEditText()

        cnorm = matplotlib.colors.Normalize(vmin=self._vmin, vmax=self._vmax, clip=False)
        cmap = matplotlib.cm.get_cmap(self._view.imageRibbon.colorMapComboBox.currentText())
        scalarMappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

        color_image = scalarMappable.to_rgba(image_data)
        return color_image
