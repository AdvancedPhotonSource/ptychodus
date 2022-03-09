from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

import numpy

from .geometry import Interval
from .observer import Observable, Observer
from .settings import SettingsGroup, SettingsRegistry


class ColorMapCategory(Enum):
    PERCEPTUALLY_UNIFORM_SEQUENTIAL = 'Perceptually Uniform Sequential'
    SEQUENTIAL1 = 'Sequential (1)'
    SEQUENTIAL2 = 'Sequential (2)'
    DIVERGING = 'Diverging'
    CYCLIC = 'Cyclic'
    QUALITATIVE = 'Qualitative'
    MISCELLANEOUS = 'Miscellaneous'


class ColorMapListFactory:
    def __init__(self) -> None:
        self._cmap_dict = dict()
        # Source: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        self._cmap_dict[ColorMapCategory.PERCEPTUALLY_UNIFORM_SEQUENTIAL] = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ]
        self._cmap_dict[ColorMapCategory.SEQUENTIAL1] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
            'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ]
        self._cmap_dict[ColorMapCategory.SEQUENTIAL2] = [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer',
            'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
        ]
        self._cmap_dict[ColorMapCategory.DIVERGING] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
            'coolwarm', 'bwr', 'seismic'
        ]
        self._cmap_dict[ColorMapCategory.CYCLIC] = ['twilight', 'twilight_shifted', 'hsv']
        self._cmap_dict[ColorMapCategory.QUALITATIVE] = [
            'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10',
            'tab20', 'tab20b', 'tab20c'
        ]
        self._cmap_dict[ColorMapCategory.MISCELLANEOUS] = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2',
            'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo',
            'nipy_spectral', 'gist_ncar'
        ]

    def createCyclicColorMapList(self) -> list[str]:
        return self._cmap_dict[ColorMapCategory.CYCLIC]

    def createAcyclicColorMapList(self) -> list[str]:
        acyclic_cmap_list = list()

        for cat, cmap_list in self._cmap_dict.items():
            if cat != ColorMapCategory.CYCLIC:
                acyclic_cmap_list.extend(cmap_list)

        return acyclic_cmap_list


@dataclass(frozen=True)
class ComplexToRealStrategy:
    complexToRealFunction: Callable[[numpy.ndarray], numpy.ndarray]
    isCyclic: bool


@dataclass(frozen=True)
class ScalarTransformation:
    transformFunction: Callable[[numpy.ndarray], numpy.ndarray]


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

    def __floordiv__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width // other
            h = self.height // other
            return ImageExtent(width=w, height=h)


class ImageSequence(Sequence, Observable, Observer):
    @abstractmethod
    def setCurrentDatasetIndex(self, index: int) -> None:
        pass

    @abstractmethod
    def getCurrentDatasetIndex(self) -> int:
        pass

    @abstractmethod
    def getWidth(self) -> int:
        pass

    @abstractmethod
    def getHeight(self) -> int:
        pass


class CropSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', False)
        self.centerXInPixels = settingsGroup.createIntegerEntry('CenterXInPixels', 32)
        self.centerYInPixels = settingsGroup.createIntegerEntry('CenterYInPixels', 32)
        self.extentXInPixels = settingsGroup.createIntegerEntry('ExtentXInPixels', 64)
        self.extentYInPixels = settingsGroup.createIntegerEntry('ExtentYInPixels', 64)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> CropSettings:
        settings = cls(settingsRegistry.createGroup('Crop'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class CropSizer(Observer, Observable):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: CropSettings) -> None:
        super().__init__()
        self._settings = settings
        self._imageWidth = CropSizer.MAX_INT  # FIXME use real image width
        self._imageHeight = CropSizer.MAX_INT  # FIXME use real image height

    @classmethod
    def createInstance(cls, settings: CropSettings) -> CropSizer:
        sizer = cls(settings)
        settings.addObserver(sizer)
        return sizer

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.value

    def getExtentXLimits(self) -> Interval[int]:
        return Interval[int](1, self._imageWidth)

    def getExtentX(self) -> int:
        limits = self.getExtentXLimits()
        return limits.clamp(self._settings.extentXInPixels.value)

    def getCenterXLimits(self) -> Interval[int]:
        radius = self.getExtentX() // 2
        return Interval[int](radius, self._imageWidth - 1 - radius)

    def getCenterX(self) -> int:
        limits = self.getCenterXLimits()
        return limits.clamp(self._settings.centerXInPixels.value)

    def getSliceX(self) -> slice:
        center = self.getCenterX()
        radius = self.getExtentX() // 2
        return slice(center - radius, center + radius + 1)

    def getExtentYLimits(self) -> Interval[int]:
        return Interval[int](1, self._imageHeight)

    def getExtentY(self) -> int:
        limits = self.getExtentYLimits()
        return limits.clamp(self._settings.extentYInPixels.value)

    def getCenterYLimits(self) -> Interval[int]:
        radius = self.getExtentY() // 2
        return Interval[int](radius, self._imageHeight - 1 - radius)

    def getCenterY(self) -> int:
        limits = self.getCenterYLimits()
        return limits.clamp(self._settings.centerYInPixels.value)

    def getSliceY(self) -> slice:
        center = self.getCenterY()
        radius = self.getExtentY() // 2
        return slice(center - radius, center + radius + 1)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class CroppedImageSequence(ImageSequence):
    def __init__(self, sizer: CropSizer, imageSequence: ImageSequence) -> None:
        super().__init__()
        self._sizer = sizer
        self._imageSequence = imageSequence

    @classmethod
    def createInstance(cls, sizer: CropSizer,
                       imageSequence: ImageSequence) -> CroppedImageSequence:
        croppedImageSequence = cls(sizer, imageSequence)
        sizer.addObserver(croppedImageSequence)
        imageSequence.addObserver(croppedImageSequence)
        return croppedImageSequence

    def setCurrentDatasetIndex(self, index: int) -> None:
        self._imageSequence.setCurrentDatasetIndex(index)

    def getCurrentDatasetIndex(self) -> int:
        return self._imageSequence.getCurrentDatasetIndex()

    def getWidth(self) -> int:
        return self._imageSequence.getWidth()

    def getHeight(self) -> int:
        return self._imageSequence.getHeight()

    def __getitem__(self, index: int) -> numpy.ndarray:
        img = self._imageSequence[index]

        if self._sizer.isCropEnabled():
            sliceX = self._sizer.getSliceX()
            sliceY = self._sizer.getSliceY()
            img = numpy.copy(img[sliceY, sliceX])

        return img

    def __len__(self) -> int:
        return len(self._imageSequence)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._imageSequence:
            self.notifyObservers()


class CropPresenter(Observer, Observable):
    def __init__(self, settings: CropSettings, sizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: CropSettings, sizer: CropSizer) -> CropPresenter:
        presenter = cls(settings, sizer)
        sizer.addObserver(presenter)
        return presenter

    def isCropEnabled(self) -> bool:
        return self._sizer.isCropEnabled()

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getMinCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimits().lower

    def getMaxCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimits().upper

    def getCenterXInPixels(self) -> int:
        return self._sizer.getCenterX()

    def setCenterXInPixels(self, value: int) -> None:
        self._settings.centerXInPixels.value = value

    def getMinCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimits().lower

    def getMaxCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimits().upper

    def getCenterYInPixels(self) -> int:
        return self._sizer.getCenterY()

    def setCenterYInPixels(self, value: int) -> None:
        self._settings.centerYInPixels.value = value

    def getMinExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimits().lower

    def getMaxExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimits().upper

    def getExtentXInPixels(self) -> int:
        return self._sizer.getExtentX()

    def setExtentXInPixels(self, value: int) -> None:
        self._settings.extentXInPixels.value = value

    def getMinExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimits().lower

    def getMaxExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimits().upper

    def getExtentYInPixels(self) -> int:
        return self._sizer.getExtentY()

    def setExtentYInPixels(self, value: int) -> None:
        self._settings.extentYInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
