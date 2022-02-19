from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy

from .observer import Observable, Observer


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
