from __future__ import annotations
from collections.abc import Callable, Sequence

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
import matplotlib
import numpy

from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import VisualizationArray


class MappedColorizer(Colorizer):

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation], name: str,
                 arrayComponent: Callable[[], RealArrayType],
                 variantChooser: PluginChooser[Colormap], *, isCyclic: bool) -> None:
        super().__init__(array, displayRange, transformChooser)
        self._name = name
        self._arrayComponent = arrayComponent
        self._variantChooser = variantChooser
        self._variantChooser.addObserver(self)
        self._isCyclic = isCyclic

    @classmethod
    def createColorizerVariants(
            cls, array: VisualizationArray, displayRange: DisplayRange,
            transformChooser: PluginChooser[ScalarTransformation]) -> Sequence[Colorizer]:
        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        cyclicColormapNames = ['hsv', 'twilight', 'twilight_shifted']

        cyclicColormapChooser = PluginChooser[Colormap]()
        acyclicColormapChooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            if name in cyclicColormapNames:
                cyclicColormapChooser.registerPlugin(cmap, simpleName=name)
            else:
                acyclicColormapChooser.registerPlugin(cmap, simpleName=name)

        cyclicColormapChooser.setCurrentPluginByName('hsv')
        acyclicColormapChooser.setCurrentPluginByName('viridis')

        intensity = cls(array,
                        displayRange,
                        transformChooser,
                        'Intensity',
                        array.getIntensity,
                        acyclicColormapChooser,
                        isCyclic=False)
        amplitude = cls(array,
                        displayRange,
                        transformChooser,
                        'Amplitude',
                        array.getAmplitude,
                        acyclicColormapChooser,
                        isCyclic=False)
        phase = cls(array,
                    displayRange,
                    transformChooser,
                    'Phase',
                    array.getPhaseInRadians,
                    cyclicColormapChooser,
                    isCyclic=True)
        phaseUnwrapped = cls(array,
                             displayRange,
                             transformChooser,
                             'Phase (Unwrapped)',
                             array.getPhaseUnwrappedInRadians,
                             acyclicColormapChooser,
                             isCyclic=False)
        real = cls(array,
                   displayRange,
                   transformChooser,
                   'Real',
                   array.getRealPart,
                   acyclicColormapChooser,
                   isCyclic=False)
        imag = cls(array,
                   displayRange,
                   transformChooser,
                   'Imaginary',
                   array.getImaginaryPart,
                   acyclicColormapChooser,
                   isCyclic=False)

        return [intensity, amplitude, phase, phaseUnwrapped, real, imag]

    @property
    def name(self) -> str:
        return self._name

    def getVariantNameList(self) -> Sequence[str]:
        return self._variantChooser.getDisplayNameList()

    def getVariantName(self) -> str:
        return self._variantChooser.currentPlugin.displayName

    def setVariantByName(self, name: str) -> None:
        self._variantChooser.setCurrentPluginByName(name)

    def getColorSamples(self, normalizedValues: RealArrayType) -> RealArrayType:
        cmap = self._variantChooser.currentPlugin.strategy
        return cmap(normalizedValues)

    def isCyclic(self) -> bool:
        return self._isCyclic

    def getDataArray(self) -> RealArrayType:
        transform = self._transformChooser.currentPlugin.strategy
        values = self._arrayComponent()
        return transform(values)

    def __call__(self) -> RealArrayType:
        if self._displayRange.getUpper() <= self._displayRange.getLower():
            shape = self._arrayComponent().shape
            return numpy.zeros((*shape, 4))

        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)

        cmap = self._variantChooser.currentPlugin.strategy
        scalarMappable = ScalarMappable(norm, cmap)

        return scalarMappable.to_rgba(self.getDataArray())

    def update(self, observable: Observable) -> None:
        if observable is self._variantChooser:
            self.notifyObservers()
        else:
            super().update(observable)
