from __future__ import annotations
from collections.abc import Callable, Iterator, Sequence
from typing import override

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
import matplotlib
import numpy

from ptychodus.api.observer import Observable
from ptychodus.api.patterns import PixelGeometry
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import (NumberArrayType, RealArrayType, ScalarTransformation,
                                         VisualizationProduct)

from .colorAxis import ColorAxis
from .colormap import ColormapParameter
from .components import DataArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class RendererFactory:

    def createMappedRendererVariants(cls, transformation: ScalarTransformationParameter,
                                     colorAxis: ColorAxis) -> Sequence[Renderer]:
        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        cyclicColormapNames = ['hsv', 'twilight', 'twilight_shifted']

        cyclicColormapChooser = PluginChooser[Colormap]()
        acyclicColormapChooser = PluginChooser[Colormap]()

        for name, cmap in matplotlib.colormaps.items():
            if name in cyclicColormapNames:
                cyclicColormapChooser.registerPlugin(cmap, displayName=name)
            else:
                acyclicColormapChooser.registerPlugin(cmap, displayName=name)

        cyclicColormapChooser.setCurrentPluginByName('hsv')
        acyclicColormapChooser.setCurrentPluginByName('gray')

        intensity = cls(array,
                        displayRange,
                        transformChooser,
                        'Intensity',
                        array.getIntensity,
                        acyclicColormapChooser,
                        isCyclic=False)
        variants: list[Renderer] = [intensity]

        if isComplex:
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
            variants.extend([amplitude, phase, phaseUnwrapped, real, imag])

        return variants

    def createModelRendererVariants(cls, array: VisualizationArray, displayRange: DisplayRange,
                                    transformChooser: PluginChooser[ScalarTransformation], *,
                                    isComplex: bool) -> Sequence[Renderer]:
        if not isComplex:
            return []

        variantChooser = PluginChooser[CylindricalColorModel]()
        variantChooser.registerPlugin(
            HSVSaturationColorModel(),
            simpleName='HSV-S',
            displayName='HSV Saturation',
        )
        variantChooser.registerPlugin(
            HSVValueColorModel(),
            simpleName='HSV-V',
            displayName='HSV Value',
        )
        variantChooser.registerPlugin(
            HSVAlphaColorModel(),
            simpleName='HSV-A',
            displayName='HSV Alpha',
        )
        variantChooser.registerPlugin(
            HLSLightnessColorModel(),
            simpleName='HLS-L',
            displayName='HLS Lightness',
        )
        variantChooser.registerPlugin(
            HLSSaturationColorModel(),
            simpleName='HLS-S',
            displayName='HLS Saturation',
        )
        variantChooser.registerPlugin(
            HLSAlphaColorModel(),
            simpleName='HLS-A',
            displayName='HLS Alpha',
        )
        variantChooser.setCurrentPluginByName('HSV-V')
        return [cls(array, displayRange, transformChooser, variantChooser)]
