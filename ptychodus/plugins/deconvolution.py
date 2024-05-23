import skimage.restoration

from ptychodus.api.fluorescence import DeconvolutionStrategy, ElementMap
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product


class IdentityDeconvolution(DeconvolutionStrategy):

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        return emap


class RichardsonLucyDeconvolution(DeconvolutionStrategy):

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        cps = skimage.restoration.richardson_lucy(emap.counts_per_second,
                                                  product.probe.getIntensity())
        return ElementMap(emap.name, cps)


class WienerDeconvolution(DeconvolutionStrategy):

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        balance = 0.05  # TODO
        cps = skimage.restoration.wiener(emap.counts_per_second, product.probe.getIntensity(),
                                         balance)
        return ElementMap(emap.name, cps)


class UnsupervisedWienerDeconvolution(DeconvolutionStrategy):

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        cps, _ = skimage.restoration.unsupervised_wiener(emap.counts_per_second,
                                                         product.probe.getIntensity())
        return ElementMap(emap.name, cps)


def registerPlugins(registry: PluginRegistry) -> None:
    # NOTE See https://scikit-image.org/docs/stable/api/skimage.restoration.html
    # TODO Implement method from https://doi.org/10.1364/OE.20.018287
    registry.deconvolutionStrategies.registerPlugin(
        IdentityDeconvolution(),
        displayName='Identity',
    )
    registry.deconvolutionStrategies.registerPlugin(
        RichardsonLucyDeconvolution(),
        displayName='Richardson-Lucy',
    )
    registry.deconvolutionStrategies.registerPlugin(
        WienerDeconvolution(),
        displayName='Wiener',
    )
    registry.deconvolutionStrategies.registerPlugin(
        UnsupervisedWienerDeconvolution(),
        displayName='Unsupervised Wiener',
    )
