import skimage.restoration

from ptychodus.api.fluorescence import DeconvolutionStrategy, ElementMap
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product


class IdentityDeconvolution(DeconvolutionStrategy):
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        return emap


class RichardsonLucyDeconvolution(DeconvolutionStrategy):
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        probe_intensity = product.probes.get_probe_no_opr().get_intensity()
        cps = skimage.restoration.richardson_lucy(emap.counts_per_second, probe_intensity)
        return ElementMap(emap.name, cps)


class WienerDeconvolution(DeconvolutionStrategy):
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        probe_intensity = product.probes.get_probe_no_opr().get_intensity()
        balance = 0.05  # TODO
        cps = skimage.restoration.wiener(emap.counts_per_second, probe_intensity, balance)
        return ElementMap(emap.name, cps)


class UnsupervisedWienerDeconvolution(DeconvolutionStrategy):
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        probe_intensity = product.probes.get_probe_no_opr().get_intensity()
        cps, _ = skimage.restoration.unsupervised_wiener(emap.counts_per_second, probe_intensity)
        return ElementMap(emap.name, cps)


def register_plugins(registry: PluginRegistry) -> None:
    # NOTE See https://scikit-image.org/docs/stable/api/skimage.restoration.html
    # TODO Implement method from https://doi.org/10.1364/OE.20.018287
    registry.deconvolution_strategies.register_plugin(
        IdentityDeconvolution(),
        display_name='Identity',
    )
    registry.deconvolution_strategies.register_plugin(
        RichardsonLucyDeconvolution(),
        display_name='Richardson-Lucy',
    )
    registry.deconvolution_strategies.register_plugin(
        WienerDeconvolution(),
        display_name='Wiener',
    )
    registry.deconvolution_strategies.register_plugin(
        UnsupervisedWienerDeconvolution(),
        display_name='Unsupervised Wiener',
    )
