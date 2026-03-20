import numpy

from ptychodus.api.diffraction_gen import generate_diffraction_data

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductRepository


class DiffractionSimulator:
    def __init__(
        self,
        rng: numpy.random.Generator,
        dataset: AssembledDiffractionDataset,
        repository: ProductRepository,
    ) -> None:
        super().__init__()
        self._rng = rng
        self._dataset = dataset
        self._repository = repository

    def simulate(self, product_index: int, add_poisson_noise: bool = False) -> None:
        product = self._repository[product_index].get_product()
        rng = self._rng if add_poisson_noise else None
        data = generate_diffraction_data(product, rng=rng)
        self._dataset.set_assembled_patterns(data)
