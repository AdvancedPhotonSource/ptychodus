import logging
import time

import numpy

from ptychodus.api.diffraction_gen import generate_diffraction_data

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductRepository
from .settings import DiffractionSimulatorSettings

logger = logging.getLogger(__name__)


class DiffractionSimulator:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: DiffractionSimulatorSettings,
        dataset: AssembledDiffractionDataset,
        repository: ProductRepository,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._dataset = dataset
        self._repository = repository

    def simulate(self, product_index: int) -> None:
        product = self._repository[product_index].get_product()
        rng = self._rng if self._settings.add_poisson_noise.get_value() else None

        logger.info('Computing diffraction data...')
        tic = time.perf_counter()
        data = generate_diffraction_data(product, rng=rng)
        toc = time.perf_counter()
        logger.info(f'Computed diffraction data in {toc - tic:.4f} seconds.')

        self._dataset.set_assembled_patterns(data)
