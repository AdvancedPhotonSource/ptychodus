import logging
import time

import numpy

from ptychodus.api.simulate.diffraction import generate_diffraction_data

from ..diffraction import DiffractionDatasetRepository
from ..product import ProductRepository
from .settings import DiffractionSimulatorSettings

logger = logging.getLogger(__name__)


class DiffractionSimulator:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: DiffractionSimulatorSettings,
        diffraction_repository: DiffractionDatasetRepository,
        repository: ProductRepository,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._diffraction_repository = diffraction_repository
        self._repository = repository

    def simulate(self, product_index: int) -> int:
        product_item = self._repository[product_index]
        product = product_item.get_product()
        rng = self._rng if self._settings.add_poisson_noise.get_value() else None

        logger.info('Computing diffraction data...')
        tic = time.perf_counter()
        data = generate_diffraction_data(product, rng=rng)
        toc = time.perf_counter()
        logger.info(f'Computed diffraction data in {toc - tic:.4f} seconds.')

        dataset = self._diffraction_repository.create_dataset(product_item.get_name())
        dataset_index = self._diffraction_repository.insert_dataset(dataset)
        dataset.set_assembled_patterns(data)
        return dataset_index
