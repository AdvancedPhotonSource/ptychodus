import logging

from ptychodus.api.diffraction import Polarization
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product, ProductFileReader

from ..diffraction import AssembledDiffractionDataset, PatternSizer
from .geometry import ProductGeometry
from .item import ProductRepositoryItem, ProductState
from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItemFactory
from .probe import ProbeRepositoryItemFactory
from .repository import ProductRepository
from .probe_positions import ProbePositionsRepositoryItemFactory
from .settings import ProductSettings

logger = logging.getLogger(__name__)


class ProductRepositoryItemFactory:
    def __init__(
        self,
        settings: ProductSettings,
        pattern_sizer: PatternSizer,
        scan_item_factory: ProbePositionsRepositoryItemFactory,
        probe_item_factory: ProbeRepositoryItemFactory,
        object_item_factory: ObjectRepositoryItemFactory,
        repository: ProductRepository,
        file_reader_chooser: PluginChooser[ProductFileReader],
    ) -> None:
        super().__init__()
        self._settings = settings
        self._pattern_sizer = pattern_sizer
        self._scan_item_factory = scan_item_factory
        self._probe_item_factory = probe_item_factory
        self._object_item_factory = object_item_factory
        self._repository = repository
        self._file_reader_chooser = file_reader_chooser

    @staticmethod
    def _bind_dataset_geometry(
        geometry: ProductGeometry, dataset: AssembledDiffractionDataset | None
    ) -> None:
        """Push the dataset's detector extent and raw pixel geometry into ``geometry``
        so probe & object items built next see a valid geometry inside their own
        __init__ rebuild. This keeps the observer-triggered rebuild that fires later
        (from ProductRepositoryItem._bind_dataset) a no-op — the setters short-circuit
        on unchanged values, avoiding a spurious rebuild during ProductRepositoryItem
        construction (which would fire index<0 warnings from the repository)."""
        if dataset is None:
            return
        geometry.set_detector_extent(dataset.get_metadata().detector_extent)
        geometry.set_detector_pixel_geometry(dataset.get_raw_pixel_geometry())

    def create_from_values(
        self,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
        tomography_angle_deg: float | None = None,
        tilt_angle_deg: float | None = None,
        polarization: Polarization | None = None,
        dataset: AssembledDiffractionDataset | None = None,
    ) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            self._repository,
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
            tilt_angle_deg=tilt_angle_deg,
            polarization=polarization,
        )

        # probe_photon_count auto-estimation from diffraction data now lives in the
        # controller layer (see ProductEditorViewController._estimate_probe_photon_count).
        # This factory takes whatever value the caller supplied.

        scan_item = self._scan_item_factory.create()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        self._bind_dataset_geometry(geometry, dataset)
        probe_item = self._probe_item_factory.create(geometry)
        object_item = self._object_item_factory.create(geometry)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            probe_positions_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            losses=list(),
            dataset=dataset,
        )

    def create_from_product(
        self, product: Product, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            self._repository,
            name=product.metadata.name,
            comments=product.metadata.comments,
            detector_distance_m=product.metadata.detector_distance_m,
            probe_energy_eV=product.metadata.probe_energy_eV,
            probe_photon_count=product.metadata.probe_photon_count,
            exposure_time_s=product.metadata.exposure_time_s,
            mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
            tomography_angle_deg=product.metadata.tomography_angle_deg,
            tilt_angle_deg=product.metadata.tilt_angle_deg,
            polarization=product.metadata.polarization,
        )

        scan_item = self._scan_item_factory.create(product.probe_positions)
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        self._bind_dataset_geometry(geometry, dataset)
        probe_item = self._probe_item_factory.create(geometry, product.probes)
        object_item = self._object_item_factory.create(geometry, product.object_)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            probe_positions_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            losses=product.losses,
            dataset=dataset,
        )

    def create_pending_stub(self, name: str = 'Unnamed') -> ProductRepositoryItem:
        """Build a fresh ProductRepositoryItem in the 'pending' state with default
        subgroups and no dataset. Its inner content is replaced later via
        ProductRepositoryItem.copy_contents_from once the source dataset finishes
        loading."""
        metadata_item = MetadataRepositoryItem(self._settings, self._repository, name=name)
        scan_item = self._scan_item_factory.create()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create(geometry)
        object_item = self._object_item_factory.create(geometry)
        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            probe_positions_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            losses=list(),
            dataset=None,
            state=ProductState.PENDING,
        )

    def create_from_settings(
        self, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ProductRepositoryItem:
        file_path = self._settings.file_path.get_value()

        if file_path.is_file():
            file_type = self._file_reader_chooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{file_path}" as "{file_type}"')
            file_reader = self._file_reader_chooser.get_current_plugin().strategy

            try:
                product = file_reader.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                return self.create_from_product(product, dataset=dataset)

        metadata_item = MetadataRepositoryItem(self._settings, self._repository)
        scan_item = self._scan_item_factory.create_from_settings()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        self._bind_dataset_geometry(geometry, dataset)
        probe_item = self._probe_item_factory.create_from_settings(geometry, dataset=dataset)
        object_item = self._object_item_factory.create_from_settings(geometry, dataset=dataset)

        item = ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            probe_positions_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            losses=list(),
            dataset=dataset,
        )
        logger.debug(f'Created product from settings: {item.get_name()}')
        return item
