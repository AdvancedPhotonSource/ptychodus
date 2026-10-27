from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging
import threading

from ptychodus.api.accumulate import compute_diffraction_moments, estimate_incident_beam
from ptychodus.api.diffraction import Polarization
from ptychodus.api.parallax import ParallaxEstimate, estimate_defocus_from_parallax
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.probe import OPRWeightPolicy
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter

from ..diffraction import AssembledDiffractionDataset
from ..file_types import FileTypeChooser
from ..task_manager import TaskManager
from .item import ProductRepositoryItem, ProductState
from .item_factory import ProductRepositoryItemFactory
from .monitor import ProductTaskMonitor
from .object.builder_factory import ObjectBuilderFactory
from .object.settings import ObjectSettings
from .object_repository import ObjectRepository
from .probe.builder_factory import ProbeBuilderFactory
from .probe.settings import ProbeSettings
from .probe_repository import ProbeRepository
from .repository import ProductRepository
from .probe_positions.builder_factory import ProbePositionsBuilderFactory
from .probe_positions.settings import ProbePositionsSettings
from .scan_repository import ProbePositionsRepository
from .settings import ProductSettings

logger = logging.getLogger(__name__)


class PositionsStreamingContext:
    def __init__(self) -> None:
        self._positions_x_m: list[float] = []
        self._triggers_x: list[int] = []
        self._positions_y_m: list[float] = []
        self._triggers_y: list[int] = []

    def start(self) -> None:
        self._positions_x_m.clear()
        self._triggers_x.clear()
        self._positions_y_m.clear()
        self._triggers_y.clear()

    def append_positions_x(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_x_m.extend(values_m)
        self._triggers_x.extend(trigger_counts)

    def append_positions_y(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_y_m.extend(values_m)
        self._triggers_y.extend(trigger_counts)

    def stop(self) -> None:
        pass  # TODO


class ProbePositionsAPI:
    def __init__(
        self,
        settings: ProbePositionsSettings,
        repository: ProbePositionsRepository,
        builder_factory: ProbePositionsBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory
        self.file_readers = builder_factory.file_readers
        self.file_writers = builder_factory.file_writers

    def create_streaming_context(self) -> PositionsStreamingContext:
        return PositionsStreamingContext()

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_probe_positions(
        self, index: int, builder_name: str, builder_parameters: Mapping[str, Any] | None = None
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builder_factory.create(builder_name)
        except KeyError:
            logger.warning(f'Failed to create builder {builder_name}!')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Scan builder "{builder.get_name()}" does not have parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_probe_positions_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builder_factory.create_from_settings()
        except KeyError:
            logger.warning('Failed to create builder from settings!')
            return

        item.set_builder(builder)

    def open_probe_positions(
        self, index: int, file_path: Path, *, file_type: str | None = None
    ) -> None:
        builder = self._builder_factory.create_scan_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open scan {index}!')
        else:
            item.set_builder(builder)

    def copy_scan(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source scan {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination scan {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def save_scan(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save scan {index}!')
        else:
            self._builder_factory.save_scan(file_path, file_type, item.get_probe_positions())


class ProbeAPI:
    def __init__(
        self,
        settings: ProbeSettings,
        repository: ProbeRepository,
        builder_factory: ProbeBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory
        self.file_readers = builder_factory.file_readers
        self.file_writers = builder_factory.file_writers

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_probe(
        self,
        index: int,
        builder_name: str,
        builder_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create(
                builder_name, has_diffraction_data=dataset is not None
            )
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder {builder_name}: {exc}')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Probe builder "{builder.get_name()}" does not have'
                        f' parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_probe_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create_from_settings(
                has_diffraction_data=dataset is not None
            )
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder from settings: {exc}')
            return

        item.set_builder(builder)

    def peek_opr_weights(
        self, file_path: Path, *, file_type: str | None = None
    ) -> tuple[int, int] | None:
        """Shape of the OPR weights in a probe file, or ``None`` when it carries none.

        Returns ``(number of weight rows, number of coherent modes)``. The row count is
        the number of probe positions the file was solved on, which is what a caller
        compares against this run before choosing how to ingest it.

        The file is read to answer this, and a read error propagates rather than being
        reported as an absence.
        """
        probes = self._builder_factory.read_probe_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )
        weights = probes.get_opr_weights_or_none()

        if weights is None:
            return None

        return int(weights.shape[0]), int(weights.shape[1])

    def open_probe(
        self,
        index: int,
        file_path: Path,
        *,
        file_type: str | None = None,
        opr_weight_policy: OPRWeightPolicy | None = None,
    ) -> None:
        builder = self._builder_factory.create_probe_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
            opr_weight_policy=opr_weight_policy,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.set_builder(builder)

    def copy_probe(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source probe {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination probe {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def save_probe(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._builder_factory.save_probe(file_path, file_type, item.get_probes())


class ObjectAPI:
    def __init__(
        self,
        settings: ObjectSettings,
        repository: ObjectRepository,
        builder_factory: ObjectBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory
        self.file_readers = builder_factory.file_readers
        self.file_writers = builder_factory.file_writers

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_object(
        self,
        index: int,
        builder_name: str,
        builder_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create(
                builder_name, has_diffraction_data=dataset is not None
            )
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder {builder_name}: {exc}')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Object builder "{builder.get_name()}" does not have'
                        f' parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_object_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create_from_settings(
                has_diffraction_data=dataset is not None
            )
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder from settings: {exc}')
            return

        item.set_builder(builder)

    def open_object(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
        builder = self._builder_factory.create_object_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.set_builder(builder)

    def copy_object(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source object {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination object {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def save_object(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._builder_factory.save_object(file_path, file_type, item.get_object())


class ProductAPI:
    def __init__(
        self,
        settings: ProductSettings,
        repository: ProductRepository,
        item_factory: ProductRepositoryItemFactory,
        file_reader_chooser: PluginChooser[ProductFileReader],
        file_writer_chooser: PluginChooser[ProductFileWriter],
        task_manager: TaskManager,
        task_monitor: ProductTaskMonitor,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._item_factory = item_factory
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)
        self._task_manager = task_manager
        self._task_monitor = task_monitor
        self._queue_lock = threading.Lock()
        self._queued_total = 0
        self._queued_done = 0

    def get_task_monitor(self) -> ProductTaskMonitor:
        return self._task_monitor

    def _begin_queued_product(self) -> None:
        """Register one queued product, entering the monitor for the first of a burst."""
        with self._queue_lock:
            if self._queued_total == 0:
                self._queued_done = 0
                self._task_monitor.__enter__()

            self._queued_total += 1
            self._task_monitor.update_progress(self._queued_done, self._queued_total)

    def _end_queued_product(self) -> None:
        """Retire one queued product, leaving the monitor once the burst is drained."""
        with self._queue_lock:
            self._queued_done += 1
            self._task_monitor.update_progress(self._queued_done, self._queued_total)

            if self._queued_done >= self._queued_total:
                self._queued_total = 0
                self._queued_done = 0
                self._task_monitor.__exit__(None, None, None)

    def _insert_via_queue(
        self,
        dataset: AssembledDiffractionDataset | None,
        build: Callable[[], ProductRepositoryItem],
        *,
        block: bool,
        stub_name: str,
    ) -> int:
        """If dataset is None or already loaded, run build() synchronously and insert.
        Otherwise, insert a pending stub, then enqueue construction to run after the
        dataset's load task finishes (via the shared FIFO background worker)."""
        if dataset is None or not dataset.is_load_in_progress():
            item = build()
            return self._repository.insert_product(item)

        finished_event = dataset.get_last_load_finished_event()

        if block:
            if finished_event is not None:
                while not self._task_manager.is_stopping:
                    if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                        break
            error = dataset.get_last_load_error()
            if error is not None:
                raise RuntimeError('Diffraction dataset failed to load') from error
            item = build()
            return self._repository.insert_product(item)

        stub = self._item_factory.create_pending_stub(name=stub_name)
        index = self._repository.insert_product(stub)

        # Mark the monitor busy from enqueue, not from when the background step runs:
        # a queued product spends nearly all its life waiting behind the dataset load,
        # and that is exactly the window in which Stop has to be reachable.
        self._begin_queued_product()

        def background_finalize() -> Callable[[], None]:
            stopping = self._task_monitor.is_stopping
            error = None if stopping else dataset.get_last_load_error()

            def foreground_finalize() -> None:
                try:
                    if stopping:
                        logger.info(f'Cancelling queued product {index}: stop requested.')
                        stub.set_state(ProductState.FAILED)
                        return

                    if error is not None:
                        logger.error(
                            f'Cancelling queued product {index} because dataset '
                            f'failed to load: {error!r}'
                        )
                        stub.set_state(ProductState.FAILED)
                        return

                    try:
                        real = build()
                    except Exception:
                        logger.exception(f'Queued product {index} construction failed')
                        stub.set_state(ProductState.FAILED)
                        return

                    stub.copy_contents_from(real)
                    stub.set_state(ProductState.READY)
                finally:
                    self._end_queued_product()

            return foreground_finalize

        self._task_manager.put_background_task(background_finalize)
        return index

    def insert_new_product(
        self,
        name: str = 'Unnamed',
        *,
        comments: str = '',
        detector_distance_m: float | None = None,
        focus_object_distance_m: float | None = None,
        photon_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_per_kg: float | None = None,
        tomography_angle_deg: float | None = None,
        tilt_angle_deg: float | None = None,
        polarization: Polarization | None = None,
        far_field: bool | None = None,
        i0_photons_per_count: float | None = None,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_values(
                name=name,
                comments=comments,
                detector_distance_m=detector_distance_m,
                focus_object_distance_m=focus_object_distance_m,
                photon_energy_eV=photon_energy_eV,
                probe_photon_count=probe_photon_count,
                exposure_time_s=exposure_time_s,
                mass_attenuation_m2_per_kg=mass_attenuation_m2_per_kg,
                tomography_angle_deg=tomography_angle_deg,
                tilt_angle_deg=tilt_angle_deg,
                polarization=polarization,
                far_field=far_field,
                i0_photons_per_count=i0_photons_per_count,
                dataset=dataset,
            )

        return self._insert_via_queue(dataset, build, block=block, stub_name=name)

    def insert_product(
        self,
        product: Product,
        *,
        name: str | None = None,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        """Insert a copy of product, optionally overriding the name it carries.

        Passing name here rather than renaming the inserted item keeps the repository from
        ever reserving the product's own name as a transient collision, which would push
        the intended name onto a counter suffix.
        """

        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_product(product, name=name, dataset=dataset)

        return self._insert_via_queue(
            dataset,
            build,
            block=block,
            stub_name=product.metadata.name if name is None else name,
        )

    def insert_product_from_settings(
        self,
        *,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_settings(dataset=dataset)

        return self._insert_via_queue(dataset, build, block=block, stub_name='Unnamed')

    def get_item(self, product_index: int) -> ProductRepositoryItem:
        return self._repository[product_index]

    def estimate_defocus_from_parallax(
        self,
        product_index: int,
        *,
        bin_size_px: int = 4,
        minimum_significance: float = 3.0,
        minimum_defocus_dof: float = 1.0,
        on_finished: Callable[[ParallaxEstimate | None, bool], None] | None = None,
    ) -> None:
        """Measure the defocus from the parallax between bright-field images, in the background.

        One pass over the bound patterns, binned *bin_size_px* detector pixels a side and
        corrected for beam decay by I0, gives :func:`estimate_defocus_from_parallax`'s
        estimate. When :meth:`ParallaxEstimate.is_resolved` holds for the thresholds, the
        product's focus position is set to the negated defocus, in the foreground.
        *on_finished* then receives the estimate, None when none could be made, and
        whether it was applied. Far field only.
        """
        item = self._repository[product_index]
        dataset = item.get_dataset()
        metadata = item.get_metadata_item().get_metadata()

        def finish(estimate: ParallaxEstimate | None, applied: bool) -> None:
            if on_finished is not None:
                on_finished(estimate, applied)

        if dataset is None:
            logger.warning('No diffraction data to estimate the defocus from.')
            finish(None, False)
            return

        if not metadata.far_field:
            logger.warning('The defocus is estimated from far-field parallax only.')
            finish(None, False)
            return

        data = dataset.get_assembled_data()
        positions = item.get_probe_positions_item().get_probe_positions()
        geometry = item.get_geometry()
        object_geometry = geometry.get_object_geometry()
        probe_geometry = geometry.get_probe_geometry()

        def estimate_in_background() -> Callable[[], None]:
            estimate: ParallaxEstimate | None = None

            try:
                moments = compute_diffraction_moments(
                    data.get_indexes(),
                    data.get_patterns(),
                    bad_pixels=data.get_bad_pixels(),
                    background=data.get_background(),
                    flat_field=data.get_flat_field(),
                    binning=BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px),
                )

                if moments is not None:
                    moments = estimate_incident_beam(
                        moments,
                        positions.get_probe_photon_counts_by_index(),
                        i0_photons_per_count=metadata.i0_photons_per_count,
                    ).moments
                    estimate = estimate_defocus_from_parallax(
                        moments,
                        positions,
                        object_geometry,
                        probe_geometry,
                        photon_wavelength_m=metadata.photon_wavelength_m,
                    )
            except Exception:
                logger.exception('Failed to estimate the defocus from the parallax!')

            def apply_in_foreground() -> None:
                applied = estimate is not None and estimate.is_resolved(
                    minimum_significance=minimum_significance,
                    minimum_defocus_dof=minimum_defocus_dof,
                )

                if estimate is None:
                    logger.warning('The parallax gives no defocus estimate.')
                elif applied:
                    logger.info(
                        f'Parallax defocus {estimate.defocus_distance_m * 1e3:.4g} '
                        f'+/- {estimate.standard_error_m * 1e3:.2g} mm; setting the focus position.'
                    )
                    item.get_metadata_item().focus_object_distance_m.set_value(
                        -estimate.defocus_distance_m
                    )
                else:
                    logger.warning(
                        f'Parallax defocus {estimate.defocus_distance_m * 1e3:.4g} '
                        f'+/- {estimate.standard_error_m * 1e3:.2g} mm is not resolved; '
                        'leaving the focus position.'
                    )

                finish(estimate, applied)

            return apply_in_foreground

        self._task_manager.put_background_task(estimate_in_background)

    def open_product(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        if file_path.is_file():
            plugin = self.file_readers.select(file_type)
            logger.debug(f'Reading "{file_path}" as "{plugin.simple_name}"')

            try:
                product = plugin.strategy.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                return self.insert_product(product, dataset=dataset, block=block)
        else:
            logger.warning(f'Refusing to create product with invalid file path "{file_path}"')

        return -1

    def rename_product(self, product_index: int, new_name: str) -> None:
        try:
            item = self._repository[product_index]
        except IndexError:
            logger.warning(f'Failed to access product {product_index} for renaming!')
            return

        item.set_name(new_name)

    def save_product(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save product {index}!')
            return

        plugin = self.file_writers.select(file_type)
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, item.get_product())
