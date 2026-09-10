"""Read and write CXI files (Coherent X-ray Imaging Data Bank format, spec v1.6).

Writers produce split ``diffraction.cxi`` / ``product.cxi`` files that follow the
CXI spec where a mapping exists, and stash ptychodus-specific fields (OPR weights,
loss history, layer spacing, etc.) under ``/entry_1/ptychodus/``. Readers accept
either split files or a single combined file that carries both raw detector data
and reconstructed image_N groups.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final
import logging

import h5py
import numpy

from ptychodus import __version__ as ptychodus_version
from ptychodus.api.constants import ELECTRON_VOLT_J
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import (
    LossValue,
    Product,
    ProductFileReader,
    ProductFileWriter,
    ProductMetadata,
)

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


CXI_VERSION: Final[int] = 160  # spec version 1.6 stored as int * 100
CXI_PIXEL_IS_INVALID: Final[int] = 0x00000001  # spec Table 7 / Table 11


class _P:
    """CXI HDF5 path constants used by both readers and writers."""

    ENTRY: Final[str] = '/entry_1'
    START_TIME: Final[str] = '/entry_1/start_time'
    TITLE: Final[str] = '/entry_1/title'
    EXPERIMENT_DESCRIPTION: Final[str] = '/entry_1/experiment_description'

    INSTRUMENT: Final[str] = '/entry_1/instrument_1'
    SOURCE: Final[str] = '/entry_1/instrument_1/source_1'
    SOURCE_ENERGY: Final[str] = '/entry_1/instrument_1/source_1/energy'
    ILLUMINATION: Final[str] = '/entry_1/instrument_1/source_1/illumination'  # legacy probe

    DETECTOR: Final[str] = '/entry_1/instrument_1/detector_1'
    DETECTOR_DATA: Final[str] = '/entry_1/instrument_1/detector_1/data'
    DETECTOR_DISTANCE: Final[str] = '/entry_1/instrument_1/detector_1/distance'
    DETECTOR_X_PIXEL_SIZE: Final[str] = '/entry_1/instrument_1/detector_1/x_pixel_size'
    DETECTOR_Y_PIXEL_SIZE: Final[str] = '/entry_1/instrument_1/detector_1/y_pixel_size'
    DETECTOR_MASK: Final[str] = '/entry_1/instrument_1/detector_1/mask'

    SAMPLE: Final[str] = '/entry_1/sample_1'
    SAMPLE_NAME: Final[str] = '/entry_1/sample_1/name'
    SAMPLE_GEOMETRY: Final[str] = '/entry_1/sample_1/geometry_1'
    TRANSLATION: Final[str] = '/entry_1/sample_1/geometry_1/translation'

    DATA_GROUP: Final[str] = '/entry_1/data_1'
    DATA_DATA: Final[str] = '/entry_1/data_1/data'
    DATA_TRANSLATION: Final[str] = '/entry_1/data_1/translation'  # legacy positions

    PROBE_IMAGE: Final[str] = '/entry_1/image_1'
    PROBE_IMAGE_DATA: Final[str] = '/entry_1/image_1/data'
    OBJECT_IMAGE: Final[str] = '/entry_1/image_2'
    OBJECT_IMAGE_DATA: Final[str] = '/entry_1/image_2/data'

    PROCESS: Final[str] = '/entry_1/process_1'

    # ptychodus extension namespace ---------------------------------------------------
    PTYCHODUS: Final[str] = '/entry_1/ptychodus'
    PT_PROBE_PHOTON_COUNT: Final[str] = '/entry_1/ptychodus/probe_photon_count'
    PT_EXPOSURE_TIME: Final[str] = '/entry_1/ptychodus/exposure_time_s'
    PT_MASS_ATTENUATION: Final[str] = '/entry_1/ptychodus/mass_attenuation_m2_kg'
    PT_TOMOGRAPHY_ANGLE: Final[str] = '/entry_1/ptychodus/tomography_angle_deg'
    PT_POSITION_INDEXES: Final[str] = '/entry_1/ptychodus/probe_position_indexes'
    PT_PROBE_PIXEL_WIDTH: Final[str] = '/entry_1/ptychodus/probe_pixel_width_m'
    PT_PROBE_PIXEL_HEIGHT: Final[str] = '/entry_1/ptychodus/probe_pixel_height_m'
    PT_OPR_WEIGHTS: Final[str] = '/entry_1/ptychodus/opr_weights'
    PT_OBJECT_PIXEL_WIDTH: Final[str] = '/entry_1/ptychodus/object_pixel_width_m'
    PT_OBJECT_PIXEL_HEIGHT: Final[str] = '/entry_1/ptychodus/object_pixel_height_m'
    PT_OBJECT_CENTER_X: Final[str] = '/entry_1/ptychodus/object_center_x_m'
    PT_OBJECT_CENTER_Y: Final[str] = '/entry_1/ptychodus/object_center_y_m'
    PT_OBJECT_LAYER_SPACING: Final[str] = '/entry_1/ptychodus/object_layer_spacing_m'
    PT_LOSS_EPOCHS: Final[str] = '/entry_1/ptychodus/loss_epochs'
    PT_LOSS_VALUES: Final[str] = '/entry_1/ptychodus/loss_values'


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_scalar(h5_file: h5py.File, path: str) -> Any | None:
    """Return the value at *path*, or None if the path does not exist."""
    if path in h5_file:
        return h5_file[path][()]
    return None


def _read_string(h5_file: h5py.File, path: str) -> str:
    """Decode a possibly-bytes string dataset at *path*; return '' if absent."""
    value = _read_scalar(h5_file, path)
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return str(value)


class CXIDiffractionFileReader(DiffractionFileReader):
    """Read a diffraction dataset from a CXI file."""

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

            # Canonical data location (soft link in files we write); fall back to
            # the detector data path if the data_1 group is absent.
            data_path = _P.DATA_DATA if _P.DATA_DATA in h5_file else _P.DETECTOR_DATA
            data = h5_file.get(data_path)

            if not isinstance(data, h5py.Dataset):
                raise ValueError(
                    f'Expected diffraction pattern dataset at "{data_path}" in {file_path};'
                    f' got {type(data).__name__}.'
                )

            if data.ndim != 3:
                raise ValueError(
                    f'Diffraction pattern dataset at "{data_path}" must be 3D (N,H,W);'
                    f' got shape {data.shape}.'
                )

            num_patterns, detector_height, detector_width = data.shape
            detector_extent = ImageExtent(detector_width, detector_height)

            detector_distance_m: float | None = None
            distance = _read_scalar(h5_file, _P.DETECTOR_DISTANCE)
            if distance is not None:
                detector_distance_m = float(distance)

            detector_pixel_geometry: PixelGeometry | None = None
            x_pixel = _read_scalar(h5_file, _P.DETECTOR_X_PIXEL_SIZE)
            y_pixel = _read_scalar(h5_file, _P.DETECTOR_Y_PIXEL_SIZE)
            if x_pixel is not None and y_pixel is not None:
                detector_pixel_geometry = PixelGeometry(float(x_pixel), float(y_pixel))

            probe_energy_eV: float | None = None  # noqa: N806
            energy_J = _read_scalar(h5_file, _P.SOURCE_ENERGY)  # noqa: N806
            if energy_J is not None:
                probe_energy_eV = float(energy_J) / ELECTRON_VOLT_J  # noqa: N806

            probe_photon_count: int | None = None
            photon_count = _read_scalar(h5_file, _P.PT_PROBE_PHOTON_COUNT)
            if photon_count is not None:
                probe_photon_count = int(photon_count)

            exposure_time_s: float | None = None
            exposure = _read_scalar(h5_file, _P.PT_EXPOSURE_TIME)
            if exposure is not None:
                exposure_time_s = float(exposure)

            tomography_angle_deg: float | None = None
            angle = _read_scalar(h5_file, _P.PT_TOMOGRAPHY_ANGLE)
            if angle is not None:
                tomography_angle_deg = float(angle)

            bad_pixels: BadPixels | None = None
            if _P.DETECTOR_MASK in h5_file:
                mask = h5_file[_P.DETECTOR_MASK][()]
                bad_pixels = numpy.asarray(mask != 0, dtype=bool)

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data.dtype,
                detector_distance_m=detector_distance_m,
                detector_extent=detector_extent,
                detector_pixel_geometry=detector_pixel_geometry,
                probe_energy_eV=probe_energy_eV,
                probe_photon_count=probe_photon_count,
                exposure_time_s=exposure_time_s,
                tomography_angle_deg=tomography_angle_deg,
                file_path=file_path,
            )

        array = H5DiffractionPatternArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            file_path=file_path,
            data_path=data_path,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array], bad_pixels)


class CXIDiffractionFileWriter(DiffractionFileWriter):
    """Write a diffraction dataset to a CXI file."""

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        patterns = numpy.concatenate([array.get_patterns() for array in dataset])
        metadata = dataset.get_metadata()
        bad_pixels = dataset.get_bad_pixels()

        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset('cxi_version', data=CXI_VERSION)

            entry = h5_file.create_group(_P.ENTRY)
            entry.create_dataset('start_time', data=_iso_now())

            instrument = h5_file.create_group(_P.INSTRUMENT)
            source = instrument.create_group('source_1')
            detector = instrument.create_group('detector_1')

            if metadata.probe_energy_eV is not None:
                source.create_dataset('energy', data=metadata.probe_energy_eV * ELECTRON_VOLT_J)

            if metadata.detector_distance_m is not None:
                detector.create_dataset('distance', data=metadata.detector_distance_m)

            pixel_geometry = metadata.detector_pixel_geometry
            if pixel_geometry is not None:
                detector.create_dataset('x_pixel_size', data=pixel_geometry.width_m)
                detector.create_dataset('y_pixel_size', data=pixel_geometry.height_m)

            detector.create_dataset('data', data=patterns, compression='lzf')

            if bad_pixels is not None:
                mask = numpy.where(bad_pixels, CXI_PIXEL_IS_INVALID, 0).astype(numpy.uint32)
                detector.create_dataset('mask', data=mask, compression='lzf')

            data_group = h5_file.create_group(_P.DATA_GROUP)
            data_group['data'] = h5py.SoftLink(_P.DETECTOR_DATA)

            pt_extras: dict[str, Any] = {}
            if metadata.probe_photon_count is not None:
                pt_extras[_P.PT_PROBE_PHOTON_COUNT] = int(metadata.probe_photon_count)
            if metadata.exposure_time_s is not None:
                pt_extras[_P.PT_EXPOSURE_TIME] = float(metadata.exposure_time_s)
            if metadata.tomography_angle_deg is not None:
                pt_extras[_P.PT_TOMOGRAPHY_ANGLE] = float(metadata.tomography_angle_deg)

            if pt_extras:
                h5_file.create_group(_P.PTYCHODUS)
                for path, value in pt_extras.items():
                    h5_file.create_dataset(path, data=value)


class CXIProductFileIO(ProductFileReader, ProductFileWriter):
    """Read and write ptychodus data products in CXI format."""

    SIMPLE_NAME: Final[str] = 'CXI'
    DISPLAY_NAME: Final[str] = 'Coherent X-ray Imaging Files (*.cxi)'

    def read(self, file_path: Path) -> Product:
        with h5py.File(file_path, 'r') as h5_file:
            metadata = self._read_metadata(h5_file)
            probe = self._read_probe(h5_file)
            object_ = self._read_object(h5_file)
            positions = self._read_positions(h5_file)
            losses = self._read_losses(h5_file)

        return Product(
            metadata=metadata,
            probe_positions=positions,
            probes=probe,
            object_=object_,
            losses=losses,
        )

    def write(self, file_path: Path, product: Product) -> None:
        metadata = product.metadata
        probe = product.probes
        object_ = product.object_
        object_geometry = object_.get_geometry()
        probe_pixel_geometry = probe.get_pixel_geometry()

        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset('cxi_version', data=CXI_VERSION)

            entry = h5_file.create_group(_P.ENTRY)
            entry.create_dataset('start_time', data=_iso_now())

            if metadata.name:
                entry.create_dataset('title', data=metadata.name)
            if metadata.comments:
                entry.create_dataset('experiment_description', data=metadata.comments)

            sample = h5_file.create_group(_P.SAMPLE)
            if metadata.name:
                sample.create_dataset('name', data=metadata.name)

            positions_xyz = numpy.zeros((len(product.probe_positions), 3), dtype=numpy.float64)
            position_indexes = numpy.empty(len(product.probe_positions), dtype=numpy.int64)
            for i, point in enumerate(product.probe_positions):
                position_indexes[i] = point.index
                positions_xyz[i, 0] = point.x_m
                positions_xyz[i, 1] = point.y_m

            sample_geometry = sample.create_group('geometry_1')
            sample_geometry.create_dataset('translation', data=positions_xyz)

            data_group = h5_file.create_group(_P.DATA_GROUP)
            data_group['translation'] = h5py.SoftLink(_P.TRANSLATION)

            instrument = h5_file.create_group(_P.INSTRUMENT)
            source = instrument.create_group('source_1')
            source.create_dataset('energy', data=metadata.probe_energy_J)
            detector = instrument.create_group('detector_1')
            detector.create_dataset('distance', data=metadata.detector_distance_m)

            self._write_probe_image(h5_file, probe, probe_pixel_geometry)
            source['illumination'] = h5py.SoftLink(_P.PROBE_IMAGE_DATA)

            self._write_object_image(h5_file, object_, object_geometry)

            process = h5_file.create_group(_P.PROCESS)
            process.create_dataset('program', data='Ptychodus')
            process.create_dataset('version', data=ptychodus_version)
            process.create_dataset('date', data=_iso_now())

            self._write_ptychodus_extras(
                h5_file,
                metadata,
                probe,
                probe_pixel_geometry,
                object_,
                object_geometry,
                position_indexes,
                product.losses,
            )

    # -- read helpers ---------------------------------------------------------------

    def _read_metadata(self, h5_file: h5py.File) -> ProductMetadata:
        name = _read_string(h5_file, _P.TITLE) or _read_string(h5_file, _P.SAMPLE_NAME)
        comments = _read_string(h5_file, _P.EXPERIMENT_DESCRIPTION)

        detector_distance_m = 0.0
        distance = _read_scalar(h5_file, _P.DETECTOR_DISTANCE)
        if distance is not None:
            detector_distance_m = float(distance)

        probe_energy_eV = 0.0  # noqa: N806
        energy_J = _read_scalar(h5_file, _P.SOURCE_ENERGY)  # noqa: N806
        if energy_J is not None:
            probe_energy_eV = float(energy_J) / ELECTRON_VOLT_J  # noqa: N806

        probe_photon_count = 0.0
        photon_count = _read_scalar(h5_file, _P.PT_PROBE_PHOTON_COUNT)
        if photon_count is not None:
            probe_photon_count = float(photon_count)

        exposure_time_s = 0.0
        exposure = _read_scalar(h5_file, _P.PT_EXPOSURE_TIME)
        if exposure is not None:
            exposure_time_s = float(exposure)

        mass_attenuation_m2_kg = 0.0
        attenuation = _read_scalar(h5_file, _P.PT_MASS_ATTENUATION)
        if attenuation is not None:
            mass_attenuation_m2_kg = float(attenuation)

        tomography_angle_deg = 0.0
        angle = _read_scalar(h5_file, _P.PT_TOMOGRAPHY_ANGLE)
        if angle is not None:
            tomography_angle_deg = float(angle)

        return ProductMetadata(
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
        )

    def _read_probe(self, h5_file: h5py.File) -> ProbeSequence:
        if _P.PROBE_IMAGE_DATA in h5_file:
            probe_path = _P.PROBE_IMAGE_DATA
            image_group_path = _P.PROBE_IMAGE
        elif _P.ILLUMINATION in h5_file:
            probe_path = _P.ILLUMINATION
            image_group_path = ''
        else:
            raise ValueError(
                f'CXI file has no probe: expected {_P.PROBE_IMAGE_DATA} or {_P.ILLUMINATION}.'
            )

        array = h5_file[probe_path][()]

        pixel_geometry: PixelGeometry | None = None
        pw = _read_scalar(h5_file, _P.PT_PROBE_PIXEL_WIDTH)
        ph = _read_scalar(h5_file, _P.PT_PROBE_PIXEL_HEIGHT)
        if pw is not None and ph is not None:
            pixel_geometry = PixelGeometry(width_m=float(pw), height_m=float(ph))
        elif image_group_path:
            # Fall back to computing from image_size / array shape.
            image_size_path = f'{image_group_path}/image_size'
            if image_size_path in h5_file:
                image_size = numpy.asarray(h5_file[image_size_path][()], dtype=float)
                if image_size.size >= 2 and array.ndim >= 2:
                    height_px, width_px = array.shape[-2], array.shape[-1]
                    if height_px > 0 and width_px > 0:
                        pixel_geometry = PixelGeometry(
                            width_m=float(image_size[0]) / width_px,
                            height_m=float(image_size[1]) / height_px,
                        )

        opr_weights = None
        if _P.PT_OPR_WEIGHTS in h5_file:
            opr_weights = h5_file[_P.PT_OPR_WEIGHTS][()]

        return ProbeSequence(array=array, opr_weights=opr_weights, pixel_geometry=pixel_geometry)

    def _read_object(self, h5_file: h5py.File) -> Object:
        if _P.OBJECT_IMAGE_DATA not in h5_file:
            raise ValueError(f'CXI file has no object at {_P.OBJECT_IMAGE_DATA}.')

        array = h5_file[_P.OBJECT_IMAGE_DATA][()]

        pixel_geometry: PixelGeometry | None = None
        pw = _read_scalar(h5_file, _P.PT_OBJECT_PIXEL_WIDTH)
        ph = _read_scalar(h5_file, _P.PT_OBJECT_PIXEL_HEIGHT)
        if pw is not None and ph is not None:
            pixel_geometry = PixelGeometry(width_m=float(pw), height_m=float(ph))
        else:
            image_size_path = f'{_P.OBJECT_IMAGE}/image_size'
            if image_size_path in h5_file:
                image_size = numpy.asarray(h5_file[image_size_path][()], dtype=float)
                if image_size.size >= 2 and array.ndim >= 2:
                    height_px, width_px = array.shape[-2], array.shape[-1]
                    if height_px > 0 and width_px > 0:
                        pixel_geometry = PixelGeometry(
                            width_m=float(image_size[0]) / width_px,
                            height_m=float(image_size[1]) / height_px,
                        )

        center: ObjectCenter | None = None
        cx = _read_scalar(h5_file, _P.PT_OBJECT_CENTER_X)
        cy = _read_scalar(h5_file, _P.PT_OBJECT_CENTER_Y)
        if cx is not None and cy is not None:
            center = ObjectCenter(x_m=float(cx), y_m=float(cy))

        layer_spacing_m: list[float] = []
        if _P.PT_OBJECT_LAYER_SPACING in h5_file:
            spacing = h5_file[_P.PT_OBJECT_LAYER_SPACING][()]
            layer_spacing_m = [float(v) for v in numpy.atleast_1d(spacing)]

        return Object(
            array=array,
            pixel_geometry=pixel_geometry,
            center=center,
            layer_spacing_m=layer_spacing_m,
        )

    def _read_positions(self, h5_file: h5py.File) -> ProbePositionSequence:
        if _P.TRANSLATION in h5_file:
            translation_path = _P.TRANSLATION
        elif _P.DATA_TRANSLATION in h5_file:
            translation_path = _P.DATA_TRANSLATION
        else:
            logger.warning(
                'CXI file has no probe positions at %s or %s; returning empty sequence.',
                _P.TRANSLATION,
                _P.DATA_TRANSLATION,
            )
            return ProbePositionSequence()

        translation = numpy.asarray(h5_file[translation_path][()])
        if translation.ndim != 2 or translation.shape[1] < 2:
            raise ValueError(
                f'Expected translation dataset at "{translation_path}" to have shape (N, >=2);'
                f' got {translation.shape}.'
            )

        num_positions = translation.shape[0]

        if _P.PT_POSITION_INDEXES in h5_file:
            indexes = numpy.asarray(h5_file[_P.PT_POSITION_INDEXES][()], dtype=int)
        else:
            indexes = numpy.arange(num_positions, dtype=int)

        if len(indexes) != num_positions:
            raise ValueError(
                f'Position index count {len(indexes)} does not match'
                f' translation count {num_positions}.'
            )

        points = [
            ProbePosition(int(idx), float(translation[i, 0]), float(translation[i, 1]))
            for i, idx in enumerate(indexes)
        ]
        return ProbePositionSequence(points)

    def _read_losses(self, h5_file: h5py.File) -> list[LossValue]:
        if _P.PT_LOSS_VALUES not in h5_file:
            return []

        values = numpy.atleast_1d(h5_file[_P.PT_LOSS_VALUES][()])

        if _P.PT_LOSS_EPOCHS in h5_file:
            epochs = numpy.atleast_1d(h5_file[_P.PT_LOSS_EPOCHS][()])
        else:
            epochs = numpy.arange(len(values))

        return [LossValue(int(epoch), float(value)) for epoch, value in zip(epochs, values)]

    # -- write helpers --------------------------------------------------------------

    def _write_probe_image(
        self, h5_file: h5py.File, probe: ProbeSequence, pixel_geometry: PixelGeometry
    ) -> None:
        image = h5_file.create_group(_P.PROBE_IMAGE)
        data = image.create_dataset('data', data=probe.get_array())
        data.attrs['axes'] = 'coherent:incoherent:y:x'
        image.create_dataset('data_type', data='amplitude')
        image.create_dataset('data_space', data='real')
        image.create_dataset(
            'image_size',
            data=numpy.array(
                [
                    pixel_geometry.width_m * probe.width_px,
                    pixel_geometry.height_m * probe.height_px,
                    0.0,
                ],
                dtype=numpy.float64,
            ),
        )
        image['source_1'] = h5py.SoftLink(_P.SOURCE)
        image['detector_1'] = h5py.SoftLink(_P.DETECTOR)

    def _write_object_image(self, h5_file: h5py.File, object_: Object, geometry: Any) -> None:
        image = h5_file.create_group(_P.OBJECT_IMAGE)
        data = image.create_dataset('data', data=object_.get_array())
        data.attrs['axes'] = 'layer:y:x'
        image.create_dataset('data_type', data='electron density')
        image.create_dataset('data_space', data='real')
        image.create_dataset(
            'image_size',
            data=numpy.array(
                [
                    geometry.pixel_width_m * geometry.width_px,
                    geometry.pixel_height_m * geometry.height_px,
                    0.0,
                ],
                dtype=numpy.float64,
            ),
        )
        image['source_1'] = h5py.SoftLink(_P.SOURCE)
        image['detector_1'] = h5py.SoftLink(_P.DETECTOR)

    def _write_ptychodus_extras(
        self,
        h5_file: h5py.File,
        metadata: ProductMetadata,
        probe: ProbeSequence,
        probe_pixel_geometry: PixelGeometry,
        object_: Object,
        object_geometry: Any,
        position_indexes: numpy.ndarray,
        losses: Any,
    ) -> None:
        h5_file.create_group(_P.PTYCHODUS)

        h5_file.create_dataset(_P.PT_POSITION_INDEXES, data=position_indexes)
        h5_file.create_dataset(_P.PT_PROBE_PIXEL_WIDTH, data=probe_pixel_geometry.width_m)
        h5_file.create_dataset(_P.PT_PROBE_PIXEL_HEIGHT, data=probe_pixel_geometry.height_m)

        try:
            opr_weights = probe.get_opr_weights()
        except ValueError:
            pass
        else:
            h5_file.create_dataset(_P.PT_OPR_WEIGHTS, data=opr_weights)

        h5_file.create_dataset(_P.PT_OBJECT_PIXEL_WIDTH, data=object_geometry.pixel_width_m)
        h5_file.create_dataset(_P.PT_OBJECT_PIXEL_HEIGHT, data=object_geometry.pixel_height_m)
        h5_file.create_dataset(_P.PT_OBJECT_CENTER_X, data=object_geometry.center_x_m)
        h5_file.create_dataset(_P.PT_OBJECT_CENTER_Y, data=object_geometry.center_y_m)

        layer_spacing = list(object_.layer_spacing_m)
        if layer_spacing:
            h5_file.create_dataset(
                _P.PT_OBJECT_LAYER_SPACING, data=numpy.asarray(layer_spacing, dtype=numpy.float64)
            )

        if metadata.probe_photon_count:
            h5_file.create_dataset(_P.PT_PROBE_PHOTON_COUNT, data=metadata.probe_photon_count)
        if metadata.exposure_time_s:
            h5_file.create_dataset(_P.PT_EXPOSURE_TIME, data=metadata.exposure_time_s)
        if metadata.mass_attenuation_m2_kg:
            h5_file.create_dataset(_P.PT_MASS_ATTENUATION, data=metadata.mass_attenuation_m2_kg)
        if metadata.tomography_angle_deg:
            h5_file.create_dataset(_P.PT_TOMOGRAPHY_ANGLE, data=metadata.tomography_angle_deg)

        loss_epochs = [loss.epoch for loss in losses]
        loss_values = [loss.value for loss in losses]
        if loss_epochs:
            h5_file.create_dataset(
                _P.PT_LOSS_EPOCHS, data=numpy.asarray(loss_epochs, dtype=numpy.int64)
            )
            h5_file.create_dataset(
                _P.PT_LOSS_VALUES, data=numpy.asarray(loss_values, dtype=numpy.float64)
            )


def register_plugins(registry: PluginRegistry) -> None:
    display_name: Final[str] = CXIProductFileIO.DISPLAY_NAME
    simple_name: Final[str] = CXIProductFileIO.SIMPLE_NAME

    registry.diffraction_file_readers.register_plugin(
        CXIDiffractionFileReader(),
        simple_name=simple_name,
        display_name=display_name,
    )
    registry.diffraction_file_writers.register_plugin(
        CXIDiffractionFileWriter(),
        simple_name=simple_name,
        display_name=display_name,
    )

    cxi_product_io = CXIProductFileIO()
    registry.register_product_file_reader_with_adapters(
        cxi_product_io,
        simple_name=simple_name,
        display_name=display_name,
    )
    registry.product_file_writers.register_plugin(
        cxi_product_io,
        simple_name=simple_name,
        display_name=display_name,
    )
