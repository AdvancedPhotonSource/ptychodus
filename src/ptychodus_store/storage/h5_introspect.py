"""Read HDF5 attrs / dataset shapes for the DB cache without loading array data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import DiffractionFileKeys, ProductFileKeys, load_fluorescence_data
from ptychodus.api.probe import Probe


class IntrospectionError(Exception):
    """Raised when an HDF5 file cannot be read or lacks expected structure."""


def _attr(group: h5py.HLObject, key: str, *, cast: type) -> Any:
    raw = group.attrs.get(key)
    if raw is None:
        return None
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return None


def _dataset_nbytes(item: Any) -> int:
    """Byte size of an HDF5 dataset from its shape and dtype, without reading it."""
    if not isinstance(item, h5py.Dataset):
        return 0

    return int(numpy.prod(item.shape)) * int(item.dtype.itemsize)


def introspect_diffraction(path: Path) -> dict[str, Any]:
    """Read scalar attrs and dataset shape/dtype from a diffraction.h5 file.

    Returns a dict of HDF5-derived fields:
      * pattern_dtype:        str
      * pattern_shape:        tuple[int, int] | None — (height, width)
      * num_patterns_total:   int | None
      * detector_pixel_width_m, detector_pixel_height_m: float | None
    """
    try:
        with h5py.File(path, 'r') as f:
            patterns = f.get(DiffractionFileKeys.PATTERNS)
            if not isinstance(patterns, h5py.Dataset):
                raise IntrospectionError(
                    f'{path}: missing {DiffractionFileKeys.PATTERNS!r} dataset'
                )

            shape = tuple(int(x) for x in patterns.shape)
            num_patterns_total = shape[0] if len(shape) >= 1 else None
            pattern_shape = (shape[1], shape[2]) if len(shape) == 3 else None
            pattern_dtype = str(patterns.dtype)

            pixel_width = _attr(patterns, DiffractionFileKeys.DETECTOR_PIXEL_WIDTH, cast=float)
            pixel_height = _attr(patterns, DiffractionFileKeys.DETECTOR_PIXEL_HEIGHT, cast=float)

            bad_pixels = f.get(DiffractionFileKeys.BAD_PIXELS)
            num_bad_pixels: int | None = None
            if isinstance(bad_pixels, h5py.Dataset):
                # A 2-D mask, so counting it costs one small read rather than a
                # pass over the pattern stack.
                num_bad_pixels = int(numpy.count_nonzero(bad_pixels[()]))

            return {
                'pattern_dtype': pattern_dtype,
                'pattern_shape': pattern_shape,
                'num_patterns_total': num_patterns_total,
                'detector_pixel_width_m': pixel_width,
                'detector_pixel_height_m': pixel_height,
                'num_bad_pixels': num_bad_pixels,
                'nbytes': _dataset_nbytes(patterns)
                + _dataset_nbytes(f.get(DiffractionFileKeys.INDEXES))
                + _dataset_nbytes(bad_pixels),
            }
    except (OSError, KeyError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc


def _probe_mode_relative_power(probe: Any) -> list[float]:
    """Fraction of the probe's power in each incoherent mode.

    The one place introspection reads array data rather than shapes: the powers are a
    reduction over the probe, which is megabytes rather than the gigabytes a pattern
    stack would be. :class:`Probe` owns the normalisation, so this does not restate it.
    """
    if not isinstance(probe, h5py.Dataset):
        return []

    array = probe[()]

    if array.ndim == 4:
        # (coherent, incoherent, h, w): the relative powers belong to the coherent
        # mode the viewer shows, which is the first.
        array = array[0]

    if array.ndim != 3:
        return []

    # The pixel geometry plays no part in a power ratio; Probe merely requires one.
    modes = Probe(array=array, pixel_geometry=PixelGeometry(width_m=1.0, height_m=1.0))
    return [
        float(modes.get_incoherent_mode_relative_power(i))
        for i in range(modes.num_incoherent_modes)
    ]


def _scan_path_length_m(h5_file: h5py.File) -> float | None:
    """Total length of the scan path, summed over consecutive positions."""
    h5_x = h5_file.get(ProductFileKeys.PROBE_POSITION_X)
    h5_y = h5_file.get(ProductFileKeys.PROBE_POSITION_Y)

    if not isinstance(h5_x, h5py.Dataset) or not isinstance(h5_y, h5py.Dataset):
        return None

    x_m = numpy.asarray(h5_x[()], dtype=float)
    y_m = numpy.asarray(h5_y[()], dtype=float)

    if x_m.size < 2 or x_m.shape != y_m.shape:
        return 0.0

    return float(numpy.sum(numpy.hypot(numpy.diff(x_m), numpy.diff(y_m))))


def introspect_product(path: Path) -> dict[str, Any]:
    """Read root-level attrs and probe/object dataset shapes from product.h5.

    Returns a dict of HDF5-derived fields:
      * name, comments
      * detector_distance_m, focus_object_distance_m, photon_energy_eV,
        probe_photon_count, exposure_time_s, mass_attenuation_m2_per_kg,
        tomography_angle_deg, tilt_angle_deg, polarization, far_field
      * object_shape:        tuple[int, int, int] | None — (layers, h, w)
      * object_pixel_width_m, object_pixel_height_m: float | None
      * probe_shape:         tuple[int, int, int] | None — (modes, h, w)
      * num_scan_points:     int | None
      * num_loss_epochs:     int
    """
    try:
        with h5py.File(path, 'r') as f:
            name = str(f.attrs.get(ProductFileKeys.NAME, ''))
            comments = str(f.attrs.get(ProductFileKeys.COMMENTS, ''))

            def _root_attr(key: str, cast: type) -> Any:
                raw = f.attrs.get(key)
                if raw is None:
                    return None
                try:
                    return cast(raw)
                except (TypeError, ValueError):
                    return None

            def _root_str_attr(key: str) -> str | None:
                raw = f.attrs.get(key)
                if raw is None:
                    return None
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8', errors='replace')
                text = str(raw)
                return text or None

            obj = f.get(ProductFileKeys.OBJECT_ARRAY)
            probe = f.get(ProductFileKeys.PROBE_ARRAY)
            positions = f.get(ProductFileKeys.PROBE_POSITION_INDEXES)
            loss_epochs = f.get(ProductFileKeys.LOSS_EPOCHS)

            object_shape: tuple[int, int, int] | None = None
            object_pixel_width_m: float | None = None
            object_pixel_height_m: float | None = None
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                object_shape = (int(obj.shape[0]), int(obj.shape[1]), int(obj.shape[2]))
                object_pixel_width_m = _attr(obj, ProductFileKeys.OBJECT_PIXEL_WIDTH, cast=float)
                object_pixel_height_m = _attr(obj, ProductFileKeys.OBJECT_PIXEL_HEIGHT, cast=float)

            probe_shape: tuple[int, int, int] | None = None
            if isinstance(probe, h5py.Dataset):
                shape = tuple(int(x) for x in probe.shape)
                if len(shape) == 3:
                    probe_shape = (shape[0], shape[1], shape[2])
                elif len(shape) == 4:
                    # (coherent, incoherent, h, w) — collapse coherent x incoherent into modes
                    probe_shape = (shape[0] * shape[1], shape[2], shape[3])

            num_scan_points: int | None = None
            if isinstance(positions, h5py.Dataset):
                num_scan_points = int(positions.shape[0])

            num_loss_epochs = 0
            if isinstance(loss_epochs, h5py.Dataset):
                num_loss_epochs = int(loss_epochs.shape[0])

            layer_spacing_m: list[float] = []
            h5_layer_spacing = f.get(ProductFileKeys.OBJECT_LAYER_SPACING)
            if isinstance(h5_layer_spacing, h5py.Dataset):
                layer_spacing_m = [float(v) for v in h5_layer_spacing[()]]

            return {
                'name': name,
                'comments': comments,
                'probe_mode_relative_power': _probe_mode_relative_power(probe),
                'probe_dtype': str(probe.dtype) if isinstance(probe, h5py.Dataset) else None,
                'probe_nbytes': _dataset_nbytes(probe),
                'object_dtype': str(obj.dtype) if isinstance(obj, h5py.Dataset) else None,
                'object_nbytes': _dataset_nbytes(obj),
                'object_layer_spacing_m': layer_spacing_m,
                'scan_length_m': _scan_path_length_m(f),
                'scan_nbytes': _dataset_nbytes(positions)
                + _dataset_nbytes(f.get(ProductFileKeys.PROBE_POSITION_X))
                + _dataset_nbytes(f.get(ProductFileKeys.PROBE_POSITION_Y)),
                'detector_distance_m': _root_attr(ProductFileKeys.DETECTOR_OBJECT_DISTANCE, float),
                'focus_object_distance_m': _root_attr(ProductFileKeys.FOCUS_OBJECT_DISTANCE, float),
                'far_field': _root_attr(ProductFileKeys.FAR_FIELD, bool),
                'photon_energy_eV': _root_attr(ProductFileKeys.PHOTON_ENERGY, float),
                'probe_photon_count': _root_attr(ProductFileKeys.PROBE_PHOTON_COUNT, int),
                'exposure_time_s': _root_attr(ProductFileKeys.EXPOSURE_TIME, float),
                'mass_attenuation_m2_per_kg': _root_attr(ProductFileKeys.MASS_ATTENUATION, float),
                'tomography_angle_deg': _root_attr(ProductFileKeys.TOMOGRAPHY_ANGLE, float),
                'tilt_angle_deg': _root_attr(ProductFileKeys.TILT_ANGLE, float),
                'polarization': _root_str_attr(ProductFileKeys.POLARIZATION),
                'object_shape': object_shape,
                'object_pixel_width_m': object_pixel_width_m,
                'object_pixel_height_m': object_pixel_height_m,
                'probe_shape': probe_shape,
                'num_scan_points': num_scan_points,
                'num_loss_epochs': num_loss_epochs,
            }
    except (OSError, KeyError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc


def introspect_fluorescence(path: Path) -> dict[str, Any]:
    """Read element names and map shape from a fluorescence.h5 file (XRF-Maps layout).

    Returns a dict with `element_names: list[str]` and `map_shape: tuple[int, int] | None`.
    Delegates to :func:`ptychodus.api.io.load_fluorescence_data`, which recognises the
    v10 NNLS/Fitted and legacy v9 layouts.
    """
    try:
        dataset = load_fluorescence_data(path)
    except (OSError, KeyError, ValueError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc

    element_names = [emap.name for emap in dataset]
    map_shape: tuple[int, int] | None = None
    if len(dataset):
        cps = dataset[0].counts_per_second
        if cps.ndim == 2:
            map_shape = (int(cps.shape[0]), int(cps.shape[1]))

    # Element maps are object-sized rather than detector-sized, so summing them is
    # cheap -- unlike the diffraction counts, which is why that column is absent.
    element_counts = [float(numpy.sum(emap.counts_per_second)) for emap in dataset]

    return {
        'element_names': element_names,
        'map_shape': map_shape,
        'element_counts': element_counts,
        'nbytes': int(sum(emap.counts_per_second.nbytes for emap in dataset)),
    }
