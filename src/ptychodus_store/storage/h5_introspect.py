"""Read HDF5 attrs / dataset shapes for the DB cache without loading array data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from ptychodus.api.io import DiffractionFileKeys, ProductFileKeys


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

            return {
                'pattern_dtype': pattern_dtype,
                'pattern_shape': pattern_shape,
                'num_patterns_total': num_patterns_total,
                'detector_pixel_width_m': pixel_width,
                'detector_pixel_height_m': pixel_height,
            }
    except (OSError, KeyError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc


def introspect_product(path: Path) -> dict[str, Any]:
    """Read root-level attrs and probe/object dataset shapes from product.h5.

    Returns a dict of HDF5-derived fields:
      * name, comments
      * detector_distance_m, probe_energy_eV, probe_photon_count, exposure_time_s,
        mass_attenuation_m2_kg, tomography_angle_deg
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

            return {
                'name': name,
                'comments': comments,
                'detector_distance_m': _root_attr(ProductFileKeys.DETECTOR_OBJECT_DISTANCE, float),
                'probe_energy_eV': _root_attr(ProductFileKeys.PROBE_ENERGY, float),
                'probe_photon_count': _root_attr(ProductFileKeys.PROBE_PHOTON_COUNT, int),
                'exposure_time_s': _root_attr(ProductFileKeys.EXPOSURE_TIME, float),
                'mass_attenuation_m2_kg': _root_attr(ProductFileKeys.MASS_ATTENUATION, float),
                'tomography_angle_deg': _root_attr(ProductFileKeys.TOMOGRAPHY_ANGLE, float),
                'object_shape': object_shape,
                'object_pixel_width_m': object_pixel_width_m,
                'object_pixel_height_m': object_pixel_height_m,
                'probe_shape': probe_shape,
                'num_scan_points': num_scan_points,
                'num_loss_epochs': num_loss_epochs,
            }
    except (OSError, KeyError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc


FLUORESCENCE_ELEMENT_NAMES_KEY = 'element_names'
FLUORESCENCE_ELEMENT_MAPS_KEY = 'element_maps'


def introspect_fluorescence(path: Path) -> dict[str, Any]:
    """Read element names and map shape from fluorescence.h5.

    The service convention is a flat layout:
      * `element_names` — 1D string dataset, length N_elements
      * `element_maps`  — 3D float dataset, shape (N_elements, H, W)

    Returns a dict with `element_names: list[str]` and `map_shape: tuple[int, int] | None`.
    """
    try:
        with h5py.File(path, 'r') as f:
            names_ds = f.get(FLUORESCENCE_ELEMENT_NAMES_KEY)
            maps_ds = f.get(FLUORESCENCE_ELEMENT_MAPS_KEY)

            element_names: list[str] = []
            if isinstance(names_ds, h5py.Dataset):
                raw = names_ds[()]
                for entry in raw:
                    if isinstance(entry, bytes):
                        element_names.append(entry.decode('utf-8', errors='replace'))
                    else:
                        element_names.append(str(entry))

            map_shape: tuple[int, int] | None = None
            if isinstance(maps_ds, h5py.Dataset) and len(maps_ds.shape) == 3:
                map_shape = (int(maps_ds.shape[1]), int(maps_ds.shape[2]))

            return {
                'element_names': element_names,
                'map_shape': map_shape,
            }
    except (OSError, KeyError) as exc:
        raise IntrospectionError(f'{path}: {exc}') from exc
