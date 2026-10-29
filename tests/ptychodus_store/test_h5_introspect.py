from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ptychodus_store.storage.h5_introspect import (
    IntrospectionError,
    introspect_diffraction,
    introspect_fluorescence,
    introspect_product,
)


def test_introspect_diffraction(tmp_path: Path) -> None:
    path = tmp_path / 'diffraction.h5'
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('patterns', data=np.zeros((7, 64, 128), dtype=np.uint16))
        ds.attrs['detector_pixel_width_m'] = 1.5e-5
        ds.attrs['detector_pixel_height_m'] = 1.5e-5
        f.create_dataset('indexes', data=np.arange(7))
        f.create_dataset('bad_pixels', data=np.zeros((64, 128), dtype=bool))

    result = introspect_diffraction(path)
    assert result['num_patterns_total'] == 7
    assert result['pattern_shape'] == (64, 128)
    assert result['pattern_dtype'] == 'uint16'
    assert result['detector_pixel_width_m'] == pytest.approx(1.5e-5)


def test_introspect_diffraction_missing_dataset(tmp_path: Path) -> None:
    path = tmp_path / 'bad.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('something_else', data=np.zeros(1))
    with pytest.raises(IntrospectionError):
        introspect_diffraction(path)


def test_introspect_product(tmp_path: Path) -> None:
    path = tmp_path / 'product.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['name'] = 'p1'
        f.attrs['comments'] = 'hello'
        f.attrs['detector_object_distance_m'] = 2.0
        f.attrs['photon_energy_eV'] = 8500.0
        f.attrs['probe_photon_count'] = 12345
        f.attrs['exposure_time_s'] = 0.05
        f.attrs['mass_attenuation_m2_per_kg'] = 0.0
        f.attrs['tomography_angle_deg'] = 30.0
        f.attrs['tilt_angle_deg'] = 12.5
        f.attrs['polarization'] = 'left_circular'
        obj = f.create_dataset('object', data=np.zeros((2, 32, 48), dtype=np.complex64))
        obj.attrs['pixel_width_m'] = 1e-9
        obj.attrs['pixel_height_m'] = 1e-9
        probe = f.create_dataset('probe', data=np.zeros((3, 16, 16), dtype=np.complex64))
        probe.attrs['pixel_width_m'] = 1e-9
        probe.attrs['pixel_height_m'] = 1e-9
        f.create_dataset('probe_position_indexes', data=np.arange(11))
        f.create_dataset('loss_epochs', data=np.array([1, 2]))

    result = introspect_product(path)
    assert result['name'] == 'p1'
    assert result['comments'] == 'hello'
    assert result['photon_energy_eV'] == 8500.0
    assert result['object_shape'] == (2, 32, 48)
    assert result['probe_shape'] == (3, 16, 16)
    assert result['num_scan_points'] == 11
    assert result['num_loss_epochs'] == 2
    assert result['tomography_angle_deg'] == 30.0
    assert result['tilt_angle_deg'] == 12.5
    assert result['polarization'] == 'left_circular'


def test_introspect_fluorescence(tmp_path: Path) -> None:
    path = tmp_path / 'fluorescence.h5'
    with h5py.File(path, 'w') as f:
        group = f.require_group('/MAPS/XRF_Analyzed/NNLS')
        group.create_dataset('Counts_Per_Sec', data=np.zeros((2, 6, 9), dtype=np.float32))
        group.create_dataset('Channel_Names', data=np.array(['Fe', 'Cu'], dtype='S8'))

    result = introspect_fluorescence(path)
    assert result['element_names'] == ['Fe', 'Cu']
    assert result['map_shape'] == (6, 9)


def test_introspect_product_mirrors_the_desktop_property_table(tmp_path: Path) -> None:
    """The fields the desktop shows that need more than a root attribute read."""
    path = tmp_path / 'product.h5'
    modes = np.stack([np.full((8, 8), 2.0), np.full((8, 8), 1.0)]).astype(np.complex64)
    with h5py.File(path, 'w') as f:
        f.attrs['probe_photon_count'] = 1.5e9
        f.attrs['i0_photons_per_count'] = 4.25
        # (coherent, incoherent, h, w): the second coherent mode must not count.
        probe = f.create_dataset('probe', data=np.stack([modes, modes]))
        probe.attrs['pixel_width_m'] = 7e-9
        probe.attrs['pixel_height_m'] = 9e-9

    result = introspect_product(path)

    # Read as a float: the API types the count as one, and an int cast would truncate.
    assert result['probe_photon_count'] == pytest.approx(1.5e9)
    assert result['i0_photons_per_count'] == pytest.approx(4.25)
    # The probe's own pitch, which product.h5 keeps apart from the object's.
    assert result['probe_pixel_width_m'] == pytest.approx(7e-9)
    assert result['probe_pixel_height_m'] == pytest.approx(9e-9)
    # The desktop lists the incoherent modes of the first coherent mode.
    assert result['probe_shape'] == (2, 8, 8)
    # Mode powers 4:1, so the shares are 0.8 and 0.2 and the coherence their root
    # sum of squares.
    assert result['probe_mode_relative_power'] == pytest.approx([0.8, 0.2])
    assert result['probe_coherence'] == pytest.approx(np.sqrt(0.8**2 + 0.2**2))


def test_introspect_product_without_i0_calibration(tmp_path: Path) -> None:
    """Files written before the I0 calibration existed simply lack it."""
    path = tmp_path / 'product.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['name'] = 'old'

    result = introspect_product(path)

    assert result['i0_photons_per_count'] is None
    assert result['probe_coherence'] is None


def test_introspect_fluorescence_element_sizes(tmp_path: Path) -> None:
    path = tmp_path / 'fluorescence.h5'
    with h5py.File(path, 'w') as f:
        group = f.require_group('/MAPS/XRF_Analyzed/NNLS')
        group.create_dataset('Counts_Per_Sec', data=np.zeros((2, 6, 9), dtype=np.float32))
        group.create_dataset('Channel_Names', data=np.array(['Fe', 'Cu'], dtype='S8'))

    result = introspect_fluorescence(path)

    assert result['element_nbytes'] == [6 * 9 * 4, 6 * 9 * 4]


def test_introspect_product_reads_the_legacy_energy_attribute(tmp_path: Path) -> None:
    """Products written before the rename spell it probe_energy_eV; the desktop reads both."""
    path = tmp_path / 'product.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['probe_energy_eV'] = 6722.0

    assert introspect_product(path)['photon_energy_eV'] == pytest.approx(6722.0)
