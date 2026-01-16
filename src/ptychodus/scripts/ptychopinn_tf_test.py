#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import logging
import shutil
import sys
from pathlib import Path

import numpy

from ptychodus.model import ModelCore

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path('../PtychoPINN/datasets/Run1084_recon3_postPC_shrunk_3.npz')
REQUIRED_DATASET_KEYS = {
    'diffraction',
    'xcoords_start',
    'ycoords_start',
    'probeGuess',
    'objectGuess',
}
REQUIRED_TRAIN_KEYS = {
    'xcoords',
    'ycoords',
    'diff3d',
    'probeGuess',
    'objectGuess',
}


def _require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f'Missing required module "{module_name}".')


def _validate_npz_keys(file_path: Path, required_keys: set[str], label: str) -> None:
    with numpy.load(file_path) as npz_file:
        keys = set(npz_file.files)

    missing = sorted(required_keys - keys)
    if missing:
        missing_keys = ', '.join(missing)
        raise RuntimeError(f'{label} missing keys: {missing_keys}')


def _configure_reconstructor(model: ModelCore, reconstructor_name: str) -> None:
    available = sorted(model.reconstructor_core.presenter.reconstructors())
    if reconstructor_name not in available:
        options = ', '.join(available)
        raise RuntimeError(
            f'Reconstructor "{reconstructor_name}" not available. Options: {options}'
        )

    model.workflow_api.set_reconstructor(reconstructor_name)

    active = model.reconstructor_core.presenter.get_reconstructor()
    if active != reconstructor_name:
        raise RuntimeError(f'Failed to set reconstructor "{reconstructor_name}". Active: {active}')


def _configure_metadata(model: ModelCore, product_index: int, args: argparse.Namespace) -> None:
    product_item = model.product_core.product_api.get_item(product_index)
    metadata = product_item.get_metadata_item()

    metadata.detector_distance_m.set_value(args.detector_distance_m)
    metadata.probe_energy_eV.set_value(args.probe_energy_ev)
    metadata.exposure_time_s.set_value(args.exposure_time_s)

    if metadata.probe_photon_count.get_value() <= 0:
        photon_count = model.diffraction_core.dataset.get_maximum_pattern_counts()
        metadata.probe_photon_count.set_value(photon_count)

    product_item.get_object_item().rebuild()
    probe_item = product_item.get_probe_item()
    probe_item.set_builder(probe_item.get_builder().copy())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='ptychodus-ptychopinn-tf-test',
        description='Run the PtychoPINN TF train/load/infer workflow via ptychodus.',
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=DEFAULT_DATASET,
        help='Path to the SLAC NPZ dataset.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to write training data and reconstruction outputs.',
    )
    parser.add_argument(
        '--settings',
        type=Path,
        help='Optional settings file for ptychodus.',
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.INFO,
        help='Python logging level.',
    )
    parser.add_argument(
        '--detector-distance-m',
        type=float,
        default=0.75,
        help='Detector distance (meters).',
    )
    parser.add_argument(
        '--probe-energy-ev',
        type=float,
        default=8000.0,
        help='Probe energy (eV).',
    )
    parser.add_argument(
        '--exposure-time-s',
        type=float,
        default=0.1,
        help='Exposure time (seconds).',
    )
    parser.add_argument(
        '--nepochs',
        type=int,
        default=50,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size.',
    )
    parser.add_argument(
        '--gridsize',
        type=int,
        default=1,
        help='PtychoPINN gridsize setting.',
    )
    parser.add_argument(
        '--test-samples',
        type=int,
        default=512,
        help='Number of inference samples to use.',
    )
    parser.add_argument(
        '--reconstructor',
        default='PtychoPINN/PINN',
        help='Reconstructor name to use.',
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
        encoding='utf-8',
        level=args.log_level,
    )

    try:
        _require_module('ptycho')
    except RuntimeError as exc:
        logger.error(exc)
        return 1

    dataset_path = args.dataset
    if not dataset_path.is_file():
        logger.error(f'Dataset not found: {dataset_path}')
        return 1

    try:
        _validate_npz_keys(dataset_path, REQUIRED_DATASET_KEYS, 'dataset')
    except RuntimeError as exc:
        logger.error(exc)
        return 1

    output_dir = args.output_dir
    train_dir = output_dir / 'train'
    model_out_dir = output_dir / 'model_out'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ModelCore(args.settings, log_level=args.log_level) as model:
            _configure_reconstructor(model, args.reconstructor)

            model.workflow_api.open_patterns(
                dataset_path,
                file_type='SLAC_NPZ',
                process_patterns=True,
                block=True,
            )

            product_api = model.workflow_api.open_product(
                dataset_path,
                file_type='SLAC_NPZ',
            )
            _configure_metadata(model, product_api.get_product_index(), args)

            model_settings = model.ptychopinn_reconstructor_library.model_settings
            model_settings.gridsize.set_value(args.gridsize)

            train_data_path = train_dir / 'train_data.npz'
            product_api.export_training_data(train_data_path)
            test_data_path = train_dir / 'test_data.npz'
            shutil.copyfile(train_data_path, test_data_path)

            _validate_npz_keys(train_data_path, REQUIRED_TRAIN_KEYS, 'train_data.npz')
            _validate_npz_keys(test_data_path, REQUIRED_TRAIN_KEYS, 'test_data.npz')

            training_settings = model.ptychopinn_reconstructor_library.training_settings
            training_settings.nepochs.set_value(args.nepochs)
            training_settings.batch_size.set_value(args.batch_size)
            training_settings.data_dir.set_value(train_dir)
            training_settings.output_dir.set_value(model_out_dir)

            inference_settings = model.ptychopinn_reconstructor_library.inference_settings
            inference_settings.n_samples.set_value(args.test_samples)

            model.workflow_api.train_reconstructor(train_dir, model_out_dir)

            model_file = model_out_dir / 'wts.h5.zip'
            if not model_file.is_file():
                logger.error(f'Model file not found: {model_file}')
                return 1

            output_product_api = product_api.reconstruct_local(block=True)
            model.run_tasks()

            recon_product_path = output_dir / 'recon_product.h5'
            output_product_api.save_product(recon_product_path, file_type='HDF5')

            output_item = model.product_core.product_api.get_item(
                output_product_api.get_product_index()
            )
            object_array = output_item.get_object_item().get_object().get_layer(0)

            if object_array.size == 0:
                logger.error('Reconstruction produced an empty object array.')
                return 1

            numpy.save(output_dir / 'recon_object.npy', object_array)
    except Exception:
        logger.exception('PtychoPINN TF workflow failed.')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
