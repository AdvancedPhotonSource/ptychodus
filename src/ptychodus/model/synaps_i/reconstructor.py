from __future__ import annotations
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final
import logging

import numpy
import yaml

from ptychodus.api.object import Object
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    LossValue,
    ReconstructInput,
    ReconstructOutput,
    TrainOutput,
    TrainableReconstructor,
)

from ..analysis import BarycentricArrayStitcher
from .model import PtychoViT
from .settings import SynapsIInferenceSettings

logger = logging.getLogger(__name__)


def _load_yaml(file_path: Path) -> dict[str, Any]:
    with file_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError('SYNAPS-I config must be a YAML dictionary.')
    return data


def _resolve_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    model_section = config.get('model', {})
    if not isinstance(model_section, dict):
        raise ValueError('Invalid SYNAPS-I model configuration.')

    model_type = model_section.get('model_type')
    if model_type is None:
        model_config: dict[str, Any] = dict(model_section)
    else:
        model_type_str = str(model_type).lower()
        if model_type_str in {'vit', 'vit256'}:
            model_config = dict(model_section.get('vit256') or model_section.get('vit') or {})
        else:
            raise ValueError(f'Unsupported SYNAPS-I model type: {model_type}')

    if 'encoder_type' not in model_config:
        model_config = {**model_config, 'encoder_type': 'custom'}

    encoder_config = model_config.get('encoder', {})
    img_size = None
    if isinstance(encoder_config, dict):
        img_size = encoder_config.get('img_size')

    return model_config, int(img_size) if img_size is not None else None


def _pad_probe(probe: numpy.ndarray, target_modes: int) -> numpy.ndarray:
    if probe.ndim == 2:
        probe = probe[numpy.newaxis, numpy.newaxis, ...]
    elif probe.ndim == 3:
        probe = probe[numpy.newaxis, ...]
    elif probe.ndim != 4:
        raise ValueError(f'Unsupported probe shape: {probe.shape}')

    current_modes = probe.shape[1]
    if current_modes >= target_modes:
        return probe

    pad_shape = (probe.shape[0], target_modes - current_modes, probe.shape[2], probe.shape[3])
    padding = numpy.zeros(pad_shape, dtype=probe.dtype)
    return numpy.concatenate([probe, padding], axis=1)


class SynapsITrainableReconstructor(TrainableReconstructor):
    MODEL_FILE_FILTER: Final[str] = 'PyTorch Model (*.pth)'
    CONFIG_FILE_FILTER: Final[str] = 'YAML Files (*.yaml *.yml)'

    def __init__(
        self,
        inference_settings: SynapsIInferenceSettings,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._inference_settings = inference_settings
        self._model: PtychoViT | None = None
        self._model_config: dict[str, Any] | None = None
        self._model_size: int | None = None
        self._is_developer_mode_enabled = is_developer_mode_enabled

        try:
            import torch
        except ModuleNotFoundError:
            logger.info('PyTorch not found for SYNAPS-I.')
        else:
            logger.info(f'\tTorch {torch.__version__}')

    def get_name(self) -> str:
        return 'SYNAPS-I'

    def get_progress_goal(self) -> int:
        return 0

    def _select_device(self) -> str:
        import torch

        use_cuda = self._inference_settings.use_cuda.get_value()
        if use_cuda and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _ensure_model_loaded(self) -> PtychoViT:
        if self._model is None:
            raise RuntimeError('SYNAPS-I model not loaded.')
        return self._model

    def _resolve_config_path(self, model_path: Path | None) -> Path:
        config_path = self._inference_settings.config_path.get_value()
        if config_path.exists():
            return config_path

        if model_path is None:
            raise FileNotFoundError('SYNAPS-I config path not provided.')

        candidate_paths = sorted(
            model_path.parent.glob('config*.yml')
        ) + sorted(model_path.parent.glob('config*.yaml'))
        if candidate_paths:
            return candidate_paths[0]

        raise FileNotFoundError('Unable to locate SYNAPS-I config YAML near the model checkpoint.')

    def _load_model(self, model_path: Path, config_path: Path) -> None:
        import torch

        config = _load_yaml(config_path)
        data_section = config.get('data', {})
        if isinstance(data_section, dict):
            scale = data_section.get('scale')
            if scale is not None:
                self._inference_settings.scale.set_value(float(scale))
            max_probe_modes = data_section.get('max_probe_modes')
            if max_probe_modes is not None:
                self._inference_settings.max_probe_modes.set_value(int(max_probe_modes))
        model_config, model_size = _resolve_model_config(config)

        model = PtychoViT(config=model_config)
        device = self._select_device()

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                name = key
                if name.startswith('module.'):
                    name = name[len('module.') :]
                cleaned_state_dict[name] = value
            state_dict = cleaned_state_dict

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        self._model = model
        self._model_config = model_config
        self._model_size = model_size
    def _resolve_normalization(self, parameters: ReconstructInput) -> float:
        """Return normalization scalar for the current product.

        When "Specify Normalization" is enabled, this returns the user-provided
        value. Otherwise, it uses the maximum intensity in the diffraction
        patterns.
        """
        if self._inference_settings.specify_normalization.get_value():
            normalization = float(self._inference_settings.normalization.get_value())
            logger.info('Using specified normalization value %s.', normalization)
            return normalization

        normalization = float(numpy.max(parameters.diffraction_patterns))
        logger.info('Using max diffraction value %s for normalization.', normalization)
        return normalization

    @staticmethod
    def _get_project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        import torch

        model = self._ensure_model_loaded()
        device = self._select_device()
        model.to(device)

        patterns = parameters.diffraction_patterns
        if patterns.ndim != 3:
            raise ValueError(f'Expected diffraction patterns with 3 dimensions, got {patterns.ndim}.')

        num_patterns, height_px, width_px = patterns.shape
        if height_px != width_px:
            raise ValueError('SYNAPS-I expects square diffraction patterns.')
        if self._model_size is not None and height_px != self._model_size:
            raise ValueError(
                f'SYNAPS-I model expects {self._model_size}x{self._model_size} patterns.'
            )

        if len(parameters.product.probe_positions) != num_patterns:
            raise ValueError('Number of diffraction patterns does not match probe positions.')

        normalization = self._resolve_normalization(parameters)
        scale = self._inference_settings.scale.get_value()
        if normalization <= 0.0 or scale <= 0.0:
            raise ValueError('Normalization and scale must be positive for SYNAPS-I inference.')
        batch_size = self._inference_settings.batch_size.get_value()
        max_probe_modes = self._inference_settings.max_probe_modes.get_value()

        scaled_patterns = (patterns.astype(numpy.float32) / normalization) * scale
        scaled_patterns = numpy.sqrt(numpy.maximum(scaled_patterns, 0.0))
        input_tensor = torch.from_numpy(scaled_patterns).unsqueeze(1).to(device)

        probe_array = parameters.product.probes.get_probe_no_opr().get_array()
        if probe_array.size == 0:
            probe_array = numpy.ones(
                (1, max_probe_modes, height_px, width_px), dtype=numpy.complex64
            )
        if not numpy.iscomplexobj(probe_array):
            probe_array = probe_array.astype(numpy.complex64)

        probe_array = _pad_probe(probe_array, max_probe_modes)
        if probe_array.shape[-2:] != (height_px, width_px):
            raise ValueError('Probe dimensions do not match diffraction pattern size.')

        probe_complex = torch.from_numpy(probe_array).to(device)
        probe_complex = probe_complex.unsqueeze(0)
        probe_real = torch.view_as_real(probe_complex)

        object_in = parameters.product.object_
        object_array = object_in.get_array()
        object_geometry = object_in.get_geometry()
        stitcher = BarycentricArrayStitcher(
            upper=numpy.zeros_like(object_array),
            lower=numpy.zeros_like(object_array, dtype=float),
        )

        with torch.no_grad():
            for start in range(0, num_patterns, batch_size):
                end = min(start + batch_size, num_patterns)
                batch = input_tensor[start:end]
                batch_len = end - start

                probe_batch = probe_real.expand(batch_len, -1, -1, -1, -1, -1)
                normalization_batch = torch.full((batch_len,), normalization, device=device)
                scale_batch = torch.full((batch_len,), scale, device=device)

                _, amp_batch, ph_batch = model(
                    batch, probe_batch, normalization_batch, scale_batch
                )

                amp_np = amp_batch.squeeze(1).detach().cpu().numpy()
                ph_np = ph_batch.squeeze(1).detach().cpu().numpy()

                for offset in range(batch_len):
                    patch = amp_np[offset] * numpy.exp(1j * ph_np[offset])
                    scan_point = parameters.product.probe_positions[start + offset]
                    object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
                    weight = numpy.ones_like(patch, dtype=float)
                    stitcher.add_patch(
                        object_point.coordinate_x_px,
                        object_point.coordinate_y_px,
                        patch,
                        weight=weight,
                    )

        object_out = Object(
            array=stitcher.stitch(),
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
            layer_spacing_m=object_in.layer_spacing_m,
        )
        losses: Sequence[LossValue] = []

        product = Product(
            metadata=parameters.product.metadata,
            probe_positions=parameters.product.probe_positions,
            probes=parameters.product.probes,
            object_=object_out,
            losses=losses,
        )

        yield ReconstructOutput(product)

    def get_model_file_filter(self) -> str:
        return self.MODEL_FILE_FILTER

    def open_model(self, file_path: Path) -> None:
        self._inference_settings.model_path.set_value(file_path)
        config_path = self._resolve_config_path(file_path)
        self._inference_settings.config_path.set_value(config_path)
        self._load_model(file_path, config_path)

    def save_model(self, file_path: Path) -> None:
        import torch

        model = self._ensure_model_loaded()
        torch.save(model.state_dict(), file_path)

    def get_training_data_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        raise NotImplementedError('SYNAPS-I training is not implemented yet.')

    def get_training_data_path(self) -> Path:
        model_path = self._inference_settings.model_path.get_value()
        if model_path.exists():
            return model_path.parent
        return Path()

    def train(self, data_path: Path) -> TrainOutput:
        raise NotImplementedError('SYNAPS-I training is not implemented yet.')
