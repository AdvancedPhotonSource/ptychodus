"""Child-side subprocess entry points for the PtychoFM (ptycho-fm) backend.

Runs INSIDE a spawned subprocess. This is the only ptychodus module allowed to
import ``torch`` or ``ptycho_fm``. The parent never imports it -- it reaches
only the payload dataclasses in :mod:`._payload` and the settings-to-dict
translator in :mod:`.reconstructor`, neither of which touches torch.

Two entry points are exposed:

- :func:`run_reconstruct` -- load a ``.pth`` checkpoint into the model chosen by
  ``ptycho_fm.utils.inference.resolve_inference_model``, run one inference pass
  over the diffraction stack, stitch the per-patch amplitude/phase outputs into
  a full-object array via ``place_patches_fourier_shift``, and stream back a
  single :class:`ReconstructOutput`.
- :func:`run_train` -- run one single-device training session (no DDP, no
  mlflow, no wandb) driven by :class:`ptycho_fm.model.model.PtychoFM` and a
  minimal train/validate loop that mirrors ``ptycho_fm/train.py`` stripped of
  its distributed machinery. Emits a :class:`TrainOutput` after each epoch and
  a final ``TAG_MODEL_SAVED`` with the path of the best checkpoint.

Stitching is done here rather than reused: ptycho_fm's own
``run_inference_and_stitch`` lives in its ``scripts/`` tree, which is excluded
from the wheel and so cannot be imported from an installed ptycho-fm.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Sequence
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import ReconstructOutput, TrainOutput

from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import ReconstructPayload, TrainPayload

logger = logging.getLogger(__name__)


def _select_device() -> Any:
    """Return the best available torch device.

    Kept in its own helper so both entry points share the selection rule and
    the import of torch is confined to the child.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _pad_probe_to_modes(probe: numpy.ndarray, target_modes: int) -> numpy.ndarray:
    """Zero-pad a complex probe ``(N_modes, H, W)`` up to ``target_modes`` along axis 0.

    ptycho_fm pads the incoherent-mode axis of a 4D ``(OPR, N_modes, H, W)``
    probe and never truncates it or touches the spatial dimensions; this is the
    same rule applied to the 3D layout ptychodus holds, with the OPR axis added
    by the caller.
    """
    current_modes = probe.shape[0]
    if current_modes >= target_modes:
        return probe
    padding = numpy.zeros(
        (target_modes - current_modes, probe.shape[1], probe.shape[2]),
        dtype=probe.dtype,
    )
    return numpy.concatenate([probe, padding], axis=0)


def _zero_pad_2d_to(image: numpy.ndarray, target_size: int) -> numpy.ndarray:
    """Center a 2D array inside a ``(target_size, target_size)`` zero-padded canvas."""
    h, w = image.shape
    if h == target_size and w == target_size:
        return image
    if h > target_size or w > target_size:
        raise ValueError(
            f'Image size ({h}, {w}) exceeds target size {target_size}; refusing to crop.'
        )
    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return numpy.pad(
        image,
        ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
        mode='constant',
        constant_values=0,
    )


def _build_positions_top_left(parameters: Any) -> numpy.ndarray:
    """Convert ptychodus probe positions to top-left-origin object-pixel coords.

    Returns an ``(N, 2)`` float32 array of ``[y_px, x_px]`` centres, matching
    what ``place_patches_fourier_shift`` expects.
    """
    object_geometry = parameters.product.object_.get_geometry()
    positions = parameters.product.probe_positions
    return numpy.column_stack(
        (
            object_geometry.map_probe_positions_to_object_y_px(positions),
            object_geometry.map_probe_positions_to_object_x_px(positions),
        )
    ).astype(numpy.float32)


def _make_inference_dataset(
    diffraction_amplitude: Any,
    probe: Any,
    positions: Any,
    normalization: float,
    scale: float,
) -> Any:
    """Wrap in-memory arrays as a Dataset yielding ptycho_fm's 7-tuple sample.

    ``predict_object_patches`` destructures every sample as ``(diffraction,
    amplitude, phase, probe, position, normalization, scale)``. Only the
    diffraction is read for an inference-only model, so the amplitude and phase
    slots carry empty tensors; the probe is real and is consumed by the models
    that run the forward physics.

    The class is defined here so ``torch`` stays a child-side import.
    """
    import torch
    from torch.utils.data import Dataset

    empty = torch.empty(0)

    class _InMemoryPatternDataset(Dataset):
        def __len__(self) -> int:
            return int(diffraction_amplitude.shape[0])

        def __getitem__(self, index: int) -> tuple[Any, ...]:
            return (
                diffraction_amplitude[index],
                empty,
                empty,
                probe,
                positions[index],
                normalization,
                scale,
            )

    return _InMemoryPatternDataset()


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for one inference pass. Streams a single ReconstructOutput.

    ``resolve_inference_model`` picks the model class from the config and
    returns the encoder input size alongside it; ``load_checkpoint`` reads the
    ``.pth`` state dict with ``weights_only=True`` (safe: nothing to execute is
    expected in a plain ptycho_fm checkpoint). Patch prediction is delegated to
    ``iter_object_patch_predictions``; only the stitch is ours.
    """
    if payload.model_path is None:
        raise RuntimeError('Cannot reconstruct: no model checkpoint has been loaded.')

    import torch
    from ptychi.image_proc import place_patches_fourier_shift
    from ptycho_fm.utils.inference import (
        crop_patch_borders,
        iter_object_patch_predictions,
        load_checkpoint,
        make_inference_dataloader,
        resolve_inference_model,
    )

    device = _select_device()
    config = payload.config
    data_config = config['data']
    inference_config = config['inference']

    model, image_size = resolve_inference_model(config)
    load_checkpoint(model, payload.model_path, device)

    parameters = payload.reconstruct_input

    diff_intensity = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
    diff_intensity = numpy.asarray(diff_intensity, dtype=numpy.float32)
    if diff_intensity.ndim != 3:
        raise ValueError(
            f'Expected diffraction patterns with shape (N, H, W); got {diff_intensity.shape}.'
        )

    # Match PtychographyDataset preprocessing: normalise by dataset max, scale,
    # then take sqrt (the model's ``x`` input is amplitude-domain).
    normalization_value = float(diff_intensity.max())
    scale_value = float(data_config['scale'])
    if normalization_value <= 0.0:
        raise ValueError('Diffraction stack max is non-positive; cannot normalise for inference.')
    diff_amp = numpy.sqrt(diff_intensity / normalization_value * scale_value)

    # The encoder's position embedding is sized for exactly image_size**2
    # patches, so anything smaller is centred on a zero canvas.
    if diff_amp.shape[-1] != image_size or diff_amp.shape[-2] != image_size:
        diff_amp = numpy.stack([_zero_pad_2d_to(p, image_size) for p in diff_amp], axis=0)

    probe_array = parameters.product.probes.get_probe_no_opr().get_array()
    if probe_array.ndim != 3:
        raise ValueError(f'Expected probe with shape (N_modes, H, W); got {probe_array.shape}.')
    max_modes = int(data_config['max_probe_modes'])
    probe_padded = _pad_probe_to_modes(probe_array, max_modes)
    # Samples carry the probe with an OPR axis in front: (OPR, N_modes, H, W).
    probe_sample = torch.from_numpy(numpy.ascontiguousarray(probe_padded)).to(
        dtype=torch.complex64
    )[None]

    positions_np = _build_positions_top_left(parameters)
    positions = torch.from_numpy(positions_np)

    object_in = parameters.product.object_
    object_array = object_in.get_array()
    object_shape = (object_array.shape[-2], object_array.shape[-1])

    pred_amp_object = torch.zeros(object_shape, dtype=torch.float32)
    pred_ph_object = torch.zeros(object_shape, dtype=torch.float32)
    buffer = torch.zeros(object_shape, dtype=torch.float32)

    central_crop = int(inference_config['central_crop'])
    pad = int(inference_config['pad'])
    batch_size = max(1, int(inference_config['batch_size']))

    dataset = _make_inference_dataset(
        diffraction_amplitude=torch.from_numpy(diff_amp).unsqueeze(1),
        probe=probe_sample,
        positions=positions,
        normalization=normalization_value,
        scale=scale_value,
    )
    dataloader = make_inference_dataloader(dataset, batch_size)

    # make_inference_dataloader pins shuffle=False, so batches arrive in
    # dataset order and a running offset indexes back into positions.
    scan_index = 0

    for output_amp, output_ph in iter_object_patch_predictions(model, dataloader, device):
        amp_patches = crop_patch_borders(output_amp, central_crop)
        ph_patches = crop_patch_borders(output_ph, central_crop)
        batch_positions = positions[scan_index : scan_index + amp_patches.shape[0]]
        scan_index += amp_patches.shape[0]

        pred_amp_object = place_patches_fourier_shift(
            pred_amp_object,
            batch_positions,
            amp_patches,
            op='add',
            adjoint_mode=False,
            pad=pad,
        )
        pred_ph_object = place_patches_fourier_shift(
            pred_ph_object,
            batch_positions,
            ph_patches,
            op='add',
            adjoint_mode=False,
            pad=pad,
        )
        buffer = place_patches_fourier_shift(
            buffer,
            batch_positions,
            torch.ones_like(ph_patches),
            op='add',
            adjoint_mode=False,
            pad=pad,
        )

    if scan_index != positions.shape[0]:
        raise RuntimeError(
            f'Stitched {scan_index} patches but hold {positions.shape[0]} probe positions.'
        )

    divisor = torch.clip(buffer, min=1.0)
    pred_amp_object = pred_amp_object / divisor
    pred_ph_object = pred_ph_object / divisor

    # Fold amplitude + phase back into a complex layer. Preserve the input
    # object's outer shape (layers, H, W) by using layer 0 only.
    complex_layer = (pred_amp_object.numpy() * numpy.exp(1j * pred_ph_object.numpy())).astype(
        numpy.complex64
    )
    if object_array.ndim == 3:
        object_out_array = numpy.zeros_like(object_array)
        object_out_array[0] = complex_layer
    else:
        object_out_array = complex_layer

    object_out = Object(
        array=object_out_array,
        layer_spacing_m=object_in.layer_spacing_m,
        pixel_geometry=object_in.get_pixel_geometry(),
        center=object_in.get_center(),
    )

    losses: Sequence[LossValue] = []
    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_out,
        losses=losses,
    )

    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product=product, progress=1))))


def _build_criterion(training_config: dict[str, Any]) -> Any:
    """Instantiate the loss module named in ``training_config['loss_function']``.

    Mirrors the branch at ``ptycho_fm/train.py`` line ~728.
    """
    import torch.nn as nn

    from ptycho_fm.custom_loss import WeightedLoss

    name = training_config['loss_function']
    if name == 'smooth_l1':
        return nn.SmoothL1Loss()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'l1':
        return nn.L1Loss()
    if name == 'poisson_nll':
        return nn.PoissonNLLLoss(log_input=False, full=False)
    if name == 'weighted':
        w = training_config['weighted_loss']
        return WeightedLoss(loss_type=w['loss_type'], threshold=w['threshold'], alpha=w['alpha'])
    raise ValueError(f'Unknown loss function: {name!r}')


def run_train(payload: TrainPayload, queue: Queue[Any]) -> None:
    """Child entry point for one training session.

    Single-device, no DDP, no mlflow, no wandb. Builds a CombinedDataset over
    ``payload.input_path`` (a directory of paired ``*_dp.hdf5`` / ``*_para.hdf5``
    files -- ptycho_fm's own training format) and runs a lightweight
    train/validate loop directly against :class:`PtychoFM`, emitting one
    :class:`TrainOutput` per epoch. The best checkpoint by validation loss is
    written to ``payload.output_path/best.pth`` and its path streamed back via
    ``TAG_MODEL_SAVED``.
    """
    # Guard the training loop against picking up an in-flight environment: a
    # user may have limited devices via CUDA_VISIBLE_DEVICES already; we
    # respect that and never override it here.
    _visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if _visible is not None:
        logger.info('Training with CUDA_VISIBLE_DEVICES=%s', _visible)

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, random_split

    from ptycho_fm.data import CombinedDataset
    from ptycho_fm.model.model import PtychoFM

    device = _select_device()
    config = payload.config
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']

    if not payload.input_path.exists():
        raise FileNotFoundError(f'Training input directory does not exist: {payload.input_path}')
    if not payload.input_path.is_dir():
        raise NotADirectoryError(
            'ptycho_fm training expects a directory of paired *_dp.hdf5 / '
            f'*_para.hdf5 files; got file: {payload.input_path}'
        )

    payload.output_path.mkdir(parents=True, exist_ok=True)

    dataset = CombinedDataset(
        file_paths=str(payload.input_path),
        rank=0,
        world_size=1,
        scale=data_config['scale'],
        normalization_dict_path=None,
        default_normalization=data_config['default_normalization'],
        apply_noise=False,
        cache_object=data_config['cache_object'],
        max_probe_modes=data_config['max_probe_modes'],
        max_OPR_modes=data_config['max_OPR_modes'],
        cache_memory_budget_mb=data_config['cache_memory_budget_mb'],
        max_files=data_config['max_files'],
        debug=False,
    )
    total_size = len(dataset)
    if total_size < 2:
        raise ValueError(
            f'Training dataset has {total_size} sample(s); need at least 2 for a train/val split.'
        )

    train_split = float(data_config['train_split'])
    train_size = max(1, int(total_size * train_split))
    val_size = max(1, total_size - train_size)
    if train_size + val_size > total_size:
        train_size = total_size - val_size
    generator = torch.Generator().manual_seed(int(data_config['random_seed']))
    train_subset: Subset[Any]
    val_subset: Subset[Any]
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    num_workers = int(data_config['num_workers'])
    batch_size = int(training_config['batch_size'])
    loader_kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': False,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(data_config['prefetch_factor'])
    train_loader = DataLoader(train_subset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_kwargs)

    model = PtychoFM(config=model_config).to(device)

    lr = float(training_config['learning_rate'])
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': lr, 'name': 'encoder'},
        {'params': model.amp_decoder.parameters(), 'lr': lr, 'name': 'amp_decoder'},
        {'params': model.ph_decoder.parameters(), 'lr': lr, 'name': 'ph_decoder'},
    ]
    optimizer = optim.Adam(param_groups)

    criterion = _build_criterion(training_config)
    is_supervised = training_config['mode'] == 'supervised'
    image_size = int(model_config['encoder']['img_size'])
    checked_pattern_size = False

    def _forward(batch: tuple[Any, ...]) -> Any:
        nonlocal checked_pattern_size
        diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

        if not checked_pattern_size:
            height, width = diff_amp.shape[-2:]
            if (height, width) != (image_size, image_size):
                raise ValueError(
                    f'Training patterns are {height}x{width}, but PtychoFM requires '
                    f'{image_size}x{image_size} to match the encoder input. Re-export the '
                    'training data or change the model image size.'
                )
            checked_pattern_size = True

        input_diff = diff_amp.to(device)
        input_probe = torch.view_as_real(probe.clone().detach()).to(device)
        # Samples carry normalization and scale as Python floats, which collate
        # to float64. Left alone they promote the modelled diffraction to
        # float64 and the unsupervised loss then disagrees with its float32
        # target; the decoded amplitude and phase never touch them.
        input_norm = norm.to(device=device, dtype=torch.float32)
        input_scale = scale.to(device=device, dtype=torch.float32)
        pred_diff, pred_amp, pred_ph = model(input_diff, input_probe, input_norm, input_scale)

        # Supervised compares the decoded object patch against the ground truth
        # the dataset carries; unsupervised compares the modelled diffraction
        # against the measurement and needs no object at all. Same split as
        # ptycho_fm's own trainer, minus its wandb-driven auxiliary terms.
        if is_supervised:
            amp_loss = criterion(pred_amp, amp_patch.to(device))
            ph_loss = criterion(pred_ph, ph_patch.to(device))
            return amp_loss + ph_loss

        return criterion(pred_diff, input_diff)

    training_losses: list[LossValue] = []
    validation_losses: list[LossValue] = []
    best_val = float('inf')
    best_path = payload.output_path / 'best.pth'
    save_epoch_models = bool(training_config['save_epoch_models'])
    epochs = int(training_config['epochs'])

    for epoch in range(epochs):
        model.train()
        running_train = 0.0
        n_train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _forward(batch)
            loss.backward()
            optimizer.step()
            running_train += float(loss.detach().cpu().item())
            n_train_batches += 1
        train_loss = running_train / max(n_train_batches, 1)
        training_losses.append(LossValue(epoch=epoch, value=train_loss))

        model.eval()
        running_val = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = _forward(batch)
                running_val += float(loss.detach().cpu().item())
                n_val_batches += 1
        val_loss = running_val / max(n_val_batches, 1)
        validation_losses.append(LossValue(epoch=epoch, value=val_loss))

        if val_loss < best_val:
            best_val = val_loss
            _atomic_save_state_dict(model.state_dict(), best_path)
        if save_epoch_models:
            epoch_path = payload.output_path / f'model_epoch_{epoch + 1:03d}.pth'
            _atomic_save_state_dict(model.state_dict(), epoch_path)

        queue.put(
            (
                TAG_TRAIN_OUTPUT,
                pickle.dumps(
                    TrainOutput(
                        training_loss=list(training_losses),
                        validation_loss=list(validation_losses),
                        progress=epoch + 1,
                    )
                ),
            )
        )

    if not best_path.exists():
        raise FileNotFoundError(
            f'Training finished but no best checkpoint was written at {best_path}.'
        )
    queue.put((TAG_MODEL_SAVED, str(best_path)))


def _atomic_save_state_dict(state_dict: Any, destination: Path) -> None:
    """Save a torch state dict atomically: write to a temp path, then rename.

    ``model.state_dict()`` may reference CUDA tensors; we move them to CPU so
    reloading does not require the same device layout.
    """
    import torch

    cpu_state = {k: v.detach().cpu() if hasattr(v, 'detach') else v for k, v in state_dict.items()}
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state, tmp)
    os.replace(tmp, destination)
