"""cdtools' FancyPtycho model as a ptychodus :class:`Reconstructor`.

torch and cdtools are imported inside the functions that use them, so importing this
module costs nothing and acquires no GPU context.

The product crosses into cdtools' conventions once, on the way in -- the probe scaled
by the measured gauge, the positions mapped onto the pixel frame of
:mod:`.geometry` -- and every streamed product is carried back out of them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import logging

import numpy
from pydantic import BaseModel

from ptychodus.api.object import Object
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput, Reconstructor
from ptychodus.api.typing import ComplexArrayType, RealArrayType

from .gauge import ProbeGauge, measure_probe_gauge
from .geometry import CDToolsFrame, apply_translation_offsets, build_frame, verify_product
from .options import (
    CDToolsDevice,
    CDToolsLineSearch,
    CDToolsOptimizer,
    CDToolsOptions,
    StageOptions,
    default_options,
)

__all__ = [
    'CDToolsReconstructor',
]

logger = logging.getLogger(__name__)


def _resolve_device(device: CDToolsDevice) -> str:
    """Map the configured device onto a torch device string.

    Runs with the reconstruction because deciding between CUDA and CPU means asking
    torch whether a device is actually present, which only the process computing may do.
    """
    import torch

    if device is CDToolsDevice.CPU:
        return 'cpu'

    if torch.cuda.is_available():
        return 'cuda'

    if device is CDToolsDevice.CUDA:
        logger.warning('A CUDA device was requested but none is available; using the CPU.')

    return 'cpu'


def _far_field_intensity(wavefield: ComplexArrayType) -> RealArrayType:
    """Detector intensity of *wavefield* under cdtools' own far-field propagator."""
    import torch
    from cdtools.tools.propagators import far_field

    with torch.no_grad():
        propagated = far_field(torch.as_tensor(numpy.asarray(wavefield, dtype=numpy.complex64)))
        return numpy.asarray(torch.abs(propagated).square().numpy(), dtype=float)


def _is_exponentiated(options: CDToolsOptions) -> bool:
    """Whether FancyPtycho holds the object as its complex phase delay.

    A phase-only object implies it: FancyPtycho discards the imaginary part of the phase
    delay only in that representation, and ignores the switch in the other.
    """
    return options.model.exponentiate_obj or options.model.phase_only


def _object_guess(object_array: ComplexArrayType, options: CDToolsOptions) -> ComplexArrayType:
    """The object as FancyPtycho holds it: the transmission, or its complex phase delay.

    An exponentiated object is ``exp(1j * obj)``, so the guess is ``-1j * log(T)``. The
    modulus is floored first, since a fully opaque pixel has no logarithm.
    """
    if not _is_exponentiated(options):
        return object_array

    modulus = numpy.maximum(numpy.abs(object_array), numpy.finfo(numpy.float32).tiny)
    return numpy.angle(object_array) - 1j * numpy.log(modulus)


def _object_transmission(
    model_object: ComplexArrayType, options: CDToolsOptions
) -> ComplexArrayType:
    """Undo :func:`_object_guess` for an object FancyPtycho has refined."""
    if not _is_exponentiated(options):
        return model_object

    if options.model.phase_only:
        return numpy.exp(1j * model_object.real)

    return numpy.exp(1j * model_object)


def _probe_support(height_px: int, width_px: int, radius_px: float) -> Any:
    """A disk of *radius_px* about the beam axis, pixel ``N // 2``, or None for no support."""
    import torch

    if radius_px <= 0.0:
        return None

    yy, xx = numpy.mgrid[:height_px, :width_px]
    rr = numpy.hypot(yy - height_px // 2, xx - width_px // 2)
    return torch.as_tensor(rr < radius_px)


def _build_model(
    input_: ReconstructInput, frame: CDToolsFrame, gauge: ProbeGauge, options: CDToolsOptions
) -> Any:
    import torch
    from cdtools.models import FancyPtycho

    product = input_.product
    probes = product.probes

    if probes.num_coherent_modes > 1:
        logger.warning(
            'cdtools models no orthogonal probe relaxation; reconstructing from the '
            'scan-mean probe of the %d coherent modes.',
            probes.num_coherent_modes,
        )

    probe_array = probes.get_mean_probe().get_array() * gauge.to_cdtools
    refinement = options.refinement

    # cdtools stores the square root of the background, and a model with none starts from
    # this floor, the one cdtools' own initialization uses.
    background_root_floor = 1.0e-6

    if refinement.use_background and input_.background is not None:
        background = torch.as_tensor(
            numpy.sqrt(numpy.clip(input_.background, 0.0, None)), dtype=torch.float32
        )
    else:
        background = torch.full(input_.bad_pixels.shape, background_root_floor)

    qe_mask = None

    if refinement.use_flat_field and input_.flat_field is not None:
        qe_mask = torch.as_tensor(input_.flat_field, dtype=torch.float32)

    probe_guess = torch.as_tensor(probe_array, dtype=torch.complex64)

    # FancyPtycho holds a Fourier-space probe in this mode but takes the guess as given;
    # only its own from_dataset converts one.
    if options.model.fourier_probe:
        from cdtools.tools.propagators import far_field

        probe_guess = far_field(probe_guess)

    model = FancyPtycho(
        wavelength=frame.wavelength_m,
        detector_geometry=frame.get_detector_geometry(),
        obj_basis=frame.obj_basis,
        probe_guess=probe_guess,
        obj_guess=torch.as_tensor(
            _object_guess(product.object_.get_layers_flattened(), options),
            dtype=torch.complex64,
        ),
        min_translation=frame.min_translation,
        background=background,
        translation_offsets=(
            torch.zeros((len(product.probe_positions), 2)) if refinement.refine_positions else None
        ),
        mask=torch.as_tensor(numpy.logical_not(input_.bad_pixels)),
        qe_mask=qe_mask,
        translation_scale=refinement.translation_scale,
        probe_support=_probe_support(
            probes.height_px, probes.width_px, options.model.probe_support_radius_px
        ),
        fourier_probe=options.model.fourier_probe,
        loss=str(options.model.loss),
        exponentiate_obj=_is_exponentiated(options),
        phase_only=options.model.phase_only,
    )

    # The optimizers take every parameter the model has; one without a gradient is
    # left as it is.
    model.probe.requires_grad_(refinement.refine_probe)
    model.obj.requires_grad_(refinement.refine_object)
    model.background.requires_grad_(refinement.refine_background)
    return model


def _build_dataset(input_: ReconstructInput, frame: CDToolsFrame) -> Any:
    import torch
    from cdtools.datasets import Ptycho2DDataset

    return Ptycho2DDataset(
        torch.as_tensor(frame.translations, dtype=torch.float32),
        torch.as_tensor(input_.diffraction_patterns, dtype=torch.float32),
        wavelength=frame.wavelength_m,
        detector_geometry=frame.get_detector_geometry(),
        mask=torch.as_tensor(numpy.logical_not(input_.bad_pixels)),
    )


def _run_stage(model: Any, dataset: Any, stage: StageOptions, options: CDToolsOptions) -> Any:
    """Start one stage, returning the generator that yields its per-epoch losses.

    A fresh optimizer per stage, so a new learning rate never inherits the previous
    stage's moment estimates.
    """
    from cdtools.reconstructors import AdamReconstructor, LBFGSReconstructor, SGDReconstructor

    optimizer = options.optimizer
    common: dict[str, Any] = {
        'lr': stage.learning_rate,
        'thread': False,
        'calculation_width': optimizer.calculation_width,
    }

    match optimizer.optimizer:
        case CDToolsOptimizer.ADAM:
            return AdamReconstructor(model, dataset).optimize(
                stage.epochs,
                batch_size=stage.batch_size,
                betas=(optimizer.adam_beta1, optimizer.adam_beta2),
                amsgrad=optimizer.adam_amsgrad,
                schedule=optimizer.adam_schedule,
                shuffle=optimizer.shuffle,
                **common,
            )
        case CDToolsOptimizer.SGD:
            return SGDReconstructor(model, dataset).optimize(
                stage.epochs,
                batch_size=stage.batch_size,
                momentum=optimizer.sgd_momentum,
                dampening=optimizer.sgd_dampening,
                weight_decay=optimizer.sgd_weight_decay,
                nesterov=optimizer.sgd_nesterov,
                shuffle=optimizer.shuffle,
                **common,
            )
        case CDToolsOptimizer.LBFGS:
            line_search = optimizer.lbfgs_line_search
            return LBFGSReconstructor(model, dataset).optimize(
                stage.epochs,
                history_size=optimizer.lbfgs_history_size,
                line_search_fn=None if line_search is CDToolsLineSearch.NONE else str(line_search),
                **common,
            )

    raise ValueError(f'Unknown cdtools optimizer {optimizer.optimizer!r}.')


def _read_product(
    model: Any, input_: ReconstructInput, gauge: ProbeGauge, options: CDToolsOptions
) -> Product:
    """The model's current state, carried back into ptychodus's conventions."""
    product_in = input_.product
    probe = model.probe.detach()

    if bool(model.fourier_probe):
        from cdtools.tools.propagators import inverse_far_field

        probe = inverse_far_field(probe)

    probe_array = (probe * model.probe_norm).cpu().numpy() / gauge.to_cdtools
    object_array = _object_transmission(model.obj.detach().cpu().numpy(), options)
    object_in = product_in.object_
    positions = product_in.probe_positions

    if model.translation_offsets is not None:
        offsets_px = (model.translation_offsets.detach() * model.translation_scale).cpu().numpy()
        pixel_geometry = object_in.get_pixel_geometry()
        positions = apply_translation_offsets(
            positions,
            numpy.asarray(offsets_px, dtype=float),
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    return Product(
        metadata=product_in.metadata,
        probe_positions=positions,
        probes=ProbeEnsemble(
            array=probe_array[numpy.newaxis].astype(complex),
            opr_weights=None,
            pixel_geometry=product_in.probes.get_pixel_geometry(),
        ),
        object_=Object(
            array=object_array[numpy.newaxis].astype(complex),
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        ),
        losses=[
            LossValue(epoch=epoch, value=float(loss))
            for epoch, loss in enumerate(model.loss_history)
        ],
    )


def _give_the_model_unit_weights(model: Any, num_positions: int, device: str) -> None:
    """Work around a device-less tensor in cdtools' forward model.

    With no weights set, ``FancyPtycho.forward`` builds its per-shot weights as
    ``torch.ones(len(index))``, which lands on the CPU whatever device the model
    is on, and then multiplies it by the probes. Every reconstruction ptychodus
    starts leaves weights unset, so on a CUDA host every run raises before it
    produces anything. Present in cdtools through 0.3.3.dev0, and flagged in a
    comment in the function itself.

    Supplying the weights takes the other branch of that same ``if``, where the
    weights are read from the model and so carry its device. A vector of ones
    through a stable coherent illumination is the arithmetic the buggy branch
    was doing, so nothing about the reconstruction changes.

    A plain tensor rather than a ``Parameter``, so it joins no optimizer group
    and nothing is refined that was not before. Delete this once cdtools builds
    those ones on the model's device.
    """
    import torch

    if getattr(model, 'weights', None) is None:
        model.weights = torch.ones(num_positions, device=device)


def _declared(options: BaseModel) -> CDToolsOptions:
    """Narrow the options a call was given to what this backend declares.

    Checked rather than asserted: the dispatching layer validates the type against what
    ``get_default_options`` returns, so a mismatch here means those two have drifted
    apart, and saying so beats an attribute error several frames later.
    """
    if not isinstance(options, CDToolsOptions):
        raise TypeError(f'Expected CDToolsOptions, got {type(options).__name__}.')

    return options


class CDToolsReconstructor(Reconstructor):
    """cdtools' FancyPtycho model, run stage by stage."""

    @property
    def name(self) -> str:
        return 'FancyPtycho'

    def get_default_options(self) -> BaseModel:
        return default_options()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        """Run every stage, yielding a ReconstructOutput as it goes."""
        declared = _declared(options)

        # Checked before torch is reached for, so a product FancyPtycho cannot take is
        # refused at once, and so is a schedule with nothing in it.
        verify_product(input_.product)
        stages = declared.schedule.stages()

        if not stages:
            raise ValueError('Every cdtools stage has zero epochs; there is nothing to run.')

        device = _resolve_device(declared.optimizer.device)

        frame = build_frame(input_.product)
        flat_field = input_.flat_field if declared.refinement.use_flat_field else None
        gauge = measure_probe_gauge(input_, _far_field_intensity, flat_field=flat_field)

        model = _build_model(input_, frame, gauge, declared)
        dataset = _build_dataset(input_, frame)
        model.to(device=device)
        dataset.get_as(device=device)
        _give_the_model_unit_weights(model, len(input_.product.probe_positions), device)

        num_epochs = sum(stage.epochs for stage in stages)
        num_sync_epochs = declared.optimizer.num_sync_epochs
        epoch = 0

        for stage in stages:
            for stage_epoch, _ in enumerate(_run_stage(model, dataset, stage, declared), start=1):
                epoch += 1

                if epoch % num_sync_epochs == 0 or stage_epoch == stage.epochs:
                    yield ReconstructOutput(
                        product=_read_product(model, input_, gauge, declared),
                        progress=epoch,
                        progress_goal=num_epochs,
                    )
