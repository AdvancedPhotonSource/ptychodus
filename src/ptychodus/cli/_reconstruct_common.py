"""Shared reconstruction tail for the ``cli/reconstruct_*.py`` drivers.

Every driver assembles diffraction data, builds an initial :class:`~ptychodus.api.product.Product`,
and (optionally) saves ``diffraction.h5`` -- that part differs per instrument. From there on, every
driver aligns its pty-chi options against the product, runs the reconstruction to convergence, and
writes the standard-layout outputs identically. :func:`run_reconstruction` is that common tail,
extracted so it is written and tested once rather than duplicated in eleven files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.io import StandardFileLayout, save_product
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    dump_task_options,
    reconstruct_with_ptychi,
)

__all__ = ['run_reconstruction']


def run_reconstruction(
    logger: logging.Logger,
    reconstruct_input: ReconstructInput,
    options: PtychographyTaskOptions,
    output_directory: Path,
    *,
    num_sync_epochs: int,
) -> Product:
    """Align `options` against the product, run pty-chi to completion, and write the outputs.

    Assumes `output_directory` already exists (the caller creates it ahead of saving
    `diffraction.h5`, which this function does not touch -- that artifact depends on the
    assembled dataset, not on the reconstruction, so saving it is the caller's job). Aligns
    `options` against `reconstruct_input.product`, writes the aligned options to
    `ptychi_options.json`, runs `reconstruct_with_ptychi` to completion writing a
    `product.NNNNNN.h5` checkpoint at every sync point, then writes the final `product.h5`.
    Returns the final reconstructed product.
    """
    options_file = StandardFileLayout.PTYCHI_OPTIONS.path(output_directory)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)

    task_options = align_task_options_with_product(options, reconstruct_input.product)
    task_options.check()

    # The aligned options are the ones that ran: they carry the object pixel size, the
    # wavelength and the slice spacings the product supplied, which the unaligned object
    # does not.
    logger.info('Writing %s', options_file)
    options_file.write_text(dump_task_options(task_options))

    num_epochs = int(task_options.reconstructor_options.num_epochs)
    logger.info(
        'Starting reconstruction: %d patterns, %d epochs total, sync every %d',
        reconstruct_input.diffraction_patterns.shape[0],
        num_epochs,
        num_sync_epochs,
    )

    final_output = None

    for output in reconstruct_with_ptychi(
        reconstruct_input, task_options, num_sync_epochs=num_sync_epochs
    ):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Epoch %d/%d: loss=%.6g', output.progress, num_epochs, last_loss)

        checkpoint_file = StandardFileLayout.PRODUCT.checkpoint_path(
            output_directory, output.progress
        )
        save_product(checkpoint_file, output.product)
        final_output = output

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    save_product(product_file, final_output.product)
    logger.info('Saved reconstructed product to %s', product_file)
    return final_output.product
