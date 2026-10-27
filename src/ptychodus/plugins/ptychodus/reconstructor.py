"""Parent-side factory that builds the :class:`SubprocessReconstructor` for Kickstart.

The declared options are translated here into the api's own parameter objects, so the
child only accumulates the moments and calls the estimate.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ptychodus.api.kickstart import KickstartParameters
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from .options import KickstartOptions, default_options

__all__ = [
    'build_parameters',
    'build_reconstructor',
]


def _declared(options: BaseModel) -> KickstartOptions:
    if not isinstance(options, KickstartOptions):
        raise TypeError(f'Expected KickstartOptions, got {type(options).__name__}.')

    return options


def build_parameters(options: KickstartOptions) -> KickstartParameters:
    """The api parameters the declared options describe."""
    return KickstartParameters(
        defocus_distance_m=options.defocus_distance_m,
        parallax_min_significance=options.parallax_min_significance,
        parallax_min_defocus_dof=options.parallax_min_defocus_dof,
        parallax_rtol=options.parallax_rtol,
        incident_photon_count_quantile=options.incident_photon_count_quantile,
        bright_field_mad_threshold=options.probe_bright_field_mad_threshold,
        regularization_fov_fraction=options.phase_regularization_fov_fraction,
    )


def build_reconstructor() -> SubprocessReconstructor:
    def dump_reconstruct_options(input_: ReconstructInput, options: BaseModel) -> dict[str, Any]:
        # Checked here so a near-field product is refused before a child is spawned.
        if not input_.product.metadata.far_field:
            raise ValueError('Kickstart estimates from far-field diffraction data only.')

        declared = _declared(options)
        bin_size_px = declared.parallax_bin_size_px
        return {
            'parameters': build_parameters(declared),
            'binning': BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px),
            'update_count': declared.update_count,
        }

    return SubprocessReconstructor(
        name='Kickstart',
        default_options=default_options,
        dump_reconstruct_options=dump_reconstruct_options,
    )
