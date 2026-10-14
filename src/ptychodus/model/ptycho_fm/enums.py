from collections.abc import Iterator, Sequence

# Settings spelling of ptycho_fm's model.coupled_decoder_mode selector. 'none'
# stands in for the absent/null mode, which builds separate amplitude and phase
# decoders; the coupled modes share one decoder and run the forward physics.
COUPLED_DECODER_MODES: Sequence[str] = ('none', 'real_imag', 'amp_phase')


class PtychoFMEnumerators:
    """Value lists for the settings parameters that name a ptycho_fm choice.

    ``combined`` is absent from the loss functions on purpose: it is configured
    by a list of weighted loss terms, which has no representation in a settings
    group of scalars.
    """

    def __init__(self) -> None:
        self._encoder_types: Sequence[str] = ['custom', 'pretrained']
        self._coupled_decoder_modes: Sequence[str] = COUPLED_DECODER_MODES
        self._init_methods: Sequence[str] = ['trunc_normal', 'kaiming']
        self._loss_functions: Sequence[str] = [
            'smooth_l1',
            'mse',
            'l1',
            'poisson_nll',
            'total_flux',
            'weighted',
            'probe_aware',
            'q_dependent',
        ]
        # Shared by the weighted, probe-aware and q-dependent losses, which
        # validate loss_type against the same pair.
        self._loss_types: Sequence[str] = ['mse', 'mae']

    def get_encoder_types(self) -> Iterator[str]:
        return iter(self._encoder_types)

    def get_coupled_decoder_modes(self) -> Iterator[str]:
        return iter(self._coupled_decoder_modes)

    def get_init_methods(self) -> Iterator[str]:
        return iter(self._init_methods)

    def get_loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def get_loss_types(self) -> Iterator[str]:
        return iter(self._loss_types)
