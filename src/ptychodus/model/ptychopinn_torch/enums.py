from collections.abc import Iterator, Sequence


class PtychoPINNEnumerators:
    def __init__(self) -> None:
        self._generator_architectures: Sequence[str] = [
            'cnn',
            'fno',
            'hybrid',
            'stable_hybrid',
            'fno_vanilla',
            'hybrid_resnet',
        ]
        self._fno_input_transforms: Sequence[str] = ['none', 'sqrt', 'log1p', 'instancenorm']
        self._generator_output_modes: Sequence[str] = ['real_imag', 'amp_phase_logits', 'amp_phase']
        self._loss_functions: Sequence[str] = ['MAE', 'Poisson']
        self._amplitude_losses: Sequence[str] = ['Total_Variation', 'Mean_Deviation', 'None']
        self._phase_losses: Sequence[str] = ['Total_Variation', 'Mean_Deviation', 'None']
        self._scheduler: Sequence[str] = [
            'Default',
            'Exponential',
            'MultiStage',
            'Adaptive',
            'WarmupCosine',
            'ReduceLROnPlateau',
        ]
        self._torch_loss_mode: Sequence[str] = ['poisson', 'mae']
        self._optimizers: Sequence[str] = [
            'Adadelta',
            'Adafactor',
            'Adagrad',
            'Adam',
            'AdamW',
            'SparseAdam',
            'Adamax',
            'ASGD',
            'LBFGS',
            'Muon',
            'NAdam',
            'RAdam',
            'RMSprop',
            'Rprop',
            'SGD',
        ]

    def get_generator_architectures(self) -> Iterator[str]:
        return iter(self._generator_architectures)

    def get_fno_input_transforms(self) -> Iterator[str]:
        return iter(self._fno_input_transforms)

    def get_generator_output_modes(self) -> Iterator[str]:
        return iter(self._generator_output_modes)

    def get_loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def get_amplitude_losses(self) -> Iterator[str]:
        return iter(self._amplitude_losses)

    def get_phase_losses(self) -> Iterator[str]:
        return iter(self._phase_losses)

    def get_scheduler(self) -> Iterator[str]:
        return iter(self._scheduler)

    def get_torch_loss_mode(self) -> Iterator[str]:
        return iter(self._torch_loss_mode)
