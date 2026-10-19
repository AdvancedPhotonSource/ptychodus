from __future__ import annotations

from ...api.reconstruct import (
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .enums import PtychoPINNEnumerators
from .settings import (
    PtychoPINNDataSettings,
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settings_registry: SettingsRegistry) -> None:
        super().__init__('ptychopinn', 'ptychopinn', requires=('ptycho_torch', 'lightning'))
        self.data_settings = PtychoPINNDataSettings(settings_registry)
        self.model_settings = PtychoPINNModelSettings(settings_registry)
        self.training_settings = PtychoPINNTrainingSettings(settings_registry)
        self.inference_settings = PtychoPINNInferenceSettings(settings_registry)
        self.enumerators = PtychoPINNEnumerators()

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def _build(self) -> list[TrainableReconstructor]:
        # Imported lazily so this module's own import cost stays small.
        from .reconstructor import build_reconstructor

        return [
            build_reconstructor(
                mode,
                self.data_settings,
                self.model_settings,
                self.inference_settings,
                self.training_settings,
            )
            for mode in ('Unsupervised', 'Supervised')
        ]
