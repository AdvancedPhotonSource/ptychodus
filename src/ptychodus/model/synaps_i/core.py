from __future__ import annotations
from collections.abc import Iterator
import logging

from ...api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from .settings import SynapsIInferenceSettings

logger = logging.getLogger(__name__)


class SynapsIReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('synaps_i')
        self.inference_settings = SynapsIInferenceSettings(settings_registry)
        self._reconstructors: list[Reconstructor] = list()

        try:
            from .reconstructor import SynapsITrainableReconstructor
        except ModuleNotFoundError:
            logger.info('SYNAPS-I not found.')

            if is_developer_mode_enabled:
                self._reconstructors.append(NullReconstructor('SYNAPS-I'))
        else:
            self._reconstructors.append(
                SynapsITrainableReconstructor(
                    self.inference_settings,
                    is_developer_mode_enabled=is_developer_mode_enabled,
                )
            )

    @property
    def name(self) -> str:
        return 'SYNAPS-I'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
