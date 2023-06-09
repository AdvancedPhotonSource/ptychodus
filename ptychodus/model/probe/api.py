from pathlib import Path
from typing import Optional
import logging

from ...api.probe import ProbeArrayType
from .factory import ProbeRepositoryItemFactory
from .repository import ProbeRepository
from .selected import SelectedProbe

logger = logging.getLogger(__name__)


class ProbeAPI:

    def __init__(self, factory: ProbeRepositoryItemFactory, repository: ProbeRepository,
                 probe: SelectedProbe) -> None:
        self._factory = factory
        self._repository = repository
        self._probe = probe

    def insertItemIntoRepositoryFromFile(self,
                                         filePath: Path,
                                         *,
                                         simpleFileType: str = '',
                                         displayFileType: str = '') -> Optional[str]:
        item = self._factory.openItemFromFile(filePath,
                                              simpleFileType=simpleFileType,
                                              displayFileType=displayFileType)

        if item is None:
            logger.error(f'Unable to open probe from \"{filePath}\"!')

        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromArray(self,
                                          nameHint: str,
                                          array: ProbeArrayType,
                                          *,
                                          filePath: Optional[Path] = None,
                                          simpleFileType: str = '',
                                          displayFileType: str = '') -> Optional[str]:
        item = self._factory.createItemFromArray(nameHint,
                                                 array,
                                                 filePath=filePath,
                                                 simpleFileType=simpleFileType,
                                                 displayFileType=displayFileType)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializerSimpleName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromSimpleName(name)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializerDisplayName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromDisplayName(name)
        return self._repository.insertItem(item)

    def selectItem(self, itemName: str) -> None:
        self._probe.selectItem(itemName)

    def getSelectedProbeArray(self) -> ProbeArrayType:
        selectedItem = self._probe.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No probe is selected!')

        return selectedItem.getArray()
