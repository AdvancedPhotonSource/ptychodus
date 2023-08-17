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

    def insertItemIntoRepositoryFromFile(self, filePath: Path, fileType: str) -> Optional[str]:
        item = self._factory.openItemFromFile(filePath, fileType=fileType)

        if item is None:
            logger.error(f'Unable to open probe from \"{filePath}\"!')

        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromArray(self,
                                          name: str,
                                          array: ProbeArrayType,
                                          *,
                                          filePath: Optional[Path] = None,
                                          fileType: str = '',
                                          replaceItem: bool = False,
                                          selectItem: bool = False) -> Optional[str]:
        item = self._factory.createItemFromArray(name, array, filePath=filePath, fileType=fileType)
        itemName = self._repository.insertItem(item, name=name if replaceItem else None)

        if itemName is None:
            logger.error(f'Failed to insert probe array \"{name}\"!')
        elif selectItem:
            self._probe.selectItem(itemName)

        return itemName

    def insertItemIntoRepositoryFromInitializerName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromInitializerName(name)
        return self._repository.insertItem(item)

    def getSelectedProbeArray(self) -> ProbeArrayType:
        selectedItem = self._probe.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No probe is selected!')

        return selectedItem.getArray()
