from abc import abstractmethod
from pathlib import Path
import logging

from ...api.parametric import ParameterRepository
from ...api.object import Object, ObjectFileReader

logger = logging.getLogger(__name__)


class ObjectBuilder(ParameterRepository):

    def __init__(self, name: str) -> None:
        super().__init__('Builder')
        self._name = self._registerStringParameter('Name', name)

    @abstractmethod
    def build(self) -> Object:
        pass


class FromMemoryObjectBuilder(ObjectBuilder):

    def __init__(self, object_: Object) -> None:
        super().__init__('From Memory')
        self._object = object_

    def build(self) -> Object:
        return self._object


class FromFileObjectBuilder(ObjectBuilder):

    def __init__(self, filePath: Path, fileType: str, fileReader: ObjectFileReader) -> None:
        super().__init__('From File')
        self.filePath = self._registerPathParameter('FilePath', filePath)
        self.fileType = self._registerStringParameter('FileType', fileType)
        self._fileReader = fileReader

    def build(self) -> Object:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

        try:
            object_ = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{filePath}\"') from exc

        return object_
