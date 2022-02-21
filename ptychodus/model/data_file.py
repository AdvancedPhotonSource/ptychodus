from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import logging

import h5py
import numpy
import threading
import watchdog.events
import watchdog.observers

logger = logging.getLogger(__name__)


class H5FileEventHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self) -> None:
        super().__init__(patterns=['*.h5', '*.hdf5'],
                         ignore_directories=True,
                         case_sensitive=False)

    def on_any_event(self, event) -> None:
        print(f'{event.event_type}: {event.src_path}')


class DataDirectoryWatcher(threading.Thread):
    def __init__(self) -> None:
        super().__init__()
        self._directoryPath: Path = Path.home()
        self._observer = watchdog.observers.Observer()
        self._eventHandler = H5FileEventHandler()
        self._waitTimeInSeconds = 1.
        self._stopEvent = threading.Event()
        self._watch = None

    def run(self) -> None:
        self._observer.schedule(self._eventHandler, self._directoryPath, recursive=False)
        self._observer.start()

        try:
            while not self._stopEvent.wait(self._waitTimeInSeconds):
                pass

        except:
            self._observer.stop()
            logging.error('Watchdog Thread Error')  # TODO improve message

        self._observer.join()

    def stop(self) -> None:
        self._observer.stop()
        self._stopEvent.set()


class DataFileReader(ABC):
    @abstractmethod
    def read(self, rootGroup: h5py.Group) -> None:
        pass


class DataFilePresenter:
    def __init__(self) -> None:
        super().__init__()
        self._readerList: list[DataFileReader] = list()
        self._filePath: Optional[Path] = None

    def addReader(self, reader: DataFileReader) -> None:
        if reader not in self._readerList:
            self._readerList.append(reader)

    def readFile(self, filePath: Path) -> None:
        if filePath is None:
            return

        with h5py.File(filePath, 'r') as h5File:
            self._filePath = filePath

            for reader in self._readerList:
                reader.read(h5File)

    def readData(self, dataPath: str) -> Any:
        data = None

        if self._filePath and dataPath:
            with h5py.File(self._filePath, 'r') as h5File:
                if dataPath in h5File:
                    item = h5File.get(dataPath)

                    if isinstance(item, h5py.Dataset):
                        data = item[()]
                else:
                    parentPath, attrName = dataPath.rsplit('/', 1)

                    if parentPath in h5File:
                        item = h5File.get(parentPath)

                        if attrName in item.attrs:
                            data = item.attrs[attrName]

        #if isinstance(data, numpy.bytes_): # TODO h5py.check_string_dtype
        #    data = value.decode('utf-8')

        return data
