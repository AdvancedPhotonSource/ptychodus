from __future__ import annotations

import numpy

from ...api.object import ObjectFileReader, ObjectFileWriter
from ...api.plugins import PluginChooser
from .builderFactory import ObjectBuilderFactory
from .itemFactory import ObjectRepositoryItemFactory


class ObjectCore:

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self.builderFactory = ObjectBuilderFactory(rng, fileReaderChooser, fileWriterChooser)
        self.repositoryItemFactory = ObjectRepositoryItemFactory(rng)
