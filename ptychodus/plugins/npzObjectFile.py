from pathlib import Path

import numpy

from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.state import StateDataRegistry


class NPZObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> Object:
        npz = numpy.load(filePath)
        array = npz[StateDataRegistry.OBJECT_ARRAY]
        object_ = Object(array)

        try:
            layerDistanceInMeters = npz[StateDataRegistry.OBJECT_LAYER_DISTANCE]
        except KeyError:
            pass
        else:
            for layer, distanceInMeters in enumerate(layerDistanceInMeters):
                object_.setLayerDistanceInMeters(layer, distanceInMeters)

        return object_


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        NPZObjectFileReader(),
        simpleName='NPZ',
        displayName=StateDataRegistry.FILE_FILTER,
    )
