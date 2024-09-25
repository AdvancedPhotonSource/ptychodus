from pathlib import Path

import numpy
import scipy.io

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class MATObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        matDict = scipy.io.loadmat(filePath)
        array = matDict["object"]

        if array.ndim == 3:
            # array[width, height, num_layers]
            array = array.transpose(2, 0, 1)

        try:
            p = matDict["p"][0, 0]
            multi_slice_param = p["multi_slice_param"][0, 0]
            layerDistanceInMeters = numpy.squeeze(multi_slice_param["z_distance"])
        except ValueError:
            object_ = Object(array)
        else:
            object_ = Object(array, layerDistanceInMeters)

        return object_


class MATObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.array
        matDict = {"object": array.transpose(1, 2, 0)}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        MATObjectFileReader(),
        simpleName="MAT",
        displayName="MAT Files (*.mat)",
    )
    registry.objectFileWriters.registerPlugin(
        MATObjectFileWriter(),
        simpleName="MAT",
        displayName="MAT Files (*.mat)",
    )
