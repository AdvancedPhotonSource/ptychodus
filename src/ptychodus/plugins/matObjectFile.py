from collections.abc import Sequence
from pathlib import Path

import numpy
import scipy.io

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class MATObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        matDict = scipy.io.loadmat(filePath)

        # array[width, height, num_layers]
        array = matDict['object']
        layerDistanceInMeters: Sequence[float] = list()

        try:
            p = matDict['p'][0, 0]
            multi_slice_param = p['multi_slice_param'][0, 0]
            layerDistanceInMeters = numpy.squeeze(multi_slice_param['z_distance'])
        except ValueError:
            pass

        # FIXME test & add pixel geometry
        return Object(
            array=array.transpose(), pixelGeometry=None, layerDistanceInMeters=layerDistanceInMeters
        )


class MATObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.getArray()
        matDict = {'object': array.transpose()}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        MATObjectFileReader(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
    registry.objectFileWriters.registerPlugin(
        MATObjectFileWriter(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
