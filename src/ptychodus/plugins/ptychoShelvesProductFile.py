from pathlib import Path
from typing import Final, Sequence

import numpy
import scipy.io

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader, ProbeFileWriter
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
    Product,
    ProductFileReader,
    ProductMetadata,
)
from ptychodus.api.scan import Scan, ScanPoint


class PtychoShelvesProductFileIO(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'PtychoShelves'
    DISPLAY_NAME: Final[str] = 'PtychoShelves Files (*.mat)'

    def read(self, filePath: Path) -> Product:
        scanPointList: list[ScanPoint] = list()

        hc_eVm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S / ELECTRON_VOLT_J
        matDict = scipy.io.loadmat(filePath, simplify_cells=True)
        p_struct = matDict['p']
        probe_energy_eV = hc_eVm / p_struct['lambda']

        metadata = ProductMetadata(
            name=filePath.stem,
            comments='',
            detectorDistanceInMeters=0.0,  # not included in file
            probeEnergyInElectronVolts=probe_energy_eV,
            probePhotonCount=0.0,  # not included in file
            exposureTimeInSeconds=0.0,  # not included in file
        )

        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]
        pixel_height_m = dx_spec[1]
        pixel_geometry = PixelGeometry(widthInMeters=pixel_width_m, heightInMeters=pixel_height_m)

        outputs_struct = matDict['outputs']
        probe_positions = outputs_struct['probe_positions']

        for idx, pos_px in enumerate(probe_positions):
            point = ScanPoint(
                idx,
                pos_px[0] * pixel_width_m,
                pos_px[1] * pixel_height_m,
            )
            scanPointList.append(point)

        probe = Probe(
            # probeMatrix[width, height, num_shared_modes, num_varying_modes]
            matDict['probe'].transpose(),
            pixel_geometry,
        )

        object_array = matDict['object'].transpose()
        layer_distance_m: Sequence[float] = list()

        try:
            multi_slice_param = p_struct['multi_slice_param']
            z_distance = multi_slice_param['z_distance']
        except KeyError:
            pass
        else:
            num_spaces = object_array.shape[-3] - 1
            layer_distance_m = numpy.squeeze(z_distance)[:num_spaces]

        object_ = Object(
            # object[width, height, num_layers]
            array=object_array,
            pixelGeometry=pixel_geometry,
            center=None,
            layerDistanceInMeters=layer_distance_m,
        )
        costs = outputs_struct['fourier_error_out']

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
            costs=costs,
        )


class PtychoShelvesProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        matDict = scipy.io.loadmat(filePath)

        # array[width, height, num_shared_modes, num_varying_modes]
        array = matDict['probe']
        p_struct = matDict['p']
        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]
        pixel_height_m = dx_spec[1]
        pixel_geometry = PixelGeometry(widthInMeters=pixel_width_m, heightInMeters=pixel_height_m)

        return Probe(array=array.transpose(), pixelGeometry=pixel_geometry)


class PtychoShelvesProbeFileWriter(ProbeFileWriter):
    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.getArray()
        matDict = {'probe': array.transpose()}
        scipy.io.savemat(filePath, matDict)


class PtychoShelvesObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        matDict = scipy.io.loadmat(filePath)

        # array[width, height, num_layers]
        p_struct = matDict['p']
        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]
        pixel_height_m = dx_spec[1]
        pixel_geometry = PixelGeometry(widthInMeters=pixel_width_m, heightInMeters=pixel_height_m)

        object_array = matDict['object'].transpose()
        layer_distance_m: Sequence[float] = list()

        try:
            multi_slice_param = p_struct['multi_slice_param']
            z_distance = multi_slice_param['z_distance']
        except KeyError:
            pass
        else:
            num_spaces = object_array.shape[-3] - 1
            layer_distance_m = numpy.squeeze(z_distance)[:num_spaces]

        return Object(
            # object[width, height, num_layers]
            array=object_array,
            pixelGeometry=pixel_geometry,
            center=None,
            layerDistanceInMeters=layer_distance_m,
        )


class PtychoShelvesObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.getArray()
        matDict = {'object': array.transpose()}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.productFileReaders.registerPlugin(
        PtychoShelvesProductFileIO(),
        simpleName=PtychoShelvesProductFileIO.SIMPLE_NAME,
        displayName=PtychoShelvesProductFileIO.DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        PtychoShelvesProbeFileReader(),
        simpleName=PtychoShelvesProductFileIO.SIMPLE_NAME,
        displayName=PtychoShelvesProductFileIO.DISPLAY_NAME,
    )
    registry.probeFileWriters.registerPlugin(
        PtychoShelvesProbeFileWriter(),
        simpleName=PtychoShelvesProductFileIO.SIMPLE_NAME,
        displayName=PtychoShelvesProductFileIO.DISPLAY_NAME,
    )
    registry.objectFileReaders.registerPlugin(
        PtychoShelvesObjectFileReader(),
        simpleName=PtychoShelvesProductFileIO.SIMPLE_NAME,
        displayName=PtychoShelvesProductFileIO.DISPLAY_NAME,
    )
    registry.objectFileWriters.registerPlugin(
        PtychoShelvesObjectFileWriter(),
        simpleName=PtychoShelvesProductFileIO.SIMPLE_NAME,
        displayName=PtychoShelvesProductFileIO.DISPLAY_NAME,
    )
