from pathlib import Path
from typing import Final

import scipy.io

from ptychodus.api.object import Object, ObjectArrayType, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileWriter
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.propagator import WavefieldArrayType
from ptychodus.api.scan import Scan, ScanPoint


class MATProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'MAT'
    DISPLAY_NAME: Final[str] = 'MAT Files (*.mat)'

    def _load_probe_array(self, probeMatrix: WavefieldArrayType) -> WavefieldArrayType:
        if probeMatrix.ndim == 4:
            # probeMatrix[width, height, num_shared_modes, num_varying_modes]
            # TODO support spatially varying probe modes
            probeMatrix = probeMatrix[..., 0]

        if probeMatrix.ndim == 3:
            # probeMatrix[width, height, num_shared_modes]
            probeMatrix = probeMatrix

        return probeMatrix.transpose()

    def _load_object_array(self, objectMatrix: ObjectArrayType) -> ObjectArrayType:
        if objectMatrix.ndim == 3:
            # objectMatrix[width, height, num_layers]
            objectMatrix = objectMatrix.transpose(2, 0, 1)

        return objectMatrix

    def read(self, filePath: Path) -> Product:
        scanPointList: list[ScanPoint] = list()

        matDict = scipy.io.loadmat(filePath, simplify_cells=True)
        outputs_struct = matDict['outputs']
        probe_positions = outputs_struct['probe_positions']

        for idx, pos_UNITS in enumerate(probe_positions):  # FIXME
            point = ScanPoint(idx, pos_UNITS[0], pos_UNITS[1])  # FIXME
            scanPointList.append(point)

        probe_array = self._load_probe_array(matDict['probe'])
        p_struct = matDict['p']
        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]  # FIXME verify
        pixel_height_m = dx_spec[1]  # FIXME verify

        probe = Probe(
            probe_array,
            pixelWidthInMeters=pixel_width_m,
            pixelHeightInMeters=pixel_height_m,
        )

        object_array = self._load_object_array(matDict['object'])

        try:
            multi_slice_param = p_struct['multi_slice_param']
            layerDistanceInMeters = multi_slice_param['z_distance']
        except KeyError:
            object_ = Object(
                object_array,
                pixelWidthInMeters=pixel_width_m,
                pixelHeightInMeters=pixel_height_m,
            )
        else:
            object_ = Object(
                object_array,
                layerDistanceInMeters,
                pixelWidthInMeters=pixel_width_m,
                pixelHeightInMeters=pixel_height_m,
            )

        costs = outputs_struct['fourier_error_out']

        metadata = ProductMetadata(
            name=filePath.stem,
            comments='',
            detectorDistanceInMeters=0.,  # FIXME float(h5File.attrs[self.DETECTOR_OBJECT_DISTANCE]),
            probeEnergyInElectronVolts=0.,  # FIXME float(h5File.attrs[self.PROBE_ENERGY]),
            probePhotonsPerSecond=0.,  # FIXME float(h5File.attrs[self.PROBE_PHOTON_FLUX]),
            exposureTimeInSeconds=0.,  # FIXME float(h5File.attrs[self.EXPOSURE_TIME]),
        )

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
            costs=costs,
        )


class MATObjectFileWriter(ObjectFileWriter):

    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.array
        # FIXME vvv inconsistent with above vvv
        matDict = {'object': array.transpose(1, 2, 0)}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


class MATProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.array
        # FIXME vvv inconsistent with above vvv
        matDict = {'probe': array.transpose(1, 2, 0)}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.productFileReaders.registerPlugin(
        MATProductFileReader(),
        simpleName=MATProductFileReader.SIMPLE_NAME,
        displayName=MATProductFileReader.DISPLAY_NAME,
    )
    registry.probeFileWriters.registerPlugin(
        MATProbeFileWriter(),
        simpleName=MATProductFileReader.SIMPLE_NAME,
        displayName=MATProductFileReader.DISPLAY_NAME,
    )
    registry.objectFileWriters.registerPlugin(
        MATObjectFileWriter(),
        simpleName=MATProductFileReader.SIMPLE_NAME,
        displayName=MATProductFileReader.DISPLAY_NAME,
    )
