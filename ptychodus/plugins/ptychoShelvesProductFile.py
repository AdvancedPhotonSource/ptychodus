from pathlib import Path
from typing import Final, Sequence

import scipy.io

from ptychodus.api.constants import ELECTRON_VOLT_J, LIGHT_SPEED_M_PER_S, PLANCK_CONSTANT_J_PER_HZ
from ptychodus.api.object import Object, ObjectArrayType, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileWriter
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.propagator import WavefieldArrayType
from ptychodus.api.scan import Scan, ScanPoint


class MATProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'PtychoShelves'
    DISPLAY_NAME: Final[str] = 'PtychoShelves Files (*.mat)'

    def _load_probe_array(self, probeMatrix: WavefieldArrayType) -> WavefieldArrayType:
        if probeMatrix.ndim == 4:
            # probeMatrix[width, height, num_shared_modes, num_varying_modes]
            # TODO support spatially varying probe modes
            probeMatrix = probeMatrix[..., 0]

        if probeMatrix.ndim == 3:
            # probeMatrix[width, height, num_shared_modes]
            probeMatrix = probeMatrix

        return probeMatrix.transpose(2, 0, 1)

    def _load_object_array(self, objectMatrix: ObjectArrayType) -> ObjectArrayType:
        if objectMatrix.ndim == 3:
            # objectMatrix[width, height, num_layers]
            objectMatrix = objectMatrix.transpose(2, 0, 1)

        return objectMatrix

    def read(self, filePath: Path) -> Product:
        scanPointList: list[ScanPoint] = list()

        hc_eVm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S / ELECTRON_VOLT_J
        matDict = scipy.io.loadmat(filePath, simplify_cells=True)
        p_struct = matDict['p']
        probe_energy_eV = hc_eVm / p_struct['lambda']

        metadata = ProductMetadata(
            name=filePath.stem,
            comments='',
            detectorDistanceInMeters=0.,  # not included in file
            probeEnergyInElectronVolts=probe_energy_eV,
            probePhotonsPerSecond=0.,  # not included in file
            exposureTimeInSeconds=0.,  # not included in file
        )

        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]
        pixel_height_m = dx_spec[1]

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
            self._load_probe_array(matDict['probe']),
            pixelWidthInMeters=pixel_width_m,
            pixelHeightInMeters=pixel_height_m,
        )

        layer_distance_m: Sequence[float] | None = None

        try:
            multi_slice_param = p_struct['multi_slice_param']
        except KeyError:
            pass
        else:
            try:
                z_distance = multi_slice_param['z_distance']
            except KeyError:
                pass
            else:
                layer_distance_m = z_distance.tolist()

        object_ = Object(
            self._load_object_array(matDict['object']),
            layer_distance_m,
            pixelWidthInMeters=pixel_width_m,
            pixelHeightInMeters=pixel_height_m,
        )
        costs = outputs_struct['fourier_error_out']

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
        matDict = {'object': array.transpose(1, 2, 0)}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


class MATProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.array
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
