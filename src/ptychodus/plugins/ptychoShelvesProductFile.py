from pathlib import Path
from typing import Final, Sequence

import scipy.io

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileWriter
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
    Product,
    ProductFileReader,
    ProductMetadata,
)
from ptychodus.api.scan import Scan, ScanPoint


class MATProductFileReader(ProductFileReader):
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

        layer_distance_m: Sequence[float] = list()

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
            # object[width, height, num_layers]
            matDict['object'].transpose(),
            pixel_geometry,
            layer_distance_m,
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
        array = object_.getArray()
        matDict = {'object': array.transpose()}
        # TODO layer distance to p.z_distance
        scipy.io.savemat(filePath, matDict)


class MATProbeFileWriter(ProbeFileWriter):
    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.getArray()
        matDict = {'probe': array.transpose()}
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
