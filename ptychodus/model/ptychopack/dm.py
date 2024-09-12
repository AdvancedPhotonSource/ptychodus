import numpy
import torch
from ptychopack import (CorrectionPlan, CorrectionPlanElement, DataProduct, DetectorData, Device,
                        DifferenceMap)

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ptychodus.api.scan import Scan, ScanPoint

from .settings import PtychoPackSettings


class DifferenceMapReconstructor(Reconstructor):

    def __init__(self, settings: PtychoPackSettings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return 'DM'

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        scan_input = parameters.product.scan
        probe_input = parameters.product.probe
        object_input = parameters.product.object_
        object_geometry = object_input.getGeometry()

        detector_data = DetectorData.create_simple(
            torch.from_numpy(parameters.patterns.astype(numpy.int32)))
        probe_power = numpy.max(numpy.sum(parameters.patterns, axis=(-2, -1)))
        positions_px: list[float] = list()

        for scan_point in scan_input:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            positions_px.append(object_point.positionYInPixels)
            positions_px.append(object_point.positionXInPixels)

        product = DataProduct.create_simple(
            positions_px=torch.tensor(positions_px).reshape(len(scan_input), 2),
            probe=torch.tensor(numpy.expand_dims(probe_input.array, axis=0)),
            object_=torch.tensor(object_input.array),
        )
        plan = CorrectionPlan(
            object_correction=CorrectionPlanElement(
                start=self._settings.object_correction_plan_start.value,
                stop=self._settings.object_correction_plan_stop.value,
                stride=self._settings.object_correction_plan_stride.value,
            ),
            probe_correction=CorrectionPlanElement(
                start=self._settings.probe_correction_plan_start.value,
                stop=self._settings.probe_correction_plan_stop.value,
                stride=self._settings.probe_correction_plan_stride.value,
            ),
            position_correction=CorrectionPlanElement(
                start=self._settings.position_correction_plan_start.value,
                stop=self._settings.position_correction_plan_stop.value,
                stride=self._settings.position_correction_plan_stride.value,
            ),
        )

        device = Device('cuda', 0, 'cuda:0')  # TODO
        algorithm = DifferenceMap(device, detector_data, product)
        # FIXME parameters
        algorithm.set_probe_power(probe_power)
        data_error = algorithm.iterate(plan)
        pp_output_product = algorithm.get_product()
        scan_output_points: list[ScanPoint] = list()

        for scan_point_input, (y_px, x_px) in zip(scan_input,
                                                  pp_output_product.positions_px.numpy()):
            object_point = ObjectPoint(scan_point_input.index, float(x_px), float(y_px))
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            scan_output_points.append(scan_point)

        output_product = Product(
            metadata=parameters.product.metadata,
            scan=Scan(scan_output_points),
            probe=Probe(
                array=numpy.squeeze(pp_output_product.probe.numpy()),  # TODO support OPR
                pixelWidthInMeters=probe_input.pixelWidthInMeters,
                pixelHeightInMeters=probe_input.pixelHeightInMeters,
            ),
            object_=Object(
                array=pp_output_product.object_.numpy(),
                layerDistanceInMeters=object_input.layerDistanceInMeters,
                pixelWidthInMeters=object_input.pixelWidthInMeters,
                pixelHeightInMeters=object_input.pixelHeightInMeters,
            ),
            costs=data_error,
        )

        return ReconstructOutput(output_product, 0)
