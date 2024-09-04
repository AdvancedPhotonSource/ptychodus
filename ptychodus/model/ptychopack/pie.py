import numpy
import torch
from ptychopack import CorrectionPlan, DataProduct, DetectorData, PtychographicIterativeEngine

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ptychodus.api.scan import Scan, ScanPoint


class PtychographicIterativeEngineReconstructor(Reconstructor):

    @property
    def name(self) -> str:
        return 'PIE'

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        scan_input = parameters.product.scan
        probe_input = parameters.product.probe
        probe_counts = numpy.sum(probe_input.getIntensity())
        object_input = parameters.product.object_
        object_geometry = object_input.getGeometry()

        target_pattern_counts = numpy.mean(numpy.sum(parameters.patterns, axis=(-2, -1)))
        probe_rescale = numpy.sqrt(target_pattern_counts / probe_counts)
        print(f'rescaling probe {probe_counts} -> {probe_rescale * probe_counts}')
        probe = probe_input.array * probe_rescale  # FIXME remove when able

        detector_data = DetectorData.create_simple(
            torch.from_numpy(parameters.patterns.astype(numpy.int32)))
        positions_px: list[float] = list()

        for scan_point in scan_input:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            positions_px.append(object_point.positionYInPixels)
            positions_px.append(object_point.positionXInPixels)

        product = DataProduct.create_simple(
            positions_px=torch.reshape(torch.tensor(positions_px), (len(scan_input), 2)),
            probe=torch.tensor(numpy.expand_dims(probe, axis=0)),
            object_=torch.tensor(parameters.product.object_.array),
        )
        num_iterations = 10
        plan = CorrectionPlan.create_simple(
            num_iterations,
            correct_object=True,
            correct_probe=False,
            correct_positions=False,
        )

        device = 'cpu'  # TODO
        algorithm = PtychographicIterativeEngine(device, detector_data, product)
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
