import numpy
import torch
from ptychopack import CorrectionPlan, DataProduct, DetectorData, PtychographicIterativeEngine

from ptychodus.api.object import ObjectPoint
from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ptychodus.api.scan import ScanPoint


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
        max_pattern_counts = 0

        for pattern in parameters.patterns:
            pattern_counts = numpy.sum(pattern)

            if max_pattern_counts < pattern_counts:
                max_pattern_counts = pattern_counts

        probe_rescale = max_pattern_counts / probe_counts
        print(f'rescaling probe {probe_counts} -> {max_pattern_counts}')
        probe = probe_input.array * probe_rescale  # FIXME remove when able

        detector_data = DetectorData.create_simple(
            torch.tensor(parameters.patterns.astype(numpy.int32)))
        positions_px: list[float] = list()

        for scan_point in scan_input:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            positions_px.append(object_point.positionYInPixels)
            positions_px.append(object_point.positionXInPixels)

        product = DataProduct.create_simple(
            positions_px=torch.reshape(torch.tensor(positions_px), (len(scan_input), 2)),
            probe=torch.tensor(numpy.expand_dims(probe_input.array, axis=0)),
            object_=torch.tensor(parameters.product.object_.array),
        )
        num_iterations = 1
        plan = CorrectionPlan.create_simple(num_iterations,
                                            correct_object=True,
                                            correct_probe=True,
                                            correct_positions=False)

        algorithm = PtychographicIterativeEngine(detector_data, product)
        data_error = algorithm.iterate(plan)
        pp_output_product = algorithm.get_product()
        scan_output_points: list[ScanPoint] = list()

        for scan_point_input, (y_px, x_px) in zip(scan_input, pp_output_product.positions_px):
            object_point = ObjectPoint(scan_point_input.index, float(x_px), float(y_px))
            scan_point = object_geometry.mapObjectPointToScanPoint(objectPoint)
            scan_output_points.append(scan_point)

        output_product = Product(
            metadata=parameters.product.metadata,
            scan=Scan(scan_output_points),
            probe=Probe(
                array=pp_output_product.probe,
                pixelWidthInMeters=probe_input.pixelWidthInMeters,
                pixelHeightInMeters=probe_input.pixelHeightInMeters,
            ),
            object_=Object(
                array=pp_output_product.object_,
                layerDistanceInMeters=object_input.layerDistanceInMeters,
                pixelWidthInMeters=object_input.pixelWidthInMeters,
                pixelHeightInMeters=object_input.pixelHeightInMeters,
            ),
            costs=data_error,
        )

        return ReconstructOutput(output_product, 0)
