import numpy
import torch
from ptychopack import CorrectionPlan, DataProduct, DetectorData, PtychographicIterativeEngine

from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput


class PtychographicIterativeEngineReconstructor(Reconstructor):

    @property
    def name(self) -> str:
        return 'PIE'

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        scan_input = parameters.product.scan
        probe_input = parameters.product.probe
        object_input = parameters.product.object_
        object_geometry = object_input.getGeometry()

        # FIXME rescale probe

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
        num_iterations = 100
        plan = CorrectionPlan.create_simple(num_iterations,
                                            correct_object=True,
                                            correct_probe=True,
                                            correct_positions=False)

        algorithm = PtychographicIterativeEngine(detector_data, product)
        data_error = algorithm.iterate(plan)
        product = algorithm.get_product()
        print(product)  # FIXME ptychopack product -> ptychodus product
        return ReconstructOutput(parameters.product, 0)
