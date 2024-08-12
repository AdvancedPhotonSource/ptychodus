from importlib.metadata import version
import logging

from .settings import PtychoNNPositionPredictionSettings

import ptychonn.position

logger = logging.getLogger(__name__)

class PositionPredictionWorker:

    def __init__(self, positionPredictionSettings: PtychoNNPositionPredictionSettings) -> None:
        self._settings = positionPredictionSettings
        self._configs = None
        self._corrector = None
        self.predicted_positions_px = None

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PositionPredictor'
    
    def createConfigs(self):
        if str(self._settings.probePositionListPath.value) != '.':
            probePositions = ptychonn.position.ProbePositionList(file_path=self._settings.probePositionListPath.value)
        else:
            probePositions = None
        
        if str(self._settings.baselinePositionListPath.value) != '.':
            baselinePositions = ptychonn.position.ProbePositionList(file_path=self._settings.baselinePositionListPath.value)
        else:
            baselinePositions = None
        
        if int(self._settings.centralCrop.value) == 0:
            centralCrop = None
        else:
            centralCrop = tuple([int(self._settings.centralCrop.value)] * 2)
            
        registration_params = ptychonn.position.RegistrationConfig(
            registration_method=self._settings.registrationMethod.value,
            hybrid_registration_tols=tuple(float(a.strip()) for a in self._settings.hybridRegistrationTols.value.split(',')),
            nonhybrid_registration_tol=float(self._settings.nonHybridRegistrationTol.value),
            max_shift = int(self._settings.maxShift.value)
        )
            
        self._configs = ptychonn.position.InferenceConfig(
            reconstruction_image_path=self._settings.reconstructorImagePath.value,
            probe_position_list=probePositions,
            probe_position_data_unit=self._settings.probePositionDataUnit.value,
            pixel_size_nm=float(self._settings.pixelSizeNM.value),
            baseline_position_list=baselinePositions,
            central_crop=centralCrop,
            method=self._settings.method.value,
            num_neighbors_collective=int(self._settings.numberNeighborsCollective.value),
            offset_estimator_order=int(self._settings.offsetEstimatorOrder.value),
            offset_estimator_beta=float(self._settings.offsetEstimatorBeta.value),
            smooth_constraint_weight=float(self._settings.smoothConstraintWeight.value),
            rectangular_grid=self._settings.rectangularGrid.value,
            random_seed=int(self._settings.randomSeed.value),
            debug=self._settings.debug.value,
            registration_params=registration_params
        )
        
        logger.info("Position prediction configs:")
        logger.info(self._configs)
        
    def getPredictedPositions(self):
        return self.predicted_positions_px
        
    def run(self):
        self.createConfigs()
        self._corrector = ptychonn.position.core.PtychoNNProbePositionCorrector(self._configs)
        self._corrector.build()
        self._corrector.run()
        
        self.predicted_positions_px = self._corrector.new_probe_positions.array
        return
