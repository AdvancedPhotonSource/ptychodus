from importlib.metadata import version
import logging
from typing import Optional, Literal
import copy

from ptychodus.model.product.item import ProductRepositoryItem
from ptychodus.api.scan import Scan
from ptychodus.api.object import Object
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.geometry import ImageExtent
from ..analysis import ObjectLinearInterpolator
from .buffers import ObjectPatchCircularBuffer, PatternCircularBuffer
from .settings import PtychoNNPositionPredictionSettings

import ptychonn.position
import numpy as np

logger = logging.getLogger(__name__)

class PositionPredictionWorker:

    def __init__(self, 
                 positionPredictionSettings: PtychoNNPositionPredictionSettings, 
                 reconInput: ReconstructInput,
                 ) -> None:
        self._settings = positionPredictionSettings
        self._reconInput = reconInput
        self._configs = None
        self._corrector = None
        self._objPatches = None
        self.predictedPositionsPx = None

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PositionPredictor'
    
    def getPixelSizeInMetersFromReconstructInput(self):
        obj = self._reconInput.product.object_
        return (obj.pixelHeightInMeters + obj.pixelWidthInMeters) / 2.0
    
    def getProbePositionsFromReconstructInput(self):
        scanObj = self._reconInput.product.scan
        psizeM = self.getPixelSizeInMetersFromReconstructInput()
        scanArr = np.array([[scanObj[i].positionYInMeters, scanObj[i].positionXInMeters] for i in range(len(scanObj))])
        scanArr = scanArr / psizeM
        probePos = ptychonn.position.ProbePositionList(position_list=scanArr, unit='pixel')
        return probePos
    
    def generateObjectPatches(self) -> None:
        interpolator = ObjectLinearInterpolator(self._reconInput.product.object_)

        patternExtent = self._reconInput.product.probe.getExtent()
        maximumSize = max(1, len(self._reconInput.product.scan))
        objectPatchBuffer = np.zeros([maximumSize, patternExtent.heightInPixels, patternExtent.widthInPixels])

        for i, scanPoint in enumerate(self._reconInput.product.scan):
            objectPatch = interpolator.getPatch(scanPoint, patternExtent)
            # For now we take phase only
            objectPatchArr = np.angle(objectPatch.array[0])
            objectPatchBuffer[i] = objectPatchArr
        self._objPatches = objectPatchBuffer
    
    def createConfigs(self):
        initialProbePositions = self.getProbePositionsFromReconstructInput()
        baselineProbePositions = self.getProbePositionsFromReconstructInput()
        
        if int(self._settings.centralCrop.value) == 0:
            centralCrop = None
        else:
            centralCrop = tuple([int(self._settings.centralCrop.value)] * 2)
            
        registrationParams = ptychonn.position.RegistrationConfig(
            registration_method=self._settings.registrationMethod.value,
            hybrid_registration_tols=tuple(float(a.strip()) for a in self._settings.hybridRegistrationTols.value.split(',')),
            nonhybrid_registration_tol=float(self._settings.nonHybridRegistrationTol.value),
            max_shift = int(self._settings.maxShift.value)
        )
            
        self._configs = ptychonn.position.InferenceConfig(
            reconstruction_images=self._objPatches,
            probe_position_list=initialProbePositions,
            pixel_size_nm=self.getPixelSizeInMetersFromReconstructInput() * 1e9,
            baseline_position_list=baselineProbePositions,
            central_crop=centralCrop,
            method=self._settings.method.value,
            num_neighbors_collective=int(self._settings.numberNeighborsCollective.value),
            offset_estimator_order=int(self._settings.offsetEstimatorOrder.value),
            offset_estimator_beta=float(self._settings.offsetEstimatorBeta.value),
            smooth_constraint_weight=float(self._settings.smoothConstraintWeight.value),
            rectangular_grid=self._settings.rectangularGrid.value,
            random_seed=int(self._settings.randomSeed.value),
            debug=self._settings.debug.value,
            registration_params=registrationParams
        )
        
        logger.info("Position prediction configs:")
        logger.info(self._configs)
        
    def getPredictedPositions(self, unit: Literal['pixel', 'm', 'nm'] = 'pixels') -> np.ndarray:
        """Get the predicted positions as a Numpy array. Returns a 2D array of shape (N, 2),
        each row of which is the (y, x) position in the unit specified. Note that the
        y position comes first, which follows the row-major order.

        :param unit: str. The unit in which the positions should be returned.
        :return: ndarray
        """
        if unit == 'pixel':
            return self.predictedPositionsPx
        else:
            conversionFactorDict = {'m': 1e-9, 'nm': 1.0}
            assert unit in conversionFactorDict.keys()
            return self.predictedPositionsPx * self._configs.pixel_size_nm * conversionFactorDict[unit]
    
    def scanObjToArray(self, scan: Scan):
        arr = [[scan._pointSeq[i].positionYInMeters, scan._pointSeq[i].positionXInMeters] 
               for i in range(len(scan))]
        return np.array(arr)
    
    def build(self) -> None:
        self.generateObjectPatches()
        self.createConfigs()
        
    def run(self) -> None:
        self._corrector = ptychonn.position.core.PtychoNNProbePositionCorrector(self._configs)
        self._corrector.build()
        self._corrector.run()
        
        self.predictedPositionsPx = self._corrector.new_probe_positions.array
        return
