from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtyChiReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChi')
        self._settingsGroup.addObserver(self)

        self.numEpochs = self._settingsGroup.createIntegerParameter('NumEpochs', 100, minimum=1)
        self.batchSize = self._settingsGroup.createIntegerParameter('BatchSize', 100, minimum=1)
        self.batchingMode = self._settingsGroup.createStringParameter('BatchingMode', 'random')
        self.batchStride = self._settingsGroup.createIntegerParameter('BatchStride', 1, minimum=1)
        self.useDoublePrecision = self._settingsGroup.createBooleanParameter(
            'UseDoublePrecision', False
        )
        self.useDevices = self._settingsGroup.createBooleanParameter('UseDevices', True)
        self.useLowMemoryForwardModel = self._settingsGroup.createBooleanParameter(
            'UseLowMemoryForwardModel', False
        )
        self.saveDataOnDevice = self._settingsGroup.createBooleanParameter(
            'SaveDataOnDevice', False
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiObject')
        self._settingsGroup.addObserver(self)

        self.isOptimizable = self._settingsGroup.createBooleanParameter('IsOptimizable', True)
        self.optimizationPlanStart = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.createStringParameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.createRealParameter('StepSize', 1.0, minimum=0.0)

        self.patchInterpolator = self._settingsGroup.createStringParameter(
            'PatchInterpolator', 'FOURIER'
        )

        self.constrainL1Norm = self._settingsGroup.createBooleanParameter('ConstrainL1Norm', False)
        self.constrainL1NormStart = self._settingsGroup.createIntegerParameter(
            'ConstrainL1NormStart', 0, minimum=0
        )
        self.constrainL1NormStop = self._settingsGroup.createIntegerParameter(
            'ConstrainL1NormStop', -1
        )
        self.constrainL1NormStride = self._settingsGroup.createIntegerParameter(
            'ConstrainL1NormStride', 1, minimum=1
        )
        self.constrainL1NormWeight = self._settingsGroup.createRealParameter(
            'ConstrainL1NormWeight', 0.0, minimum=0.0
        )

        self.constrainSmoothness = self._settingsGroup.createBooleanParameter(
            'ConstrainSmoothness', False
        )
        self.constrainSmoothnessStart = self._settingsGroup.createIntegerParameter(
            'ConstrainSmoothnessStart', 0, minimum=0
        )
        self.constrainSmoothnessStop = self._settingsGroup.createIntegerParameter(
            'ConstrainSmoothnessStop', -1
        )
        self.constrainSmoothnessStride = self._settingsGroup.createIntegerParameter(
            'ConstrainSmoothnessStride', 1, minimum=1
        )
        self.constrainSmoothnessAlpha = self._settingsGroup.createRealParameter(
            'ConstrainSmoothnessAlpha', 0.0, minimum=0.0, maximum=1.0 / 8
        )

        self.constrainTotalVariation = self._settingsGroup.createBooleanParameter(
            'ConstrainTotalVariation', False
        )
        self.constrainTotalVariationStart = self._settingsGroup.createIntegerParameter(
            'ConstrainTotalVariationStart', 0, minimum=0
        )
        self.constrainTotalVariationStop = self._settingsGroup.createIntegerParameter(
            'ConstrainTotalVariationStop', -1
        )
        self.constrainTotalVariationStride = self._settingsGroup.createIntegerParameter(
            'ConstrainTotalVariationStride', 1, minimum=1
        )
        self.constrainTotalVariationWeight = self._settingsGroup.createRealParameter(
            'ConstrainTotalVariationWeight', 0.0, minimum=0.0
        )

        self.removeGridArtifacts = self._settingsGroup.createBooleanParameter(
            'RemoveGridArtifacts', False
        )
        self.removeGridArtifactsStart = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsStart', 0, minimum=0
        )
        self.removeGridArtifactsStop = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsStop', -1
        )
        self.removeGridArtifactsStride = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsStride', 1, minimum=1
        )
        self.removeGridArtifactsPeriodXInMeters = self._settingsGroup.createRealParameter(
            'RemoveGridArtifactsPeriodXInMeters', 1e-7, minimum=0.0
        )
        self.removeGridArtifactsPeriodYInMeters = self._settingsGroup.createRealParameter(
            'RemoveGridArtifactsPeriodYInMeters', 1e-7, minimum=0.0
        )
        self.removeGridArtifactsWindowSizeInPixels = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsWindowSizeInPixels',
            5,
            minimum=1,
        )
        self.removeGridArtifactsDirection = self._settingsGroup.createStringParameter(
            'RemoveGridArtifactsDirection', 'XY'
        )

        self.regularizeMultislice = self._settingsGroup.createBooleanParameter(
            'RegularizeMultislice', False
        )
        self.regularizeMultisliceStart = self._settingsGroup.createIntegerParameter(
            'RegularizeMultisliceStart', 0, minimum=0
        )
        self.regularizeMultisliceStop = self._settingsGroup.createIntegerParameter(
            'RegularizeMultisliceStop', -1
        )
        self.regularizeMultisliceStride = self._settingsGroup.createIntegerParameter(
            'RegularizeMultisliceStride', 1, minimum=1
        )
        self.regularizeMultisliceWeight = self._settingsGroup.createRealParameter(
            'RegularizeMultisliceWeight', 0.0, minimum=0.0
        )
        self.regularizeMultisliceUnwrapPhase = self._settingsGroup.createBooleanParameter(
            'RegularizeMultisliceUnwrapPhase', True
        )
        self.regularizeMultisliceUnwrapPhaseImageGradientMethod = (
            self._settingsGroup.createStringParameter(
                'RegularizeMultisliceUnwrapPhaseImageGradientMethod', 'FOURIER_SHIFT'
            )
        )
        self.regularizeMultisliceUnwrapPhaseImageIntegrationMethod = (
            self._settingsGroup.createStringParameter(
                'RegularizeMultisliceUnwrapPhaseImageIntegrationMethod', 'DECONVOLUTION'
            )
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiProbe')
        self._settingsGroup.addObserver(self)

        self.isOptimizable = self._settingsGroup.createBooleanParameter('IsOptimizable', True)
        self.optimizationPlanStart = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.createStringParameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.createRealParameter('StepSize', 1.0, minimum=0.0)

        self.constrainProbePower = self._settingsGroup.createBooleanParameter(
            'ConstrainProbePower', False
        )
        self.constrainProbePowerStart = self._settingsGroup.createIntegerParameter(
            'ConstrainProbePowerStart', 0, minimum=0
        )
        self.constrainProbePowerStop = self._settingsGroup.createIntegerParameter(
            'ConstrainProbePowerStop', -1
        )
        self.constrainProbePowerStride = self._settingsGroup.createIntegerParameter(
            'ConstrainProbePowerStride', 1, minimum=1
        )

        self.orthogonalizeIncoherentModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.orthogonalizeIncoherentModesStart = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeIncoherentModesStart', 0, minimum=0
        )
        self.orthogonalizeIncoherentModesStop = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeIncoherentModesStop', -1
        )
        self.orthogonalizeIncoherentModesStride = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeIncoherentModesStride', 1, minimum=1
        )
        self.orthogonalizeIncoherentModesMethod = self._settingsGroup.createStringParameter(
            'OrthogonalizeIncoherentModesMethod', 'GS'
        )

        self.orthogonalizeOPRModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeOPRModes', True
        )
        self.orthogonalizeOPRModesStart = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeOPRModesStart', 0, minimum=0
        )
        self.orthogonalizeOPRModesStop = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeOPRModesStop', -1
        )
        self.orthogonalizeOPRModesStride = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeOPRModesStride', 1, minimum=1
        )

        self.constrainSupport = self._settingsGroup.createBooleanParameter(
            'ConstrainSupport', False
        )
        self.constrainSupportStart = self._settingsGroup.createIntegerParameter(
            'ConstrainSupportStart', 0, minimum=0
        )
        self.constrainSupportStop = self._settingsGroup.createIntegerParameter(
            'ConstrainSupportStop', -1
        )
        self.constrainSupportStride = self._settingsGroup.createIntegerParameter(
            'ConstrainSupportStride', 1, minimum=1
        )
        self.constrainSupportThreshold = self._settingsGroup.createRealParameter(
            'ConstrainSupportThreshold', 0.005, minimum=0.0
        )

        self.constrainCenter = self._settingsGroup.createBooleanParameter('ConstrainCenter', False)
        self.constrainCenterStart = self._settingsGroup.createIntegerParameter(
            'ConstrainCenterStart', 0, minimum=0
        )
        self.constrainCenterStop = self._settingsGroup.createIntegerParameter(
            'ConstrainCenterStop', -1
        )
        self.constrainCenterStride = self._settingsGroup.createIntegerParameter(
            'ConstrainCenterStride', 1, minimum=1
        )

        self.relaxEigenmodeUpdate = self._settingsGroup.createRealParameter(
            'RelaxEigenmodeUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiProbePositionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiProbePosition')
        self._settingsGroup.addObserver(self)

        self.isOptimizable = self._settingsGroup.createBooleanParameter('IsOptimizable', False)
        self.optimizationPlanStart = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.createStringParameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.createRealParameter('StepSize', 1.0, minimum=0.0)

        self.positionCorrectionType = self._settingsGroup.createStringParameter(
            'PositionCorrectionType', 'Gradient'
        )
        self.crossCorrelationScale = self._settingsGroup.createIntegerParameter(
            'CrossCorrelationScale', 20000, minimum=1
        )
        self.crossCorrelationRealSpaceWidth = self._settingsGroup.createRealParameter(
            'CrossCorrelationRealSpaceWidth', 0.01, minimum=0.0
        )
        self.crossCorrelationProbeThreshold = self._settingsGroup.createRealParameter(
            'CrossCorrelationProbeThreshold', 0.1, minimum=0.0, maximum=1.0
        )

        self.limitMagnitudeUpdate = self._settingsGroup.createBooleanParameter(
            'LimitMagnitudeUpdate', False
        )
        self.limitMagnitudeUpdateStart = self._settingsGroup.createIntegerParameter(
            'LimitMagnitudeUpdateStart', 0, minimum=0
        )
        self.limitMagnitudeUpdateStop = self._settingsGroup.createIntegerParameter(
            'LimitMagnitudeUpdateStop', -1
        )
        self.limitMagnitudeUpdateStride = self._settingsGroup.createIntegerParameter(
            'LimitMagnitudeUpdateStride', 1, minimum=1
        )
        self.magnitudeUpdateLimit = self._settingsGroup.createRealParameter(
            'MagnitudeUpdateLimit', 0.0, minimum=0.0
        )

        self.constrainCentroid = self._settingsGroup.createBooleanParameter(
            'ConstrainCentroid', False
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiOPRSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiOPR')
        self._settingsGroup.addObserver(self)

        self.isOptimizable = self._settingsGroup.createBooleanParameter('IsOptimizable', False)
        self.optimizationPlanStart = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.createIntegerParameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.createStringParameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.createRealParameter('StepSize', 1.0, minimum=0.0)

        self.optimizeIntensities = self._settingsGroup.createBooleanParameter(
            'OptimizeIntensities', False
        )
        self.optimizeEigenmodeWeights = self._settingsGroup.createBooleanParameter(
            'OptimizeEigenmodeWeigts', True
        )

        self.smoothModeWeights = self._settingsGroup.createBooleanParameter(
            'SmoothModeWeights', False
        )
        self.smoothModeWeightsStart = self._settingsGroup.createIntegerParameter(
            'SmoothModeWeightsStart', 0, minimum=0
        )
        self.smoothModeWeightsStop = self._settingsGroup.createIntegerParameter(
            'SmoothModeWeightsStop', -1
        )
        self.smoothModeWeightsStride = self._settingsGroup.createIntegerParameter(
            'SmoothModeWeightsStride', 1, minimum=1
        )
        self.smoothingMethod = self._settingsGroup.createStringParameter('SmoothingMethod', '')
        self.polynomialSmoothingDegree = self._settingsGroup.createIntegerParameter(
            'PolynomialSmoothingDegree', 4, minimum=0, maximum=10
        )

        self.relaxUpdate = self._settingsGroup.createRealParameter(
            'RelaxUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiAutodiffSettings(Observable, Observer):  # FIXME to view
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiAutodiff')
        self._settingsGroup.addObserver(self)

        self.lossFunction = self._settingsGroup.createStringParameter('LossFunction', 'MSE_SQRT')
        self.forwardModelClass = self._settingsGroup.createStringParameter(
            'ForwardModelClass', 'PLANAR_PTYCHOGRAPHY'
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiDMSettings(Observable, Observer):  # FIXME to view
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiDM')
        self._settingsGroup.addObserver(self)

        self.exitWaveUpdateRelaxation = self._settingsGroup.createRealParameter(
            'ExitWaveUpdateRelaxation', 1.0, minimum=0.0, maximum=1.0
        )
        self.chunkLength = self._settingsGroup.createIntegerParameter('ChunkLength', 1, minimum=1)
        self.objectAmplitudeClampLimit = self._settingsGroup.createRealParameter(
            'ObjectAmplitudeClampLimit', 1000, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiLSQMLSettings(Observable, Observer):  # FIXME to view
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiLSQML')
        self._settingsGroup.addObserver(self)

        self.noiseModel = self._settingsGroup.createStringParameter('NoiseModel', 'GAUSSIAN')
        self.gaussianNoiseDeviation = self._settingsGroup.createRealParameter(
            'GaussianNoiseDeviation', 0.5
        )
        self.solveObjectProbeStepSizeJointlyForFirstSliceInMultislice = (
            self._settingsGroup.createBooleanParameter(
                'SolveObjectProbeStepSizeJointlyForFirstSliceInMultislice', False
            )
        )
        self.solveStepSizesOnlyUsingFirstProbeMode = self._settingsGroup.createBooleanParameter(
            'SolveStepSizesOnlyUsingFirstProbeMode', False
        )
        self.momentumAccelerationGain = self._settingsGroup.createRealParameter(
            'MomentumAccelerationGain', 0.0, minimum=0.0
        )
        self.useMomentumAccelerationGradientMixingFactor = (
            self._settingsGroup.createBooleanParameter(
                'UseMomentumAccelerationGradientMixingFactor', False
            )
        )
        self.momentumAccelerationGradientMixingFactor = self._settingsGroup.createRealParameter(
            'MomentumAccelerationGradientMixingFactor', 1.0
        )

        self.probeOptimalStepSizeScaler = self._settingsGroup.createRealParameter(
            'ProbeOptimalStepSizeScaler', 0.9
        )
        self.objectOptimalStepSizeScaler = self._settingsGroup.createRealParameter(
            'ObjectOptimalStepSizeScaler', 0.9, minimum=0.0
        )
        self.objectMultimodalUpdate = self._settingsGroup.createBooleanParameter(
            'ObjectMultimodalUpdate', True
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtyChiPIESettings(Observable, Observer):  # FIXME to view
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChiPIE')
        self._settingsGroup.addObserver(self)

        self.probeAlpha = self._settingsGroup.createRealParameter(
            'ProbeAlpha', 0.1, minimum=0.0, maximum=1.0
        )
        self.objectAlpha = self._settingsGroup.createRealParameter(
            'ObjectAlpha', 0.1, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
