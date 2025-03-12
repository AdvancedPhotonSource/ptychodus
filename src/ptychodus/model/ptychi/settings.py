from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtyChiReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChi')
        self._settingsGroup.add_observer(self)

        self.numEpochs = self._settingsGroup.create_integer_parameter('NumEpochs', 100, minimum=1)
        self.batchSize = self._settingsGroup.create_integer_parameter('BatchSize', 100, minimum=1)
        self.batchingMode = self._settingsGroup.create_string_parameter('BatchingMode', 'random')
        self.compactModeUpdateClustering = self._settingsGroup.create_integer_parameter(
            'CompactModeUpdateClustering', 1, minimum=0
        )
        self.useDoublePrecision = self._settingsGroup.create_boolean_parameter(
            'UseDoublePrecision', False
        )
        self.useDevices = self._settingsGroup.create_boolean_parameter('UseDevices', True)
        self.useLowMemoryMode = self._settingsGroup.create_boolean_parameter(
            'UseLowMemoryMode', False
        )
        self.padForShift = self._settingsGroup.create_integer_parameter('PadForShift', 0, minimum=0)

        self.useFarFieldPropagation = self._settingsGroup.create_boolean_parameter(
            'UseFarFieldPropagation', True
        )
        self.fftShiftDiffractionPatterns = self._settingsGroup.create_boolean_parameter(
            'FFTShiftDiffractionPatterns', True
        )
        self.saveDataOnDevice = self._settingsGroup.create_boolean_parameter(
            'SaveDataOnDevice', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiObject')
        self._settingsGroup.add_observer(self)

        self.isOptimizable = self._settingsGroup.create_boolean_parameter('IsOptimizable', True)
        self.optimizationPlanStart = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.create_string_parameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.patchInterpolator = self._settingsGroup.create_string_parameter(
            'PatchInterpolator', 'FOURIER'
        )

        self.constrainL1Norm = self._settingsGroup.create_boolean_parameter(
            'ConstrainL1Norm', False
        )
        self.constrainL1NormStart = self._settingsGroup.create_integer_parameter(
            'ConstrainL1NormStart', 0, minimum=0
        )
        self.constrainL1NormStop = self._settingsGroup.create_integer_parameter(
            'ConstrainL1NormStop', -1
        )
        self.constrainL1NormStride = self._settingsGroup.create_integer_parameter(
            'ConstrainL1NormStride', 1, minimum=1
        )
        self.constrainL1NormWeight = self._settingsGroup.create_real_parameter(
            'ConstrainL1NormWeight', 0.0, minimum=0.0
        )

        self.constrainSmoothness = self._settingsGroup.create_boolean_parameter(
            'ConstrainSmoothness', False
        )
        self.constrainSmoothnessStart = self._settingsGroup.create_integer_parameter(
            'ConstrainSmoothnessStart', 0, minimum=0
        )
        self.constrainSmoothnessStop = self._settingsGroup.create_integer_parameter(
            'ConstrainSmoothnessStop', -1
        )
        self.constrainSmoothnessStride = self._settingsGroup.create_integer_parameter(
            'ConstrainSmoothnessStride', 1, minimum=1
        )
        self.constrainSmoothnessAlpha = self._settingsGroup.create_real_parameter(
            'ConstrainSmoothnessAlpha', 0.0, minimum=0.0, maximum=1.0 / 8
        )

        self.constrainTotalVariation = self._settingsGroup.create_boolean_parameter(
            'ConstrainTotalVariation', False
        )
        self.constrainTotalVariationStart = self._settingsGroup.create_integer_parameter(
            'ConstrainTotalVariationStart', 0, minimum=0
        )
        self.constrainTotalVariationStop = self._settingsGroup.create_integer_parameter(
            'ConstrainTotalVariationStop', -1
        )
        self.constrainTotalVariationStride = self._settingsGroup.create_integer_parameter(
            'ConstrainTotalVariationStride', 1, minimum=1
        )
        self.constrainTotalVariationWeight = self._settingsGroup.create_real_parameter(
            'ConstrainTotalVariationWeight', 0.0, minimum=0.0
        )

        self.removeGridArtifacts = self._settingsGroup.create_boolean_parameter(
            'RemoveGridArtifacts', False
        )
        self.removeGridArtifactsStart = self._settingsGroup.create_integer_parameter(
            'RemoveGridArtifactsStart', 0, minimum=0
        )
        self.removeGridArtifactsStop = self._settingsGroup.create_integer_parameter(
            'RemoveGridArtifactsStop', -1
        )
        self.removeGridArtifactsStride = self._settingsGroup.create_integer_parameter(
            'RemoveGridArtifactsStride', 1, minimum=1
        )
        self.removeGridArtifactsPeriodXInMeters = self._settingsGroup.create_real_parameter(
            'RemoveGridArtifactsPeriodXInMeters', 1e-7, minimum=0.0
        )
        self.removeGridArtifactsPeriodYInMeters = self._settingsGroup.create_real_parameter(
            'RemoveGridArtifactsPeriodYInMeters', 1e-7, minimum=0.0
        )
        self.removeGridArtifactsWindowSizeInPixels = self._settingsGroup.create_integer_parameter(
            'RemoveGridArtifactsWindowSizeInPixels',
            5,
            minimum=1,
        )
        self.removeGridArtifactsDirection = self._settingsGroup.create_string_parameter(
            'RemoveGridArtifactsDirection', 'XY'
        )

        self.regularizeMultislice = self._settingsGroup.create_boolean_parameter(
            'RegularizeMultislice', False
        )
        self.regularizeMultisliceStart = self._settingsGroup.create_integer_parameter(
            'RegularizeMultisliceStart', 0, minimum=0
        )
        self.regularizeMultisliceStop = self._settingsGroup.create_integer_parameter(
            'RegularizeMultisliceStop', -1
        )
        self.regularizeMultisliceStride = self._settingsGroup.create_integer_parameter(
            'RegularizeMultisliceStride', 1, minimum=1
        )
        self.regularizeMultisliceWeight = self._settingsGroup.create_real_parameter(
            'RegularizeMultisliceWeight', 0.0, minimum=0.0
        )
        self.regularizeMultisliceUnwrapPhase = self._settingsGroup.create_boolean_parameter(
            'RegularizeMultisliceUnwrapPhase', True
        )
        self.regularizeMultisliceUnwrapPhaseImageGradientMethod = (
            self._settingsGroup.create_string_parameter(
                'RegularizeMultisliceUnwrapPhaseImageGradientMethod', 'FOURIER_SHIFT'
            )
        )
        self.regularizeMultisliceUnwrapPhaseImageIntegrationMethod = (
            self._settingsGroup.create_string_parameter(
                'RegularizeMultisliceUnwrapPhaseImageIntegrationMethod', 'DECONVOLUTION'
            )
        )
        self.removeObjectProbeAmbiguity = self._settingsGroup.create_boolean_parameter(
            'RemoveObjectProbeAmbiguity', True
        )
        self.removeObjectProbeAmbiguityStart = self._settingsGroup.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStart', 0, minimum=0
        )
        self.removeObjectProbeAmbiguityStop = self._settingsGroup.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStop', -1
        )
        self.removeObjectProbeAmbiguityStride = self._settingsGroup.create_integer_parameter(
            'RemoveObjectProbeAmbiguityStride', 10, minimum=1
        )
        self.buildPreconditionerWithAllModes = self._settingsGroup.create_boolean_parameter(
            'BuildPreconditionerWithAllModes', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiProbe')
        self._settingsGroup.add_observer(self)

        self.isOptimizable = self._settingsGroup.create_boolean_parameter('IsOptimizable', True)
        self.optimizationPlanStart = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.create_string_parameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.constrainProbePower = self._settingsGroup.create_boolean_parameter(
            'ConstrainProbePower', False
        )
        self.constrainProbePowerStart = self._settingsGroup.create_integer_parameter(
            'ConstrainProbePowerStart', 0, minimum=0
        )
        self.constrainProbePowerStop = self._settingsGroup.create_integer_parameter(
            'ConstrainProbePowerStop', -1
        )
        self.constrainProbePowerStride = self._settingsGroup.create_integer_parameter(
            'ConstrainProbePowerStride', 1, minimum=1
        )

        self.orthogonalizeIncoherentModes = self._settingsGroup.create_boolean_parameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.orthogonalizeIncoherentModesStart = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeIncoherentModesStart', 0, minimum=0
        )
        self.orthogonalizeIncoherentModesStop = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeIncoherentModesStop', -1
        )
        self.orthogonalizeIncoherentModesStride = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeIncoherentModesStride', 1, minimum=1
        )
        self.orthogonalizeIncoherentModesMethod = self._settingsGroup.create_string_parameter(
            'OrthogonalizeIncoherentModesMethod', 'GS'
        )

        self.orthogonalizeOPRModes = self._settingsGroup.create_boolean_parameter(
            'OrthogonalizeOPRModes', True
        )
        self.orthogonalizeOPRModesStart = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeOPRModesStart', 0, minimum=0
        )
        self.orthogonalizeOPRModesStop = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeOPRModesStop', -1
        )
        self.orthogonalizeOPRModesStride = self._settingsGroup.create_integer_parameter(
            'OrthogonalizeOPRModesStride', 1, minimum=1
        )

        self.constrainSupport = self._settingsGroup.create_boolean_parameter(
            'ConstrainSupport', False
        )
        self.constrainSupportStart = self._settingsGroup.create_integer_parameter(
            'ConstrainSupportStart', 0, minimum=0
        )
        self.constrainSupportStop = self._settingsGroup.create_integer_parameter(
            'ConstrainSupportStop', -1
        )
        self.constrainSupportStride = self._settingsGroup.create_integer_parameter(
            'ConstrainSupportStride', 1, minimum=1
        )
        self.constrainSupportThreshold = self._settingsGroup.create_real_parameter(
            'ConstrainSupportThreshold', 0.005, minimum=0.0
        )

        self.constrainCenter = self._settingsGroup.create_boolean_parameter(
            'ConstrainCenter', False
        )
        self.constrainCenterStart = self._settingsGroup.create_integer_parameter(
            'ConstrainCenterStart', 0, minimum=0
        )
        self.constrainCenterStop = self._settingsGroup.create_integer_parameter(
            'ConstrainCenterStop', -1
        )
        self.constrainCenterStride = self._settingsGroup.create_integer_parameter(
            'ConstrainCenterStride', 1, minimum=1
        )

        self.relaxEigenmodeUpdate = self._settingsGroup.create_real_parameter(
            'RelaxEigenmodeUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiProbePositionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiProbePosition')
        self._settingsGroup.add_observer(self)

        self.isOptimizable = self._settingsGroup.create_boolean_parameter('IsOptimizable', False)
        self.optimizationPlanStart = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.create_string_parameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.positionCorrectionType = self._settingsGroup.create_string_parameter(
            'PositionCorrectionType', 'Gradient'
        )
        self.crossCorrelationScale = self._settingsGroup.create_integer_parameter(
            'CrossCorrelationScale', 20000, minimum=1
        )
        self.crossCorrelationRealSpaceWidth = self._settingsGroup.create_real_parameter(
            'CrossCorrelationRealSpaceWidth', 0.01, minimum=0.0
        )
        self.crossCorrelationProbeThreshold = self._settingsGroup.create_real_parameter(
            'CrossCorrelationProbeThreshold', 0.1, minimum=0.0, maximum=1.0
        )

        self.limitMagnitudeUpdate = self._settingsGroup.create_boolean_parameter(
            'LimitMagnitudeUpdate', False
        )
        self.limitMagnitudeUpdateStart = self._settingsGroup.create_integer_parameter(
            'LimitMagnitudeUpdateStart', 0, minimum=0
        )
        self.limitMagnitudeUpdateStop = self._settingsGroup.create_integer_parameter(
            'LimitMagnitudeUpdateStop', -1
        )
        self.limitMagnitudeUpdateStride = self._settingsGroup.create_integer_parameter(
            'LimitMagnitudeUpdateStride', 1, minimum=1
        )
        self.magnitudeUpdateLimit = self._settingsGroup.create_real_parameter(
            'MagnitudeUpdateLimit', 0.0, minimum=0.0
        )

        self.constrainCentroid = self._settingsGroup.create_boolean_parameter(
            'ConstrainCentroid', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiOPRSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiOPR')
        self._settingsGroup.add_observer(self)

        self.isOptimizable = self._settingsGroup.create_boolean_parameter('IsOptimizable', False)
        self.optimizationPlanStart = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStart', 0, minimum=0
        )
        self.optimizationPlanStop = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStop', -1
        )
        self.optimizationPlanStride = self._settingsGroup.create_integer_parameter(
            'OptimizationPlanStride', 1, minimum=1
        )
        self.optimizer = self._settingsGroup.create_string_parameter('Optimizer', 'SGD')
        self.stepSize = self._settingsGroup.create_real_parameter('StepSize', 1.0, minimum=0.0)

        self.optimizeIntensities = self._settingsGroup.create_boolean_parameter(
            'OptimizeIntensities', False
        )
        self.optimizeEigenmodeWeights = self._settingsGroup.create_boolean_parameter(
            'OptimizeEigenmodeWeigts', True
        )

        self.smoothModeWeights = self._settingsGroup.create_boolean_parameter(
            'SmoothModeWeights', False
        )
        self.smoothModeWeightsStart = self._settingsGroup.create_integer_parameter(
            'SmoothModeWeightsStart', 0, minimum=0
        )
        self.smoothModeWeightsStop = self._settingsGroup.create_integer_parameter(
            'SmoothModeWeightsStop', -1
        )
        self.smoothModeWeightsStride = self._settingsGroup.create_integer_parameter(
            'SmoothModeWeightsStride', 1, minimum=1
        )
        self.smoothingMethod = self._settingsGroup.create_string_parameter('SmoothingMethod', '')
        self.polynomialSmoothingDegree = self._settingsGroup.create_integer_parameter(
            'PolynomialSmoothingDegree', 4, minimum=0, maximum=10
        )

        self.relaxUpdate = self._settingsGroup.create_real_parameter(
            'RelaxUpdate', 1.0, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiAutodiffSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiAutodiff')
        self._settingsGroup.add_observer(self)

        self.lossFunction = self._settingsGroup.create_string_parameter('LossFunction', 'MSE_SQRT')
        self.forwardModelClass = self._settingsGroup.create_string_parameter(
            'ForwardModelClass', 'PLANAR_PTYCHOGRAPHY'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiDMSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiDM')
        self._settingsGroup.add_observer(self)

        self.exitWaveUpdateRelaxation = self._settingsGroup.create_real_parameter(
            'ExitWaveUpdateRelaxation', 1.0, minimum=0.0, maximum=1.0
        )
        self.chunkLength = self._settingsGroup.create_integer_parameter('ChunkLength', 1, minimum=1)
        self.objectAmplitudeClampLimit = self._settingsGroup.create_real_parameter(
            'ObjectAmplitudeClampLimit', 1000, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiLSQMLSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiLSQML')
        self._settingsGroup.add_observer(self)

        self.noiseModel = self._settingsGroup.create_string_parameter('NoiseModel', 'GAUSSIAN')
        self.gaussianNoiseDeviation = self._settingsGroup.create_real_parameter(
            'GaussianNoiseDeviation', 0.5
        )
        self.solveObjectProbeStepSizeJointlyForFirstSliceInMultislice = (
            self._settingsGroup.create_boolean_parameter(
                'SolveObjectProbeStepSizeJointlyForFirstSliceInMultislice', False
            )
        )
        self.solveStepSizesOnlyUsingFirstProbeMode = self._settingsGroup.create_boolean_parameter(
            'SolveStepSizesOnlyUsingFirstProbeMode', False
        )
        self.momentumAccelerationGain = self._settingsGroup.create_real_parameter(
            'MomentumAccelerationGain', 0.0, minimum=0.0
        )
        self.useMomentumAccelerationGradientMixingFactor = (
            self._settingsGroup.create_boolean_parameter(
                'UseMomentumAccelerationGradientMixingFactor', False
            )
        )
        self.momentumAccelerationGradientMixingFactor = self._settingsGroup.create_real_parameter(
            'MomentumAccelerationGradientMixingFactor', 1.0
        )

        self.probeOptimalStepSizeScaler = self._settingsGroup.create_real_parameter(
            'ProbeOptimalStepSizeScaler', 0.9, minimum=0.0
        )
        self.objectOptimalStepSizeScaler = self._settingsGroup.create_real_parameter(
            'ObjectOptimalStepSizeScaler', 0.9, minimum=0.0
        )
        self.objectMultimodalUpdate = self._settingsGroup.create_boolean_parameter(
            'ObjectMultimodalUpdate', True
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtyChiPIESettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtyChiPIE')
        self._settingsGroup.add_observer(self)

        self.probeAlpha = self._settingsGroup.create_real_parameter(
            'ProbeAlpha', 0.1, minimum=0.0, maximum=1.0
        )
        self.objectAlpha = self._settingsGroup.create_real_parameter(
            'ObjectAlpha', 0.1, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
