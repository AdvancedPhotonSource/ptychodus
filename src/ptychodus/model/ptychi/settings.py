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
        self.constrainL1NormWeight = self._settingsGroup.createRealParameter(
            'ConstrainL1NormWeight', 0.0, minimum=0.0
        )
        self.constrainL1NormStride = self._settingsGroup.createIntegerParameter(
            'ConstrainL1NormStride', 1, minimum=1
        )

        self.constrainSmoothness = self._settingsGroup.createBooleanParameter(
            'ConstrainSmoothness', False
        )
        self.constrainSmoothnessAlpha = self._settingsGroup.createRealParameter(
            'ConstrainSmoothnessAlpha', 0.0, minimum=0.0, maximum=1.0 / 8
        )
        self.constrainSmoothnessStride = self._settingsGroup.createIntegerParameter(
            'ConstrainSmoothnessStride', 1, minimum=1
        )

        self.constrainTotalVariation = self._settingsGroup.createBooleanParameter(
            'ConstrainTotalVariation', False
        )
        self.constrainTotalVariationWeight = self._settingsGroup.createRealParameter(
            'ConstrainTotalVariationWeight', 0.0, minimum=0.0
        )
        self.constrainTotalVariationStride = self._settingsGroup.createIntegerParameter(
            'ConstrainTotalVariationStride', 1, minimum=1
        )
        self.removeGridArtifacts = self._settingsGroup.createBooleanParameter(
            'RemoveGridArtifacts', False
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
        self.removeGridArtifactsStride = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsStride', 1, minimum=1
        )
        self.regularizeMultislice = self._settingsGroup.createBooleanParameter(
            'RegularizeMultislice', False
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
        self.regularizeMultisliceStride = self._settingsGroup.createIntegerParameter(
            'RegularizeMultisliceStride', 1, minimum=1
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
        self.probePower = self._settingsGroup.createRealParameter('ProbePower', 0.0, minimum=0.0)
        self.constrainProbePowerStride = self._settingsGroup.createIntegerParameter(
            'ConstrainProbePowerStride', 1, minimum=1
        )

        self.orthogonalizeIncoherentModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.orthogonalizeIncoherentModesMethod = self._settingsGroup.createStringParameter(
            'OrthogonalizeIncoherentModesMethod', 'GS'
        )
        self.orthogonalizeIncoherentModesStride = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeIncoherentModesStride', 1, minimum=1
        )

        self.orthogonalizeOPRModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeOPRModes', True
        )
        self.orthogonalizeOPRModesStride = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeOPRModesStride', 1, minimum=1
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

        # vvv FIXME vvv
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
            'CrossCorrelationProbeThreshold', 0.1, minimum=0.0
        )
        # ^^^ FIXME ^^^

        self.limitMagnitudeUpdate = self._settingsGroup.createBooleanParameter(
            'LimitMagnitudeUpdate', False
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

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
