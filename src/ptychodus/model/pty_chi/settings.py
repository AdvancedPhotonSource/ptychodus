from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtyChiReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtyChi')
        self._settingsGroup.addObserver(self)

        self.numEpochs = self._settingsGroup.createIntegerParameter('NumEpochs', 100, minimum=1)
        self.batchSize = self._settingsGroup.createIntegerParameter('BatchSize', 1, minimum=1)
        self.batchingMode = self._settingsGroup.createStringParameter('BatchingMode', 'random')
        self.compactModeUpdateClustering = self._settingsGroup.createBooleanParameter(
            'CompactModeUpdateClustering', False
        )
        self.compactModeUpdateClusteringStride = self._settingsGroup.createIntegerParameter(
            'CompactModeUpdateClusteringStride', 1
        )
        self.useDevices = self._settingsGroup.createBooleanParameter('UseDevices', True)
        self.devices = self._settingsGroup.createIntegerSequenceParameter('Devices', ())
        self.useDoublePrecision = self._settingsGroup.createBooleanParameter(
            'UseDoublePrecision', False
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

        self.l1NormConstraintWeight = self._settingsGroup.createRealParameter(
            'L1NormConstraintWeight', 0.0, minimum=0.0
        )
        self.l1NormConstraintStride = self._settingsGroup.createIntegerParameter(
            'L1NormConstraintStride', 1, minimum=1
        )
        self.smoothnessConstraintAlpha = self._settingsGroup.createRealParameter(
            'SmoothnessConstraintAlpha', 0.0, minimum=0.0, maximum=1.0 / 8
        )
        self.smoothnessConstraintStride = self._settingsGroup.createIntegerParameter(
            'SmoothnessConstraintStride', 1, minimum=1
        )
        self.totalVariationWeight = self._settingsGroup.createRealParameter(
            'TotalVariationWeight', 0.0, minimum=0.0
        )
        self.totalVaritionStride = self._settingsGroup.createIntegerParameter(
            'TotalVaritionStride', 1, minimum=1
        )
        self.removeGridArtifacts = self._settingsGroup.createBooleanParameter(
            'RemoveGridArtifacts', False
        )
        self.removeGridArtifactsPeriodXInMeters = self._settingsGroup.createRealParameter(
            'RemoveGridArtifactsPeriodXInMeters', 1e-7
        )
        self.removeGridArtifactsPeriodYInMeters = self._settingsGroup.createRealParameter(
            'RemoveGridArtifactsPeriodYInMeters', 1e-7
        )
        self.removeGridArtifactsWindowSizeInPixels = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsWindowSizeInPixels', 5
        )
        self.removeGridArtifactsDirection = self._settingsGroup.createStringParameter(
            'RemoveGridArtifactsDirection', 'XY'
        )
        self.removeGridArtifactsStride = self._settingsGroup.createIntegerParameter(
            'RemoveGridArtifactsStride', 1
        )
        self.multisliceRegularizationWeight = self._settingsGroup.createRealParameter(
            'MultisliceRegularizationWeight', 0.0
        )
        self.multisliceRegularizationUnwrapPhase = self._settingsGroup.createBooleanParameter(
            'MultisliceRegularizationUnwrapPhase', True
        )
        self.multisliceRegularizationUnwrapImageGradMethod = (
            self._settingsGroup.createStringParameter(
                'MultisliceRegularizationUnwrapImageGradMethod', 'FOURIER_SHIFT'
            )
        )
        self.multisliceRegularizationStride = self._settingsGroup.createIntegerParameter(
            'MultisliceRegularizationStride', 1
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

        self.orthogonalizeIncoherentModesMethod = self._settingsGroup.createStringParameter(
            'OrthogonalizeIncoherentModesMethod', 'GS'
        )
        self.probePower = self._settingsGroup.createRealParameter('ProbePower', 0.0, minimum=0.0)
        self.probePowerConstraintStride = self._settingsGroup.createIntegerParameter(
            'ProbePowerConstraintStride', 1, minimum=1
        )
        self.orthogonalizeIncoherentModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeIncoherentModes', False
        )
        self.orthogonalizeIncoherentModesStride = self._settingsGroup.createIntegerParameter(
            'OrthogonalizeIncoherentModesStride', 1, minimum=1
        )
        self.orthogonalizeOPRModes = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeOPRModes', False
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

        self.updateMagnitudeLimit = self._settingsGroup.createRealParameter(
            'UpdateMagnitudeLimit', 0.0, minimum=0.0
        )
        self.constrainPositionMean = self._settingsGroup.createBooleanParameter(
            'ConstrainPositionMean', False
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
