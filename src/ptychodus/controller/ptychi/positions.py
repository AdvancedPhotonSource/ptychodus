from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import BooleanParameter, IntegerParameter, RealParameter

from ...model.ptychi import PtyChiEnumerators, PtyChiProbePositionSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class UpdateMagnitudeLimitViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        limitMagnitudeUpdate: BooleanParameter,
        magnitudeUpdateLimit: RealParameter,
    ) -> None:
        super().__init__(limitMagnitudeUpdate, 'Limit Update Magnitude')
        self._viewController = DecimalLineEditParameterViewController(
            magnitudeUpdateLimit,
            tool_tip='Limit update magnitudes to this value.',
        )

        layout = QFormLayout()
        layout.addRow('Limit:', self._viewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiProbePositionsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbePositionSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(settings.isOptimizable, 'Optimize Probe Positions')
        self._optimizationPlanViewController = PtyChiOptimizationPlanViewController(
            settings.optimizationPlanStart,
            settings.optimizationPlanStop,
            settings.optimizationPlanStride,
            num_epochs,
        )
        self._optimizerViewController = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._stepSizeViewController = DecimalLineEditParameterViewController(
            settings.stepSize, tool_tip='Optimizer step size'
        )
        self._algorithmViewController = ComboBoxParameterViewController(
            settings.positionCorrectionType, enumerators.positionCorrectionTypes()
        )
        self._magnitudeUpdateLimitViewController = UpdateMagnitudeLimitViewController(
            settings.limitMagnitudeUpdate,
            settings.magnitudeUpdateLimit,
        )
        self._constrainCentroidViewController = CheckBoxParameterViewController(
            settings.constrainCentroid, 'Constrain Centroid'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow('Algorithm:', self._algorithmViewController.getWidget())
        layout.addRow(self._magnitudeUpdateLimitViewController.getWidget())
        layout.addRow(self._constrainCentroidViewController.getWidget())
        self.getWidget().setLayout(layout)
