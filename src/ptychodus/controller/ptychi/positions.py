from PyQt5.QtWidgets import QFormLayout, QFrame, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import PtyChiEnumerators, PtyChiProbePositionSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbePositionsViewController']


class PtyChiCrossCorrelationViewController(ParameterViewController, Observer):
    def __init__(
        self,
        algorithm: StringParameter,
        scale: IntegerParameter,
        realSpaceWidth: RealParameter,
        probeThreshold: RealParameter,
    ) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._scaleViewController = SpinBoxParameterViewController(
            scale, tool_tip='Upsampling factor of the cross-correlation in real space.'
        )
        self._realSpaceWidthViewController = DecimalLineEditParameterViewController(
            realSpaceWidth, tool_tip='Width of the cross-correlation in real-space'
        )
        self._probeThresholdViewController = DecimalSliderParameterViewController(
            probeThreshold, tool_tip='Probe intensity threshold used to calculate the probe mask.'
        )
        self._widget = QFrame()
        self._widget.setFrameShape(QFrame.StyledPanel)

        layout = QFormLayout()
        layout.addRow('Scale:', self._scaleViewController.getWidget())
        layout.addRow('Real Space Width:', self._realSpaceWidthViewController.getWidget())
        layout.addRow('Probe Threshold:', self._probeThresholdViewController.getWidget())
        self._widget.setLayout(layout)

        algorithm.addObserver(self)
        self._syncModelToView()

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setVisible(self._algorithm.getValue().upper() == 'CROSS_CORRELATION')

    def update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._syncModelToView()


class PtyChiUpdateMagnitudeLimitViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        limitMagnitudeUpdate: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        magnitudeUpdateLimit: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            limitMagnitudeUpdate,
            'Limit Update Magnitude',
            tool_tip='Limit the magnitude of the probe update.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._viewController = DecimalLineEditParameterViewController(
            magnitudeUpdateLimit,
            tool_tip='Magnitude limit of the probe update.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        layout.addRow('Limit:', self._viewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiProbePositionsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbePositionSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.isOptimizable,
            'Optimize Probe Positions',
            tool_tip='Whether the probe positions are optimizable.',
        )
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
            settings.positionCorrectionType,
            enumerators.positionCorrectionTypes(),
            tool_tip='Algorithm used to calculate the position correction update.',
        )
        self._crossCorrelationViewController = PtyChiCrossCorrelationViewController(
            settings.positionCorrectionType,
            settings.crossCorrelationScale,
            settings.crossCorrelationRealSpaceWidth,
            settings.crossCorrelationProbeThreshold,
        )
        self._magnitudeUpdateLimitViewController = PtyChiUpdateMagnitudeLimitViewController(
            settings.limitMagnitudeUpdate,
            settings.limitMagnitudeUpdateStart,
            settings.limitMagnitudeUpdateStop,
            settings.limitMagnitudeUpdateStride,
            settings.magnitudeUpdateLimit,
            num_epochs,
        )
        self._constrainCentroidViewController = CheckBoxParameterViewController(
            settings.constrainCentroid,
            'Constrain Centroid',
            tool_tip='Whether to subtract the mean from positions after updating positions.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow('Algorithm:', self._algorithmViewController.getWidget())
        layout.addRow(self._crossCorrelationViewController.getWidget())
        layout.addRow(self._magnitudeUpdateLimitViewController.getWidget())
        layout.addRow(self._constrainCentroidViewController.getWidget())
        self.getWidget().setLayout(layout)
