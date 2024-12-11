from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import PtyChiEnumerators, PtyChiObjectSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LengthWidgetParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiObjectViewController']


class PtyChiConstrainL1NormViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainL1Norm: BooleanParameter,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainL1Norm,
            'Constrain L\u2081 Norm',
            tool_tip='Whether to constrain the L\u2081 norm.',
        )
        self._weightViewController = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight of the L\u2081 norm constraint.',
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between L\u2081 norm constraint updates.'
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiConstrainSmoothnessViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainSmoothness: BooleanParameter,
        alpha: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainSmoothness,
            'Constrain Smoothness',
            tool_tip='Whether to constrain smoothness in the magnitude (but not phase) of the object',
        )
        self._alphaViewController = DecimalSliderParameterViewController(
            alpha, tool_tip='Relaxation smoothing constant.'
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between smoothness constraint updates.'
        )

        layout = QFormLayout()
        layout.addRow('Alpha:', self._alphaViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiConstrainTotalVariationViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainTotalVariation: BooleanParameter,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainTotalVariation,
            'Constrain Total Variation',
            tool_tip='Whether to constrain the total variation.',
        )
        self._weightViewController = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight of the total variation constraint.',
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between total variation constraint updates.'
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiRemoveGridArtifactsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        removeGridArtifacts: BooleanParameter,
        periodXInMeters: RealParameter,
        periodYInMeters: RealParameter,
        windowSizeInPixels: IntegerParameter,
        direction: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            removeGridArtifacts,
            'Remove Grid Artifacts',
            tool_tip="Whether to remove grid artifacts in the object's phase at the end of an epoch.",
        )
        self._periodXViewController = LengthWidgetParameterViewController(
            periodXInMeters, tool_tip='Horizontal period of grid artifacts in meters.'
        )
        self._periodYViewController = LengthWidgetParameterViewController(
            periodYInMeters, tool_tip='Vertical period of grid artifacts in meters.'
        )
        self._windowSizeViewController = SpinBoxParameterViewController(
            windowSizeInPixels, tool_tip='Window size for grid artifact removal in pixels.'
        )
        self._directionViewController = ComboBoxParameterViewController(
            direction, enumerators.directions(), tool_tip='Direction of grid artifact removal.'
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between grid artifact removal updates.'
        )

        layout = QFormLayout()
        layout.addRow('Period X:', self._periodXViewController.getWidget())
        layout.addRow('Period Y:', self._periodYViewController.getWidget())
        layout.addRow('Window Size [px]:', self._windowSizeViewController.getWidget())
        layout.addRow('Direction:', self._directionViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiRegularizeMultisliceViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        regularizeMultislice: BooleanParameter,
        weight: RealParameter,
        unwrapPhase: BooleanParameter,
        gradientMethod: StringParameter,
        integrationMethod: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            regularizeMultislice,
            'Regularize Multislice',
            tool_tip='Whether to regularize multislice objects using cross-slice smoothing.',
        )
        self._weightViewController = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight for multislice regularization.',
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between multislice regularization updates.'
        )
        self._unwrapPhaseViewController = CheckBoxParameterViewController(
            unwrapPhase,
            'Unwrap Phase',
            tool_tip='Whether to unwrap the phase of the object during multislice regularization.',
        )
        self._gradientMethodViewController = ComboBoxParameterViewController(
            gradientMethod,
            enumerators.imageGradientMethods(),
            tool_tip='Method for calculating the phase gradient during phase unwrapping.',
        )
        self._integrationMethodViewController = ComboBoxParameterViewController(
            integrationMethod,
            enumerators.imageIntegrationMethods(),
            tool_tip='Method for integrating the phase gradient during phase unwrapping.',
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        layout.addRow(self._unwrapPhaseViewController.getWidget())
        layout.addRow('Gradient Method:', self._gradientMethodViewController.getWidget())
        layout.addRow('Integration Method:', self._integrationMethodViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiObjectViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiObjectSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.isOptimizable, 'Optimize Object', tool_tip='Whether the object is optimizable.'
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
        self._patchInterpolatorViewController = ComboBoxParameterViewController(
            settings.patchInterpolator,
            enumerators.patchInterpolationMethods(),
            tool_tip='Interpolation method used for extracting and updating patches of the object.',
        )
        self._constrainL1NormViewController = PtyChiConstrainL1NormViewController(
            settings.constrainL1Norm, settings.constrainL1NormWeight, settings.constrainL1NormStride
        )
        self._constrainSmoothnessViewController = PtyChiConstrainSmoothnessViewController(
            settings.constrainSmoothness,
            settings.constrainSmoothnessAlpha,
            settings.constrainSmoothnessStride,
        )
        self._constrainTotalVariationViewController = PtyChiConstrainTotalVariationViewController(
            settings.constrainTotalVariation,
            settings.constrainTotalVariationWeight,
            settings.constrainTotalVariationStride,
        )
        self._removeGridArtifactsViewController = PtyChiRemoveGridArtifactsViewController(
            settings.removeGridArtifacts,
            settings.removeGridArtifactsPeriodXInMeters,
            settings.removeGridArtifactsPeriodYInMeters,
            settings.removeGridArtifactsWindowSizeInPixels,
            settings.removeGridArtifactsDirection,
            settings.removeGridArtifactsStride,
            enumerators,
        )
        self._regularizeMultisliceViewController = PtyChiRegularizeMultisliceViewController(
            settings.regularizeMultislice,
            settings.regularizeMultisliceWeight,
            settings.regularizeMultisliceUnwrapPhase,
            settings.regularizeMultisliceUnwrapPhaseImageGradientMethod,
            settings.regularizeMultisliceUnwrapPhaseImageIntegrationMethod,
            settings.regularizeMultisliceStride,
            enumerators,
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow('Patch Interpolator:', self._patchInterpolatorViewController.getWidget())
        layout.addRow(self._constrainL1NormViewController.getWidget())
        layout.addRow(self._constrainSmoothnessViewController.getWidget())
        layout.addRow(self._constrainTotalVariationViewController.getWidget())
        layout.addRow(self._removeGridArtifactsViewController.getWidget())
        layout.addRow(self._regularizeMultisliceViewController.getWidget())
        self.getWidget().setLayout(layout)
