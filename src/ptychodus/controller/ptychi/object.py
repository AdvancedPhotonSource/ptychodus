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


class ConstrainL1NormViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainL1Norm: BooleanParameter,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(constrainL1Norm, 'Constrain L\u2081 Norm')
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class ConstrainSmoothnessViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainSmoothness: BooleanParameter,
        alpha: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(constrainSmoothness, 'Constrain Smoothness')
        self._alphaViewController = DecimalSliderParameterViewController(alpha)
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Alpha:', self._alphaViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class ConstrainTotalVariationViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainTotalVariation: BooleanParameter,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(constrainTotalVariation, 'Constrain Total Variation')
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class RemoveGridArtifactsViewController(CheckableGroupBoxParameterViewController):
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
        super().__init__(removeGridArtifacts, 'Remove Grid Artifacts')
        self._periodXViewController = LengthWidgetParameterViewController(periodXInMeters)
        self._periodYViewController = LengthWidgetParameterViewController(periodYInMeters)
        self._windowSizeViewController = SpinBoxParameterViewController(windowSizeInPixels)
        self._directionViewController = ComboBoxParameterViewController(
            direction, enumerators.directions()
        )
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Period X:', self._periodXViewController.getWidget())
        layout.addRow('Period Y:', self._periodYViewController.getWidget())
        layout.addRow('Window Size [px]:', self._windowSizeViewController.getWidget())
        layout.addRow('Direction:', self._directionViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class RegularizeMultisliceViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        regularizeMultislice: BooleanParameter,
        weight: RealParameter,
        unwrapPhase: BooleanParameter,
        gradientMethod: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(regularizeMultislice, 'Regularize Multislice')
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._unwrapPhaseViewController = CheckBoxParameterViewController(
            unwrapPhase, 'Unwrap Phase'
        )
        self._gradientMethodViewController = ComboBoxParameterViewController(
            gradientMethod, enumerators.imageGradientMethods()
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        layout.addRow(self._unwrapPhaseViewController.getWidget())
        layout.addRow('Gradient Method:', self._gradientMethodViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiObjectViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiObjectSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(settings.isOptimizable, 'Optimize Object')
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
            settings.patchInterpolator, enumerators.patchInterpolationMethods()
        )
        self._constrainL1NormViewController = ConstrainL1NormViewController(
            settings.constrainL1Norm, settings.constrainL1NormWeight, settings.constrainL1NormStride
        )
        self._constrainSmoothnessViewController = ConstrainSmoothnessViewController(
            settings.constrainSmoothness,
            settings.constrainSmoothnessAlpha,
            settings.constrainSmoothnessStride,
        )
        self._constrainTotalVariationViewController = ConstrainTotalVariationViewController(
            settings.constrainTotalVariation,
            settings.constrainTotalVariationWeight,
            settings.constrainTotalVariationStride,
        )
        self._removeGridArtifactsViewController = RemoveGridArtifactsViewController(
            settings.removeGridArtifacts,
            settings.removeGridArtifactsPeriodXInMeters,
            settings.removeGridArtifactsPeriodYInMeters,
            settings.removeGridArtifactsWindowSizeInPixels,
            settings.removeGridArtifactsDirection,
            settings.removeGridArtifactsStride,
            enumerators,
        )
        self._regularizeMultisliceViewController = RegularizeMultisliceViewController(
            settings.regularizeMultislice,
            settings.regularizeMultisliceWeight,
            settings.regularizeMultisliceUnwrapPhase,
            settings.regularizeMultisliceUnwrapPhaseImageGradientMethod,
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
