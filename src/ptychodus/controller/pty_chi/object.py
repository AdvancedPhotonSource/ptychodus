from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.pty_chi import PtyChiEnumerators, PtyChiObjectSettings
from ..parametric import (
    CheckBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LengthWidgetParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class L1NormConstraintViewController(ParameterViewController):
    def __init__(
        self,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__()
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._widget = QGroupBox('L\u2081 Norm Constraint')

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class SmoothnessConstraintViewController(ParameterViewController):
    def __init__(
        self,
        alpha: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__()
        self._alphaViewController = DecimalSliderParameterViewController(alpha)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._widget = QGroupBox('Smoothness Constraint')

        layout = QFormLayout()
        layout.addRow('Alpha:', self._alphaViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class TotalVariationViewController(ParameterViewController):
    def __init__(
        self,
        weight: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__()
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._widget = QGroupBox('Total Variation')

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class RemoveGridArtifactsViewController(ParameterViewController, Observer):
    def __init__(
        self,
        isEnabled: BooleanParameter,
        periodXInMeters: RealParameter,
        periodYInMeters: RealParameter,
        windowSizeInPixels: IntegerParameter,
        direction: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__()
        self._isEnabled = isEnabled
        self._periodXViewController = LengthWidgetParameterViewController(periodXInMeters)
        self._periodYViewController = LengthWidgetParameterViewController(periodYInMeters)
        self._windowSizeViewController = SpinBoxParameterViewController(windowSizeInPixels)
        self._directionViewController = ComboBoxParameterViewController(
            direction, enumerators.directions()
        )
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._widget = QGroupBox('Remove Grid Artifacts')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Period X:', self._periodXViewController.getWidget())
        layout.addRow('Period Y:', self._periodYViewController.getWidget())
        layout.addRow('Window Size [px]:', self._windowSizeViewController.getWidget())
        layout.addRow('Direction:', self._directionViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(isEnabled.setValue)
        self._isEnabled.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._isEnabled.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._isEnabled:
            self._syncModelToView()


class MultisliceRegularizationViewController(ParameterViewController):
    def __init__(
        self,
        weight: RealParameter,
        unwrapPhase: BooleanParameter,
        gradientMethod: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__()
        self._weightViewController = DecimalLineEditParameterViewController(weight)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._unwrapPhaseViewController = CheckBoxParameterViewController(
            unwrapPhase, 'Unwrap Phase'
        )
        self._gradientMethodViewController = ComboBoxParameterViewController(
            gradientMethod, enumerators.imageGradientMethods()
        )
        self._widget = QGroupBox('Multislice Regularization')

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        layout.addRow(self._unwrapPhaseViewController.getWidget())
        layout.addRow('Gradient Method:', self._gradientMethodViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtyChiObjectViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiObjectSettings, enumerators: PtyChiEnumerators) -> None:
        super().__init__()
        self._isOptimizable = settings.isOptimizable
        self._optimizationPlanViewController = PtyChiOptimizationPlanViewController(
            settings.optimizationPlanStart,
            settings.optimizationPlanStop,
            settings.optimizationPlanStride,
        )
        self._optimizerViewController = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._stepSizeViewController = DecimalLineEditParameterViewController(
            settings.stepSize, tool_tip='Optimizer step size'
        )
        self._l1NormConstraintViewController = L1NormConstraintViewController(
            settings.l1NormConstraintWeight, settings.l1NormConstraintStride
        )
        self._smoothnessConstraintViewController = SmoothnessConstraintViewController(
            settings.smoothnessConstraintAlpha, settings.smoothnessConstraintStride
        )
        self._totalVariationViewController = TotalVariationViewController(
            settings.totalVariationWeight, settings.totalVariationStride
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
        self._multisliceRegularizationViewController = MultisliceRegularizationViewController(
            settings.multisliceRegularizationWeight,
            settings.multisliceRegularizationUnwrapPhase,
            settings.multisliceRegularizationUnwrapPhaseImageGradientMethod,
            settings.multisliceRegularizationStride,
            enumerators,
        )
        self._widget = QGroupBox('Optimize Object')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._l1NormConstraintViewController.getWidget())
        layout.addRow(self._smoothnessConstraintViewController.getWidget())
        layout.addRow(self._totalVariationViewController.getWidget())
        layout.addRow(self._removeGridArtifactsViewController.getWidget())
        layout.addRow(self._multisliceRegularizationViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(self._isOptimizable.setValue)
        self._isOptimizable.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._isOptimizable.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._isOptimizable:
            self._syncModelToView()
