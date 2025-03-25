from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import QStringListModel

from ptychodus.api.observer import Observable, Observer

from ...model.visualization import VisualizationEngine
from ...view.visualization import VisualizationParametersView


class VisualizationParametersController(Observer):
    def __init__(self, engine: VisualizationEngine, view: VisualizationParametersView) -> None:
        super().__init__()
        self._engine = engine
        self._view = view
        self._rendererModel = QStringListModel()
        self._transformationModel = QStringListModel()
        self._variantModel = QStringListModel()

    @classmethod
    def create_instance(
        cls, engine: VisualizationEngine, view: VisualizationParametersView
    ) -> VisualizationParametersController:
        controller = cls(engine, view)
        view.rendererComboBox.setModel(controller._rendererModel)
        view.transformationComboBox.setModel(controller._transformationModel)
        view.variantComboBox.setModel(controller._variantModel)

        view.minDisplayValueLineEdit.value_changed.connect(
            lambda value: engine.set_min_display_value(float(value))
        )
        view.maxDisplayValueLineEdit.value_changed.connect(
            lambda value: engine.set_max_display_value(float(value))
        )

        controller._sync_model_to_view()
        engine.add_observer(controller)

        view.rendererComboBox.textActivated.connect(engine.set_renderer)
        view.transformationComboBox.textActivated.connect(engine.set_transformation)
        view.variantComboBox.textActivated.connect(engine.set_variant)

        return controller

    def _sync_model_to_view(self) -> None:
        self._view.rendererComboBox.blockSignals(True)
        self._rendererModel.setStringList([name for name in self._engine.renderers()])
        self._view.rendererComboBox.setCurrentText(self._engine.get_renderer())
        self._view.rendererComboBox.blockSignals(False)

        self._view.transformationComboBox.blockSignals(True)
        self._transformationModel.setStringList([name for name in self._engine.transformations()])
        self._view.transformationComboBox.setCurrentText(self._engine.get_transformation())
        self._view.transformationComboBox.blockSignals(False)

        self._view.variantComboBox.blockSignals(True)
        self._variantModel.setStringList([name for name in self._engine.variants()])
        self._view.variantComboBox.setCurrentText(self._engine.get_variant())
        self._view.variantComboBox.blockSignals(False)

        self._view.minDisplayValueLineEdit.set_value(
            Decimal(repr(self._engine.get_min_display_value()))
        )
        self._view.maxDisplayValueLineEdit.set_value(
            Decimal(repr(self._engine.get_max_display_value()))
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()
