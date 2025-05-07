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
        self._renderer_model = QStringListModel()
        self._transformation_model = QStringListModel()
        self._variant_model = QStringListModel()

    @classmethod
    def create_instance(
        cls, engine: VisualizationEngine, view: VisualizationParametersView
    ) -> VisualizationParametersController:
        controller = cls(engine, view)
        view.renderer_combo_box.setModel(controller._renderer_model)
        view.transformation_combo_box.setModel(controller._transformation_model)
        view.variant_combo_box.setModel(controller._variant_model)

        view.min_display_value_line_edit.value_changed.connect(
            lambda value: engine.set_min_display_value(float(value))
        )
        view.max_display_value_line_edit.value_changed.connect(
            lambda value: engine.set_max_display_value(float(value))
        )

        controller._sync_model_to_view()
        engine.add_observer(controller)

        view.renderer_combo_box.textActivated.connect(engine.set_renderer)
        view.transformation_combo_box.textActivated.connect(engine.set_transformation)
        view.variant_combo_box.textActivated.connect(engine.set_variant)

        return controller

    def _sync_model_to_view(self) -> None:
        self._view.renderer_combo_box.blockSignals(True)
        self._renderer_model.setStringList([name for name in self._engine.renderers()])
        self._view.renderer_combo_box.setCurrentText(self._engine.get_renderer())
        self._view.renderer_combo_box.blockSignals(False)

        self._view.transformation_combo_box.blockSignals(True)
        self._transformation_model.setStringList([name for name in self._engine.transformations()])
        self._view.transformation_combo_box.setCurrentText(self._engine.get_transformation())
        self._view.transformation_combo_box.blockSignals(False)

        self._view.variant_combo_box.blockSignals(True)
        self._variant_model.setStringList([name for name in self._engine.variants()])
        self._view.variant_combo_box.setCurrentText(self._engine.get_variant())
        self._view.variant_combo_box.blockSignals(False)

        self._view.min_display_value_line_edit.set_value(
            Decimal(repr(self._engine.get_min_display_value()))
        )
        self._view.max_display_value_line_edit.set_value(
            Decimal(repr(self._engine.get_max_display_value()))
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()
