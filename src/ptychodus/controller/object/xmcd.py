import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import XMCDAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import XMCDDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)
from .tree_model import ObjectTreeModel

logger = logging.getLogger(__name__)


class XMCDViewController(Observer):
    def __init__(
        self,
        analyzer: XMCDAnalyzer,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
        tree_model: ObjectTreeModel,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine
        self._file_dialog_factory = file_dialog_factory
        self._dialog = XMCDDialog()
        self._dialog.setWindowTitle('X-ray Magnetic Circular Dichroism (XMCD)')
        self._dialog.parameters_view.lcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.lcirc_combo_box.currentIndexChanged.connect(
            analyzer.set_lcirc_product
        )
        self._dialog.parameters_view.rcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.rcirc_combo_box.currentIndexChanged.connect(
            analyzer.set_rcirc_product
        )
        self._dialog.parameters_view.save_button.clicked.connect(self._save_data)

        self._difference_visualization_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.difference_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._sum_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.sum_widget, self._dialog.status_bar, file_dialog_factory
        )
        self._ratio_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.ratio_widget, self._dialog.status_bar, file_dialog_factory
        )
        self._visualization_parameters_controller = (
            VisualizationParametersController.create_instance(
                engine, self._dialog.parameters_view.visualization_parameters_view
            )
        )

        analyzer.add_observer(self)

    def analyze(self, lcirc_product_index: int, rcirc_product_index: int) -> None:
        self._analyzer.set_lcirc_product(lcirc_product_index)
        self._analyzer.set_rcirc_product(rcirc_product_index)
        self._analyzer.analyze()
        self._dialog.open()

    def _save_data(self) -> None:
        title = 'Save XMCD Data'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=self._analyzer.get_save_file_filters(),
            selected_name_filter=self._analyzer.get_save_file_filter(),
        )

        if file_path:
            try:
                self._analyzer.save_data(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        lcirc_product_index = self._analyzer.get_lcirc_product()
        self._dialog.parameters_view.lcirc_combo_box.setCurrentIndex(lcirc_product_index)

        rcirc_product_index = self._analyzer.get_rcirc_product()
        self._dialog.parameters_view.rcirc_combo_box.setCurrentIndex(rcirc_product_index)

        try:
            data = self._analyzer.get_result()
        except ValueError:
            self._difference_visualization_widget_controller.clear_array()
            self._sum_visualization_widget_controller.clear_array()
            self._ratio_visualization_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            self._difference_visualization_widget_controller.set_array(
                data.polar_difference, data.pixel_geometry
            )
            self._sum_visualization_widget_controller.set_array(data.polar_sum, data.pixel_geometry)
            self._ratio_visualization_widget_controller.set_array(
                data.polar_ratio, data.pixel_geometry
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._analyzer:
            self._sync_model_to_view()
