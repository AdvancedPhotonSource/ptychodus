import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import XMCDAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import XMCDDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .tree_model import ObjectTreeModel

logger = logging.getLogger(__name__)


class XMCDViewController(Observer):
    def __init__(
        self,
        analyzer: XMCDAnalyzer,
        structural_visualization_engine: VisualizationEngine,
        magnetic_visualization_engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
        tree_model: ObjectTreeModel,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._file_dialog_factory = file_dialog_factory
        self._dialog = XMCDDialog()
        self._dialog.setWindowTitle('X-ray Magnetic Circular Dichroism (XMCD)')
        self._dialog.parameters_view.lcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.lcirc_combo_box.currentIndexChanged.connect(
            analyzer.set_lcp_product
        )
        self._dialog.parameters_view.rcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.rcirc_combo_box.currentIndexChanged.connect(
            analyzer.set_rcp_product
        )
        self._dialog.parameters_view.save_button.clicked.connect(self._save_data)

        self._structural_image_controller = ImageController(
            structural_visualization_engine,
            self._dialog.structural_view,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._magnetic_image_controller = ImageController(
            magnetic_visualization_engine,
            self._dialog.magnetic_view,
            self._dialog.status_bar,
            file_dialog_factory,
        )

        analyzer.add_observer(self)

    def analyze(self, lcirc_product_index: int, rcirc_product_index: int) -> None:
        self._analyzer.set_lcp_product(lcirc_product_index)
        self._analyzer.set_rcp_product(rcirc_product_index)
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
        lcirc_product_index = self._analyzer.get_lcp_product()
        self._dialog.parameters_view.lcirc_combo_box.setCurrentIndex(lcirc_product_index)

        rcirc_product_index = self._analyzer.get_rcp_product()
        self._dialog.parameters_view.rcirc_combo_box.setCurrentIndex(rcirc_product_index)

        try:
            result = self._analyzer.get_result()
        except ValueError:
            self._structural_image_controller.clear_array()
            self._magnetic_image_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            structural_object = result.structural_object
            self._structural_image_controller.set_array(
                structural_object.get_layer(0), structural_object.get_pixel_geometry()
            )
            magnetic_object = result.magnetic_object
            self._magnetic_image_controller.set_array(
                magnetic_object.get_layer(0), magnetic_object.get_pixel_geometry()
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._analyzer:
            self._sync_model_to_view()
