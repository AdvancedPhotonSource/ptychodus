import logging

from ptychodus.api.xmcd import XMCDResult

from ...model.analysis import XMCDAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import XMCDDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .tree_model import ObjectTreeModel

logger = logging.getLogger(__name__)

_SAVE_FILE_FILTER = 'NumPy Zipped Archive (*.npz)'


class XMCDViewController:
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
        self._result: XMCDResult | None = None

        self._dialog = XMCDDialog()
        self._dialog.setWindowTitle('X-ray Magnetic Circular Dichroism (XMCD)')
        self._dialog.parameters_view.lcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.lcirc_combo_box.textActivated.connect(self._reanalyze)
        self._dialog.parameters_view.rcirc_combo_box.setModel(tree_model)
        self._dialog.parameters_view.rcirc_combo_box.textActivated.connect(self._reanalyze)
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

    def analyze(self, lcirc_product_index: int, rcirc_product_index: int) -> None:
        self._dialog.parameters_view.lcirc_combo_box.setCurrentIndex(lcirc_product_index)
        self._dialog.parameters_view.rcirc_combo_box.setCurrentIndex(rcirc_product_index)
        self._reanalyze()
        self._dialog.open()

    def _reanalyze(self) -> None:
        lcp_index = self._dialog.parameters_view.lcirc_combo_box.currentIndex()
        rcp_index = self._dialog.parameters_view.rcirc_combo_box.currentIndex()

        if lcp_index < 0 or rcp_index < 0:
            self._result = None
            self._render_result()
            return

        try:
            self._result = self._analyzer.analyze(lcp_index, rcp_index)
        except Exception as err:
            self._result = None
            logger.exception(err)
            ExceptionDialog.show_exception('XMCD Analysis', err)

        self._render_result()

    def _render_result(self) -> None:
        if self._result is None:
            self._structural_image_controller.clear_array()
            self._magnetic_image_controller.clear_array()
            return

        structural_object = self._result.structural_object
        self._structural_image_controller.set_array(
            structural_object.get_layer(0), structural_object.get_pixel_geometry()
        )
        magnetic_object = self._result.magnetic_object
        self._magnetic_image_controller.set_array(
            magnetic_object.get_layer(0), magnetic_object.get_pixel_geometry()
        )

    def _save_data(self) -> None:
        if self._result is None:
            logger.warning('No XMCD result to save!')
            return

        title = 'Save XMCD Data'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[_SAVE_FILE_FILTER],
            selected_name_filter=_SAVE_FILE_FILTER,
        )

        if file_path:
            try:
                self._result.save_npz(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)
