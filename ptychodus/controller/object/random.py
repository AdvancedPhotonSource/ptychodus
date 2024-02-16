import logging

from PyQt5.QtWidgets import QWidget

from ...model.object import ObjectRepositoryItem, RandomObjectBuilder
from ...view.object import ObjectEditorDialog, RandomObjectView
from ..widgets import (DecimalSliderParameterController, LengthWidgetParameterController,
                       SpinBoxParameterController)

logger = logging.getLogger(__name__)


class RandomObjectViewController:

    def __init__(self, builder: RandomObjectBuilder, view: RandomObjectView) -> None:
        self._numberOfLayersController = SpinBoxParameterController(builder.numberOfLayers,
                                                                    view.numberOfLayersSpinBox)
        self._layerDistanceController = LengthWidgetParameterController(
            builder.layerDistanceInMeters, view.layerDistanceWidget)
        self._extraPaddingXController = SpinBoxParameterController(builder.extraPaddingX,
                                                                   view.extraPaddingXSpinBox)
        self._extraPaddingYController = SpinBoxParameterController(builder.extraPaddingY,
                                                                   view.extraPaddingYSpinBox)

        self._amplitudeMeanController = DecimalSliderParameterController(
            builder.amplitudeMean, view.amplitudeMeanSlider)
        self._amplitudeDeviationController = DecimalSliderParameterController(
            builder.amplitudeDeviation, view.amplitudeDeviationSlider)
        self._phaseDeviationController = DecimalSliderParameterController(
            builder.phaseDeviation, view.phaseDeviationSlider)

    @classmethod
    def editParameters(cls, item: ObjectRepositoryItem, parent: QWidget) -> None:
        builder = item.getBuilder()

        if isinstance(builder, RandomObjectBuilder):
            view = RandomObjectView.createInstance()
            controller = cls(builder, view)

            dialog = ObjectEditorDialog.createInstance(item.getName(), view, parent)
            dialog.open()
        else:
            logger.warning('Builder is not a RandomObjectBuilder')
