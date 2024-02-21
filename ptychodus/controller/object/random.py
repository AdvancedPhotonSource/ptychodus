import logging

from PyQt5.QtWidgets import QWidget

from ...model.object import ObjectRepositoryItem, RandomObjectBuilder
from ..parametric import ParameterDialogBuilder

logger = logging.getLogger(__name__)


class RandomObjectEditorViewController:

    @classmethod
    def edit(cls, item: ObjectRepositoryItem, parent: QWidget) -> None:
        objectBuilder = item.getBuilder()

        if isinstance(objectBuilder, RandomObjectBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox('Number of Layers', objectBuilder.numberOfLayers)
            dialogBuilder.addLengthWidget('Layer Distance', objectBuilder.layerDistanceInMeters)
            dialogBuilder.addSpinBox('Extra Padding X', objectBuilder.extraPaddingX)
            dialogBuilder.addSpinBox('Extra Padding Y', objectBuilder.extraPaddingY)
            dialogBuilder.addDecimalSlider('Amplitude Mean', objectBuilder.amplitudeMean)
            dialogBuilder.addDecimalSlider('Amplitude Deviation', objectBuilder.amplitudeDeviation)
            dialogBuilder.addDecimalSlider('Phase Deviation', objectBuilder.phaseDeviation)
            dialog = dialogBuilder.build(windowTitle, parent)
            dialog.open()
        else:
            logger.warning('Builder is not a RandomObjectBuilder')
