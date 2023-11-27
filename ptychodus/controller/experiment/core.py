from ...model.experiment import DetectorPresenter, ExperimentRepositoryPresenter
from ...view.experiment import ExperimentView
from ..data import FileDialogFactory
from .detector import DetectorController
from .repository import ExperimentRepositoryController


class ExperimentController:

    def __init__(self, detectorPresenter: DetectorPresenter,
                 repositoryPresenter: ExperimentRepositoryPresenter, view: ExperimentView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._detectorController = DetectorController.createInstance(detectorPresenter,
                                                                     view.detectorView)
        self._repositoryController = ExperimentRepositoryController.createInstance(
            repositoryPresenter, view.repositoryView, fileDialogFactory)
