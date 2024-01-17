from ...model.experiment import ExperimentRepositoryPresenter
from ...view.experiment import ExperimentView
from ..data import FileDialogFactory
from .repository import ExperimentRepositoryController


class ExperimentController:

    def __init__(self, repositoryPresenter: ExperimentRepositoryPresenter, view: ExperimentView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._repositoryController = ExperimentRepositoryController.createInstance(
            repositoryPresenter, view.repositoryView, fileDialogFactory)
