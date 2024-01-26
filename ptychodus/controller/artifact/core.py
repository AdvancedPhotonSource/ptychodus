from ...model.artifact import ArtifactRepositoryPresenter
from ...view.artifact import ArtifactView
from ..data import FileDialogFactory
from .repository import ArtifactRepositoryController


class ArtifactController:

    def __init__(self, repositoryPresenter: ArtifactRepositoryPresenter, view: ArtifactView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._repositoryController = ArtifactRepositoryController.createInstance(
            repositoryPresenter, view.repositoryView, fileDialogFactory)
