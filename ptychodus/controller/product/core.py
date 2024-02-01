from ...model.product import ProductRepositoryPresenter
from ...view.product import ProductView
from ..data import FileDialogFactory
from .repository import ProductRepositoryController


class ProductController:

    def __init__(self, repositoryPresenter: ProductRepositoryPresenter, view: ProductView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._repositoryController = ProductRepositoryController.createInstance(
            repositoryPresenter, view.repositoryView, fileDialogFactory)
