from pathlib import Path
from typing import Final


from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter
from ptychodus.api.io import load_product, save_product


class H5ProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'HDF5'
    DISPLAY_NAME: Final[str] = 'Ptychodus Product Files (*.h5 *.hdf5)'

    def read(self, file_path: Path) -> Product:
        return load_product(file_path)

    def write(self, file_path: Path, product: Product) -> None:
        save_product(file_path, product)


def register_plugins(registry: PluginRegistry) -> None:
    h5_product_file_io = H5ProductFileIO()

    registry.register_product_file_reader_with_adapters(
        h5_product_file_io,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
    registry.product_file_writers.register_plugin(
        h5_product_file_io,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
