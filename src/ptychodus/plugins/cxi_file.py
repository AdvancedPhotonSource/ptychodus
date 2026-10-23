"""Coherent X-ray Imaging (CXI) file plugins over :mod:`ptychodus.api.io`.

One reader per kind of data detects the file's layout by itself; writers come in one
flavor per :class:`~ptychodus.api.io.CXIVariant`, so the file suits the program that
reads it next.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import h5py
import numpy

from ptychodus.api.diffraction import (
    BadPixels,
    BadPixelsFileReader,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionFileWriter,
    PixelMapFileReader,
    SimpleDiffractionDataset,
)
from ptychodus.api.io import (
    CXIVariant,
    load_cxi_background,
    load_cxi_bad_pixels,
    load_cxi_diffraction_contents,
    load_cxi_flat_field,
    load_cxi_product,
    save_cxi_diffraction,
    save_cxi_product,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter

from .h5_diffraction_file import H5DiffractionFileTreeBuilder, H5DiffractionPatternArray


class CXIDiffractionFileReader(DiffractionFileReader):
    """Reads patterns lazily, with the mask, background and flat field the file carries."""

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def get_file_description(self) -> str:
        return 'Coherent X-ray Imaging Files'

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

        contents = load_cxi_diffraction_contents(file_path)
        array = H5DiffractionPatternArray(
            label=file_path.stem,
            indexes=contents.indexes,
            file_path=file_path,
            data_path=contents.data_path,
        )
        return SimpleDiffractionDataset(
            contents.metadata,
            contents_tree,
            [array],
            contents.bad_pixels,
            background=contents.background,
            flat_field=contents.flat_field,
        )


_VARIANT_DESCRIPTIONS: Final[Mapping[CXIVariant, str]] = {
    CXIVariant.STANDARD: 'Coherent X-ray Imaging Files',
    CXIVariant.SHARP: 'Coherent X-ray Imaging Files, SHARP layout',
    CXIVariant.CDTOOLS: 'Coherent X-ray Imaging Files, cdtools layout',
}


class CXIDiffractionFileWriter(DiffractionFileWriter):
    def __init__(self, variant: CXIVariant) -> None:
        self._variant = variant

    def get_file_description(self) -> str:
        return _VARIANT_DESCRIPTIONS[self._variant]

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        save_cxi_diffraction(file_path, dataset, variant=self._variant)


class CXIProductFileIO(ProductFileReader, ProductFileWriter):
    def __init__(self, variant: CXIVariant = CXIVariant.STANDARD) -> None:
        self._variant = variant

    def get_file_description(self) -> str:
        return _VARIANT_DESCRIPTIONS[self._variant]

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def read(self, file_path: Path) -> Product:
        return load_cxi_product(file_path)

    def write(self, file_path: Path, product: Product) -> None:
        save_cxi_product(file_path, product, variant=self._variant)


class CXIBadPixelsFileReader(BadPixelsFileReader):
    def get_file_description(self) -> str:
        return 'Coherent X-ray Imaging Files'

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def read(self, file_path: Path) -> BadPixels:
        return load_cxi_bad_pixels(file_path)


class CXIBackgroundFileReader(PixelMapFileReader):
    def get_file_description(self) -> str:
        return 'Coherent X-ray Imaging Files'

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def read(self, file_path: Path) -> numpy.ndarray:
        return load_cxi_background(file_path)


class CXIFlatFieldFileReader(PixelMapFileReader):
    def get_file_description(self) -> str:
        return 'Coherent X-ray Imaging Files'

    def get_file_extensions(self) -> set[str]:
        return {'.cxi'}

    def read(self, file_path: Path) -> numpy.ndarray:
        return load_cxi_flat_field(file_path)


def register_plugins(registry: PluginRegistry) -> None:
    simple_name: Final[str] = 'CXI'

    registry.diffraction_file_readers.register_plugin(
        CXIDiffractionFileReader(), simple_name=simple_name
    )
    registry.bad_pixels_file_readers.register_plugin(
        CXIBadPixelsFileReader(), simple_name=simple_name
    )
    registry.background_file_readers.register_plugin(
        CXIBackgroundFileReader(), simple_name=simple_name
    )
    registry.flat_field_file_readers.register_plugin(
        CXIFlatFieldFileReader(), simple_name=simple_name
    )

    # The product reader returns an empty probe and object for a file that has neither,
    # so the positions it yields serve as a positions reader for raw-data files too.
    registry.register_product_file_reader_with_adapters(CXIProductFileIO(), simple_name=simple_name)

    writers = (
        (CXIVariant.STANDARD, simple_name),
        (CXIVariant.SHARP, 'CXI_SHARP'),
        (CXIVariant.CDTOOLS, 'CXI_cdtools'),
    )

    for variant, variant_simple_name in writers:
        registry.diffraction_file_writers.register_plugin(
            CXIDiffractionFileWriter(variant), simple_name=variant_simple_name
        )

        # Product images are the same in the standard and cdtools layouts.
        if variant != CXIVariant.CDTOOLS:
            registry.product_file_writers.register_plugin(
                CXIProductFileIO(variant), simple_name=variant_simple_name
            )
