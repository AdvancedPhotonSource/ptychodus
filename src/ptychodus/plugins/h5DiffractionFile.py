from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class H5DiffractionPatternArray(DiffractionPatternArray):
    def __init__(
        self, label: str, indexes: PatternIndexesType, filePath: Path, dataPath: str
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._filePath = filePath
        self._dataPath = dataPath

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        with h5py.File(self._filePath, 'r') as h5File:
            try:
                item = h5File[self._dataPath]
            except KeyError:
                raise ValueError(f'Symlink {self._filePath}:{self._dataPath} is broken!')
            else:
                if not isinstance(item, h5py.Dataset):
                    raise ValueError(f'Symlink {self._filePath}:{self._dataPath} is not a dataset!')

            data = item[()]

        return data


class H5DiffractionFileTreeBuilder:
    def _addAttributes(
        self, treeNode: SimpleTreeNode, attributeManager: h5py.AttributeManager
    ) -> None:
        for name, value in attributeManager.items():
            if isinstance(value, str):
                itemDetails = f'STRING = "{value}"'
            elif isinstance(value, h5py.Empty):
                logger.debug(f'Skipping empty attribute {name}.')
            elif isinstance(value, numpy.ndarray):
                itemDetails = f'ARRAY = {value}'
            else:
                stringInfo = h5py.check_string_dtype(value.dtype)

                if stringInfo:
                    itemDetails = f'STRING = "{value.decode(stringInfo.encoding)}"'
                else:
                    itemDetails = f'SCALAR {value.dtype} = {value}'

            treeNode.create_child([str(name), 'Attribute', itemDetails])

    def createRootNode(self) -> SimpleTreeNode:
        return SimpleTreeNode.create_root(['Name', 'Type', 'Details'])

    def build(self, h5File: h5py.File) -> SimpleTreeNode:
        rootNode = self.createRootNode()
        unvisited = [(rootNode, h5File)]

        while unvisited:
            parentItem, h5Group = unvisited.pop()

            for itemName in h5Group:
                itemType = 'Unknown'
                itemDetails = ''
                h5Item = h5Group.get(itemName, getlink=True)

                treeNode = parentItem.create_child(list())

                if isinstance(h5Item, h5py.HardLink):
                    itemType = 'Hard Link'
                    h5Item = h5Group.get(itemName, getlink=False)

                    if isinstance(h5Item, h5py.Group):
                        itemType = 'Group'
                        self._addAttributes(treeNode, h5Item.attrs)
                        unvisited.append((treeNode, h5Item))
                    elif isinstance(h5Item, h5py.Dataset):
                        itemType = 'Dataset'
                        self._addAttributes(treeNode, h5Item.attrs)
                        spaceId = h5Item.id.get_space()

                        if spaceId.get_simple_extent_type() == h5py.h5s.SCALAR:
                            value = h5Item[()]

                            if isinstance(value, bytes):
                                itemDetails = value.decode()
                            elif isinstance(value, numpy.ndarray):
                                itemDetails = f'STRING = {h5Item.asstr()}'
                            else:
                                stringInfo = h5py.check_string_dtype(value.dtype)

                                if stringInfo:
                                    itemDetails = f'STRING = "{value.decode(stringInfo.encoding)}"'
                                else:
                                    itemDetails = f'SCALAR {value.dtype} = {value}'
                        elif h5Item.size == 1:
                            value = h5Item[()]
                            itemDetails = f'DATASET {value.dtype} = {value}'
                        else:
                            itemDetails = f'{h5Item.shape} {h5Item.dtype}'
                elif isinstance(h5Item, h5py.SoftLink):
                    itemType = 'Soft Link'
                    itemDetails = f'{h5Item.path}'
                elif isinstance(h5Item, h5py.ExternalLink):
                    itemType = 'External Link'
                    itemDetails = f'{h5Item.filename}/{h5Item.path}'
                else:
                    logger.debug(f'Unknown item "{itemName}"')

                treeNode.item_data = [itemName, itemType, itemDetails]

        return rootNode


class H5DiffractionFileReader(DiffractionFileReader):
    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.create_null(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.warning('Unable to find data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=numberOfPatterns,
                        num_patterns_total=numberOfPatterns,
                        pattern_dtype=data.dtype,
                        detector_extent=ImageExtent(detectorWidth, detectorHeight),
                        file_path=filePath,
                    )

                array = H5DiffractionPatternArray(
                    label=filePath.stem,
                    indexes=numpy.arange(numberOfPatterns),
                    filePath=filePath,
                    dataPath=self._dataPath,
                )

                dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset


class H5DiffractionFileWriter(DiffractionFileWriter):
    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        data = numpy.concatenate([array.get_data() for array in dataset])

        with h5py.File(filePath, 'w') as h5File:
            h5File.create_dataset(self._dataPath, data=data, compression='gzip')


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(dataPath='/entry/data/data'),
        simple_name='APS_HXN',
        display_name='CNM/APS HXN Diffraction Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(dataPath='/entry/measurement/Eiger/data'),
        simple_name='MAX_IV_NanoMax',
        display_name='MAX IV NanoMax Diffraction Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(dataPath='/dp'),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Diffraction Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_writers.register_plugin(
        H5DiffractionFileWriter(dataPath='/dp'),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Diffraction Files (*.h5 *.hdf5)',
    )
