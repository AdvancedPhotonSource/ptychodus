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
        self, label: str, indexes: PatternIndexesType, file_path: Path, data_path: str
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._file_path = file_path
        self._data_path = data_path

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        with h5py.File(self._file_path, 'r') as h5_file:
            try:
                item = h5_file[self._data_path]
            except KeyError:
                raise ValueError(f'Symlink {self._file_path}:{self._data_path} is broken!')
            else:
                if not isinstance(item, h5py.Dataset):
                    raise ValueError(
                        f'Symlink {self._file_path}:{self._data_path} is not a dataset!'
                    )

            data = item[()]

        return data


class H5DiffractionFileTreeBuilder:
    def _add_attributes(
        self, tree_node: SimpleTreeNode, attribute_manager: h5py.AttributeManager
    ) -> None:
        for name, value in attribute_manager.items():
            if isinstance(value, str):
                item_details = f'STRING = "{value}"'
            elif isinstance(value, h5py.Empty):
                logger.debug(f'Skipping empty attribute {name}.')
            elif isinstance(value, numpy.ndarray):
                item_details = f'ARRAY = {value}'
            else:
                string_info = h5py.check_string_dtype(value.dtype)

                if string_info:
                    item_details = f'STRING = "{value.decode(string_info.encoding)}"'
                else:
                    item_details = f'SCALAR {value.dtype} = {value}'

            tree_node.create_child([str(name), 'Attribute', item_details])

    def create_root_node(self) -> SimpleTreeNode:
        return SimpleTreeNode.create_root(['Name', 'Type', 'Details'])

    def build(self, h5_file: h5py.File) -> SimpleTreeNode:
        root_node = self.create_root_node()
        unvisited = [(root_node, h5_file)]

        while unvisited:
            parent_item, h5_group = unvisited.pop()

            for item_name in h5_group:
                item_type = 'Unknown'
                item_details = ''
                h5_item = h5_group.get(item_name, getlink=True)

                tree_node = parent_item.create_child(list())

                if isinstance(h5_item, h5py.HardLink):
                    item_type = 'Hard Link'
                    h5_item = h5_group.get(item_name, getlink=False)

                    if isinstance(h5_item, h5py.Group):
                        item_type = 'Group'
                        self._add_attributes(tree_node, h5_item.attrs)
                        unvisited.append((tree_node, h5_item))
                    elif isinstance(h5_item, h5py.Dataset):
                        item_type = 'Dataset'
                        self._add_attributes(tree_node, h5_item.attrs)
                        space_id = h5_item.id.get_space()

                        if space_id.get_simple_extent_type() == h5py.h5s.SCALAR:
                            value = h5_item[()]

                            if isinstance(value, bytes):
                                item_details = value.decode()
                            elif isinstance(value, numpy.ndarray):
                                item_details = f'STRING = {h5_item.asstr()}'
                            else:
                                string_info = h5py.check_string_dtype(value.dtype)

                                if string_info:
                                    item_details = (
                                        f'STRING = "{value.decode(string_info.encoding)}"'
                                    )
                                else:
                                    item_details = f'SCALAR {value.dtype} = {value}'
                        elif h5_item.size == 1:
                            value = h5_item[()]
                            item_details = f'DATASET {value.dtype} = {value}'
                        else:
                            item_details = f'{h5_item.shape} {h5_item.dtype}'
                elif isinstance(h5_item, h5py.SoftLink):
                    item_type = 'Soft Link'
                    item_details = f'{h5_item.path}'
                elif isinstance(h5_item, h5py.ExternalLink):
                    item_type = 'External Link'
                    item_details = f'{h5_item.filename}/{h5_item.path}'
                else:
                    logger.debug(f'Unknown item "{item_name}"')

                tree_node.item_data = [item_name, item_type, item_details]

        return root_node


class H5DiffractionFileReader(DiffractionFileReader):
    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5_file:
                metadata = DiffractionMetadata.create_null(file_path)
                contents_tree = self._tree_builder.build(h5_file)

                try:
                    data = h5_file[self._data_path]
                except KeyError:
                    logger.warning('Unable to find data.')
                    return dataset

                num_patterns, detector_height, detector_width = data.shape

                metadata = DiffractionMetadata(
                    num_patterns_per_array=num_patterns,
                    num_patterns_total=num_patterns,
                    pattern_dtype=data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    file_path=file_path,
                )

                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=file_path,
                    data_path=self._data_path,
                )

                dataset = SimpleDiffractionDataset(metadata, contents_tree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')

        return dataset


class H5DiffractionFileWriter(DiffractionFileWriter):
    def __init__(self, data_path: str) -> None:
        self._data_path = data_path

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        data = numpy.concatenate([array.get_data() for array in dataset])

        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset(self._data_path, data=data, compression='gzip')


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(data_path='/exchange/data'),
        simple_name='APS_CSSI',
        display_name='APS CSSI Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(data_path='/entry/data/data'),
        simple_name='APS_HXN',
        display_name='CNM/APS HXN Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(data_path='/entry/measurement/Eiger/data'),
        simple_name='MAX_IV_NanoMax',
        display_name='MAX IV NanoMax Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(data_path='/dp'),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_writers.register_plugin(
        H5DiffractionFileWriter(data_path='/dp'),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Files (*.h5 *.hdf5)',
    )
