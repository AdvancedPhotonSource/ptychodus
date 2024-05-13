from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.xrf import (ElementMap, FluorescenceDataset, FluorescenceFileReader,
                               FluorescenceFileWriter)
from ptychodus.api.typing import RealArrayType

logger = logging.getLogger(__name__)


class FluorescenceDatasetReader(FluorescenceFileReader):  # FIXME

    def read(cls, file_path: Path) -> FluorescenceDataset:
        logger.debug(f'Reading fluorescence data from \"{file_path}\"')
        element_maps: dict[str, ElementMap] = dict()
        counts_per_second_path = str()
        channel_names_path = str()

        with h5py.File(file_path, 'r') as h5file:
            # try to see if v10 layout, Non Negative Lease squares fitting tech was used
            h5_counts_per_second = h5file['/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec']
            h5_channel_names = h5file['/MAPS/XRF_Analyzed/NNLS/Channel_Names']

            if h5_counts_per_second is None:
                # try to see if v10 layout, iterative matrix fitting tech was used
                h5_counts_per_second = h5file['/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec']
                h5_channel_names = h5file['/MAPS/XRF_Analyzed/Fitted/Channel_Names']

                if h5_counts_per_second is None:
                    # try to see if was saved in v9 layout
                    h5_counts_per_second = h5file['/MAPS/XRF_fits']
                    h5_channel_names = h5file['/MAPS/channel_names']

            if h5_counts_per_second is not None:
                # Counts_Per_Sec is an N x H x W
                # where N is number of elements, use channel_names to find what element index
                counts_per_second = h5_counts_per_second[...]
                channel_names = h5_channel_names[...]

                for bname, cps in zip(channel_names, counts_per_second):
                    string_info = h5py.check_string_dtype(bname.dtype)
                    name = bname.decode(string_info.encoding)
                    element_maps[name] = ElementMap(cps)

            counts_per_second_path = h5_counts_per_second.name
            channel_names_path = h5_channel_names.name

        return FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=counts_per_second_path,
            channel_names_path=channel_names_path,
        )


class FluorescenceDatasetWriter(FluorescenceFileWriter):  # FIXME

    @staticmethod
    def _split_path(data_path: str) -> tuple[str, str]:
        parts = data_path.split('/')
        return '/'.join(parts[:-1]), parts[-1]

    def write(self, file_path: Path, xrf: FluorescenceDataset) -> None:
        channel_names: list[str] = list()
        counts_per_sec: list[RealArrayType] = list()

        for ch, emap in xrf.element_maps.items():
            channel_names.append(ch)
            counts_per_sec.append(emap.counts_per_second)

        cps_group_path, cps_dataset_name = self._split_path(xrf.counts_per_second_path)
        ch_group_path, ch_dataset_name = self._split_path(xrf.channel_names_path)

        logger.info(f'Writing element maps to \"{file_path}\"')

        with h5py.File(file_path, 'w') as h5file:
            cps_group = h5file.require_group(cps_group_path)
            cps_group.create_dataset(cps_dataset_name, data=numpy.stack(counts_per_sec))
            ch_group = h5file.require_group(ch_group_path)
            ch_group.create_dataset(ch_dataset_name, data=channel_names, dtype='S256')

    def write_npz(self, file_path: Path, xrf: FluorescenceDataset) -> None:
        element_maps = {name: emap.counts_per_second for name, emap in xrf.element_maps.items()}
        numpy.savez(file_path, **element_maps)
