from collections import defaultdict
from pathlib import Path
from typing import Final
import logging

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

logger = logging.getLogger(__name__)


class APS12IDPositionFileReader(PositionFileReader):
    ONE_NANOMETER_M: Final[float] = 1.0e-9

    def read(self, file_path: Path) -> PositionSequence:
        stem_parts = file_path.stem.split('_')
        stem_prefix = '_'.join(stem_parts[:-2])
        logger.debug(f'{stem_prefix=}')
        scan_num = int(stem_parts[-3])
        logger.debug(f'{scan_num=}')

        lines: set[int] = set()
        points: set[int] = set()
        points_per_line: dict[int, set[int]] = defaultdict(set[int])
        file_dict: dict[tuple[int, int], Path] = dict()

        for p in file_path.parent.glob(f'{stem_prefix}_*{file_path.suffix}'):
            stem_parts = p.stem.split('_')
            line = int(stem_parts[-2])
            point = int(stem_parts[-1])

            lines.add(line)
            points.add(point)
            points_per_line[line].add(point)
            file_dict[line, point] = p

        logger.debug(f'{lines=}')
        lines_min = min(lines)
        # lines_max = max(lines)
        # lines_num = lines_max - lines_min + 1
        logger.debug(f'{points=}')
        points_min = min(points)
        points_max = max(points)
        points_num = points_max - points_min + 1

        for line, line_points in points_per_line.items():
            missing_points = points - line_points

            if missing_points:
                logger.warning(f'Line {line} is missing points {missing_points}')

        scan_point_list: list[ScanPoint] = list()

        for (line, point), fp in sorted(file_dict.items()):
            index = (point - points_min) + (line - lines_min) * points_num
            position_data = numpy.genfromtxt(fp)

            for row in position_data:
                scan_point = ScanPoint(
                    index=index,
                    position_x_m=-self.ONE_NANOMETER_M * row[2],
                    position_y_m=+self.ONE_NANOMETER_M * row[1],
                )
                scan_point_list.append(scan_point)

        return PositionSequence(scan_point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        APS12IDPositionFileReader(),
        simple_name='APS_PtychoSAXS',
        display_name='APS 12-ID PtychoSAXS Files (*.dat)',
    )
