from dataclasses import dataclass
from pathlib import Path
import csv
import logging
import re

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

logger = logging.getLogger(__name__)


class PtychodusAutoloadProductFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return True

    def get_watch_file_pattern(self) -> str:
        return 'product-out.h5'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        api.open_product(file_path)


class APS2IDFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return False

    def get_watch_file_pattern(self) -> str:
        return '*.csv'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        scan_name = file_path.stem
        scan_id = int(re.findall(r'\d+', scan_name)[-1])

        diffraction_file_path = file_path.parents[1] / 'raw_data' / f'scan{scan_id}_master.h5'
        api.open_patterns(diffraction_file_path)
        product_api = api.create_product(f'scan{scan_id}')
        product_api.open_scan(file_path)
        product_api.build_probe()
        product_api.build_object()
        product_api.reconstruct_remote()


class APS26IDFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return False

    def get_watch_file_pattern(self) -> str:
        return '*.mda'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        scan_name = file_path.stem
        scan_id = int(re.findall(r'\d+', scan_name)[-1])

        diffraction_dir_path = file_path.parents[1] / 'h5'

        for diffraction_file_path in diffraction_dir_path.glob(f'scan_{scan_id}_*.h5'):
            digits = int(re.findall(r'\d+', diffraction_file_path.stem)[-1])

            if digits != 0:
                break

        api.open_patterns(diffraction_file_path)
        product_api = api.create_product(f'scan_{scan_id}')
        product_api.open_scan(file_path)
        product_api.build_probe()
        product_api.build_object()
        product_api.reconstruct_remote()


@dataclass(frozen=True)
class APS31IDEMetadata:
    scan_no: int
    golden_angle: str
    encoder_angle: str
    measurement_id: str
    subtomo_no: str
    detector_position: str
    label: str

    def __str__(self) -> str:
        return f"""scan_no={self.scan_no}
        golden_angle={self.golden_angle}
        encoder_angle={self.encoder_angle}
        measurement_id={self.measurement_id}
        subtomo_no={self.subtomo_no}
        detector_position={self.detector_position}
        label={self.label}
        """


class APS31IDEFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return True

    def get_watch_file_pattern(self) -> str:
        return '*.h5'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        experiment_dir = file_path.parents[3]
        scan_num = int(re.findall(r'\d+', file_path.stem)[0])
        scan_file = experiment_dir / 'scan_positions' / f'scan_{scan_num:05d}.dat'
        scan_numbers_file = experiment_dir / 'dat-files' / 'tomography_scannumbers.txt'
        metadata: APS31IDEMetadata | None = None

        with scan_numbers_file.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')

            for row in csv_reader:
                if row[0].startswith('#'):
                    continue

                if len(row) != 7:
                    logger.warning('Unexpected row in tomography_scannumbers.txt!')
                    logger.debug(row)
                    continue

                try:
                    row_no = int(row[0])
                except ValueError:
                    logger.warning('Failed to parse row ID in tomography_scannumbers.txt!')
                    logger.debug(row[0])
                    continue

                if row_no == scan_num:
                    metadata = APS31IDEMetadata(
                        scan_no=scan_num,
                        golden_angle=str(row[1]),
                        encoder_angle=str(row[2]),
                        measurement_id=str(row[3]),
                        subtomo_no=str(row[4]),
                        detector_position=str(row[5]),
                        label=str(row[6]),
                    )
                    break

        if metadata is None:
            logger.warning(f'Failed to locate label for {row_no}!')
        else:
            product_name = f'scan{scan_num:05d}_' + metadata.label
            api.open_patterns(file_path)
            input_product_api = api.create_product(product_name, comments=str(metadata))
            input_product_api.open_scan(scan_file)
            input_product_api.build_probe()
            input_product_api.build_object()
            # TODO would prefer to write instructions and submit to queue
            output_product_api = input_product_api.reconstruct_local()
            output_product_api.save_product(experiment_dir / 'ptychodus' / f'{product_name}.h5')


def register_plugins(registry: PluginRegistry) -> None:
    registry.file_based_workflows.register_plugin(
        PtychodusAutoloadProductFileBasedWorkflow(),
        simple_name='Autoload_Product',
        display_name='Autoload Product',
    )
    registry.file_based_workflows.register_plugin(
        APS2IDFileBasedWorkflow(),
        simple_name='APS_2ID',
        display_name='APS 2-ID',
    )
    registry.file_based_workflows.register_plugin(
        APS26IDFileBasedWorkflow(),
        simple_name='APS_26ID',
        display_name='APS 26-ID',
    )
    registry.file_based_workflows.register_plugin(
        APS31IDEFileBasedWorkflow(),
        simple_name='APS_31IDE',
        display_name='APS 31-ID-E',
    )
