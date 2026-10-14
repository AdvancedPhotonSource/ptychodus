from pathlib import Path
import logging
import re

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI
from ptychodus.plugins.aps31id_lamni._scan_table import read_aps31ide_scan_table

logger = logging.getLogger(__name__)


class PtychodusAutoloadProductFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return True

    def get_watch_file_pattern(self) -> str:
        # Match only products landing in an 'output/' subdirectory — remote
        # workflows (Globus, Genesis) return results under
        # <workflow_root>/output/product.h5. Path.match matches from the right,
        # so this rejects input-side product.h5 files.
        return 'output/product.h5'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        api.load_product(file_path)


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
        diffraction_api = api.load_diffraction_data(diffraction_file_path)
        product_api = api.create_product(f'scan{scan_id}', diffraction=diffraction_api)
        product_api.load_probe_positions(file_path)
        product_api.generate_probe()
        product_api.generate_object()
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

            if digits == 0:
                diffraction_api = api.load_diffraction_data(diffraction_file_path)
                product_api = api.create_product(f'scan_{scan_id}', diffraction=diffraction_api)
                product_api.load_probe_positions(file_path)
                product_api.generate_probe()
                product_api.generate_object()
                product_api.reconstruct_remote()


class APS31IDEFileBasedWorkflow(FileBasedWorkflow):
    @property
    def is_watch_recursive(self) -> bool:
        return True

    def get_watch_file_pattern(self) -> str:
        return '*.h5'

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        # eiger_4/<block>/S<NNNNN>/<stem>.h5, so the detector directory is three levels up
        # and the data directory four. The master file indexes the series rather than
        # holding it.
        if file_path.parents[2].name != 'eiger_4' or '_master_' in file_path.name:
            return

        data_dir = file_path.parents[3]
        scan_num = int(re.findall(r'\d+', file_path.stem)[0])
        scan_file = data_dir / 'scan_positions' / f'scan_{scan_num:05d}.dat'
        scan_numbers_file = data_dir / 'dat-files' / 'tomography_scannumbers.txt'

        record = next(
            (r for r in read_aps31ide_scan_table(scan_numbers_file) if r.scan_no == scan_num),
            None,
        )

        if record is None:
            logger.warning(f'Failed to locate metadata for {scan_num}!')
        else:
            product_name = f'scan{scan_num:05d}_' + record.label
            diffraction_api = api.load_diffraction_data(file_path)
            input_product_api = api.create_product(
                product_name,
                comments=str(record),
                tomography_angle_deg=record.encoder_angle_deg,
                diffraction=diffraction_api,
            )
            input_product_api.load_probe_positions(scan_file)
            input_product_api.generate_probe()
            input_product_api.generate_object()
            # TODO would prefer to write instructions and submit to queue
            output_dir = (
                data_dir.parent / 'analysis' / 'ptychodus' / record.label / f'S{scan_num:05d}'
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            input_product_api.reconstruct_local(output_product_file=output_dir / 'product.h5')


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
        simple_name='APS_26IDC',
        display_name='APS 26-ID-C',
    )
    registry.file_based_workflows.register_plugin(
        APS31IDEFileBasedWorkflow(),
        simple_name='APS_31IDE',
        display_name='APS 31-ID-E',
    )
