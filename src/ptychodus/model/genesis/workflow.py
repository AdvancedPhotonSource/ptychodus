from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import queue
import threading

from ptychodus.api.io import load_product

from ..product import ProductAPI
from ..task_manager import ForegroundTask
from .iri import IRIComputeClient, JobSpecification
from .tasks import compute_task, transfer_task
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .status import GenesisStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadProductTask:
    product_api: ProductAPI
    file_path: Path

    def __call__(self) -> None:
        self.product_api.open_product(self.file_path, file_type='HDF5')


class PtychodusWorkflow:
    def __init__(
        self,
        product_api: ProductAPI,
        transfer_client: AmSCGlobusTransferClient,
        compute_client: IRIComputeClient,
        outbound_transfer_inputs: GlobusTransferInputs,
        compute_resource_id: str,
        job_specification: JobSpecification,
        inbound_transfer_inputs: GlobusTransferInputs,
        stop_event: threading.Event,
        status_q: queue.Queue[GenesisStatus],
        status_interval_s: float,
        flow_label: str,
        load_product_path: Path | None = None,
    ) -> None:
        self._product_api = product_api
        self._transfer_client = transfer_client
        self._compute_client = compute_client
        self._local_to_remote_transfer_inputs = outbound_transfer_inputs
        self._compute_resource_id = compute_resource_id
        self._job_specification = job_specification
        self._remote_to_local_transfer_inputs = inbound_transfer_inputs
        self._stop_event = stop_event
        self._status_q = status_q
        self._status_interval_s = status_interval_s
        self._load_product = load_product
        self._flow_label = flow_label
        self._load_product_path = load_product_path

        self._start_time = datetime.now()
        self._status_q.put(
            GenesisStatus(
                label=self._flow_label,
                action='Starting',
                status='Waiting',
                start_time=self._start_time,
                completion_time=None,
            )
        )

    def __call__(self) -> ForegroundTask | None:
        self._status_q.put(
            GenesisStatus(
                label=self._flow_label,
                action='Starting',
                status='Succeeded',
                start_time=self._start_time,
                completion_time=datetime.now(),
            )
        )

        # Step 1: Transfer input data from local to remote
        logger.info('Workflow step 1: transferring input data from local to remote...')
        try:
            for status in transfer_task(
                self._transfer_client,
                self._local_to_remote_transfer_inputs,
                self._stop_event,
                self._status_interval_s,
                self._flow_label,
            ):
                self._status_q.put(status)
        except Exception:
            logger.exception('Local-to-remote transfer failed!')
            return None

        if self._stop_event.is_set():
            return None

        # Step 2: Submit compute job and wait for completion
        logger.info('Workflow step 2: submitting compute job...')
        try:
            for status in compute_task(
                self._compute_client,
                self._compute_resource_id,
                self._job_specification,
                self._stop_event,
                self._status_interval_s,
                self._flow_label,
            ):
                self._status_q.put(status)
        except Exception:
            logger.exception('Compute job failed!')
            return None

        if self._stop_event.is_set():
            return None

        # Step 3: Transfer output data from remote to local
        logger.info('Workflow step 3: transferring output data from remote to local...')
        try:
            for status in transfer_task(
                self._transfer_client,
                self._remote_to_local_transfer_inputs,
                self._stop_event,
                self._status_interval_s,
                self._flow_label,
            ):
                self._status_q.put(status)
        except Exception:
            logger.exception('Remote-to-local transfer failed!')
            return None

        result: ForegroundTask | None = None
        finishing_start_time = datetime.now()

        # Step 4: Load data product (optional)
        if self._load_product_path is not None:
            logger.info('Workflow step 4: loading reconstructed product...')

            parent = self._load_product_path.parent
            stem = self._load_product_path.stem
            suffix = self._load_product_path.suffix

            best_path = max(
                (
                    path
                    for path in parent.glob(f'{stem}.*{suffix}')
                    if path.stem[len(stem) + 1 :].isdigit()
                ),
                key=lambda p: int(p.stem[len(stem) + 1 :]),
                default=None,
            )

            if best_path is None:
                logger.warning('No epoch-stamped product files found; falling back to base path.')
                best_path = self._load_product_path

            result = LoadProductTask(self._product_api, best_path)

        self._status_q.put(
            GenesisStatus(
                label=self._flow_label,
                action='Finishing',
                status='Succeeded',
                start_time=finishing_start_time,
                completion_time=datetime.now(),
            )
        )

        logger.info('Workflow completed successfully.')
        return result
