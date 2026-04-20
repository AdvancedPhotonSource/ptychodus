from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import queue
import threading

from ptychodus.api.io import load_product
from ptychodus.api.product import Product

from ..product import ProductAPI
from ..task_manager import ForegroundTask
from .iri import IRIComputeClient, JobSpecification
from .tasks import compute_task, transfer_task
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .status import GenesisStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertProductTask:
    product_api: ProductAPI
    product: Product

    def __call__(self) -> None:
        self.product_api.insert_product(self.product)


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
        label: str,
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
        self._label = label
        self._load_product_path = load_product_path

        self._start_time = datetime.now()
        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=None,
                status='Waiting',
                action='Workflow',
            )
        )

    def __call__(self) -> ForegroundTask | None:
        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=None,
                status='Running',
                action='Workflow',
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
            ):
                self._status_q.put(status)
        except Exception:
            logger.exception('Remote-to-local transfer failed!')
            return None

        result: ForegroundTask | None = None

        # Step 4: Load data product (optional)
        if self._load_product_path is not None:
            logger.info('Workflow step 4: loading reconstructed product...')

            try:
                product = load_product(self._load_product_path)
            except Exception:
                logger.exception('Failed to load reconstructed product!')
            else:
                result = InsertProductTask(self._product_api, product)

        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=datetime.now(),
                status='Completed',
                action='Workflow',
            )
        )

        logger.info('Workflow completed successfully.')
        return result
